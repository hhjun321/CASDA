"""
Vanilla SD v1.5 Baseline Generation Script
===========================================

ControlNet 없이 SD v1.5 text-to-image만 사용하는 generative baseline 생성 스크립트.

Review-2 대응:
  - Comment 5 (Insufficient baseline comparison): generative/diffusion-based baseline 추가
  - Comment 7 (ControlNet conditioning ablation): ControlNet conditioning 컴포넌트 제거 variant

출력 포맷은 test_controlnet.py와 동일하게 유지하여 compose_casda_images.py 재사용 가능.

Usage (Colab):
    !python $SCRIPTS/run_vanilla_sd.py \\
        --jsonl_path $CN_DATASET/train.jsonl \\
        --output_dir $VANILLA_AUG_IMAGES \\
        --num_inference_steps 30 \\
        --guidance_scale 7.5 \\
        --num_images_per_class '{"1":2,"2":10,"3":1,"4":2}' \\
        --seed 42
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

import re
_CLASS_PATTERN = re.compile(r'class\s*(\d+)', re.IGNORECASE)


# =============================================================================
# Pipeline Loading
# =============================================================================

def load_pipeline(args, device):
    """SD v1.5 text-to-image 파이프라인 로드 (ControlNet 없음)."""
    model_name = args.pretrained_model_name_or_path
    logger.info(f"Loading SD v1.5 pipeline: {model_name}")

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        safety_checker=None,
    )

    pipeline.scheduler = UniPCMultistepScheduler.from_config(
        pipeline.scheduler.config
    )

    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    if device.type == "cuda":
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            logger.info("xformers memory efficient attention enabled")
        except Exception:
            logger.info("xformers not available, using default attention")

    logger.info("Pipeline loaded successfully")
    return pipeline


# =============================================================================
# Image Generation
# =============================================================================

def generate_single(
    pipeline, prompt, negative_prompt,
    num_inference_steps, guidance_scale, seed, device,
    num_images=1,
    grayscale_postprocess=True,
    resolution=512,
):
    """SD v1.5 text-to-image 생성. hint 이미지를 conditioning에 사용하지 않음.

    grayscale_postprocess: True이면 SD 1.5 VAE RGB 아티팩트 제거를 위해
        grayscale 변환 후 RGB 3채널로 복제. test_controlnet.py와 동일한 처리.
    """
    gen_h = gen_w = resolution
    results = []

    for i in range(num_images):
        gen = torch.Generator(device=device).manual_seed(seed + i)

        with torch.autocast(str(device)):
            output = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=gen_h,
                width=gen_w,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=gen,
            )

        image = output.images[0]

        if grayscale_postprocess:
            image = image.convert("L").convert("RGB")

        results.append(image)

    return results


# =============================================================================
# Batch Generation (generate_from_jsonl 상당)
# =============================================================================

def generate_from_jsonl(pipeline, args, device):
    """train.jsonl의 모든 샘플에 대해 SD text-to-image 생성.

    출력 구조 및 generation_summary.json 포맷은 test_controlnet.py와 동일.
    hint는 conditioning에 미사용이나, compose 단계의 마스크 추출을 위해
    hint_path를 summary에 그대로 기록.
    """
    jsonl_path = Path(args.jsonl_path)
    output_dir = Path(args.output_dir)

    generated_dir = output_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    logger.info(f"Loaded {len(samples)} samples from {jsonl_path}")

    # 클래스별 생성 수 파싱
    class_num_images = None
    if args.num_images_per_class:
        try:
            class_num_images = json.loads(args.num_images_per_class)
            class_num_images = {str(k): int(v) for k, v in class_num_images.items()}
            logger.info(f"Class-aware multi-generation: {class_num_images}")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Invalid --num_images_per_class format: {e}")
            class_num_images = None

    results_summary = []

    for idx, sample in enumerate(tqdm(samples, desc="Generating")):
        prompt = sample.get("prompt", "")
        negative_prompt = sample.get(
            "negative_prompt",
            "blurry, low quality, artifacts, noise, distorted"
        )
        hint_path_str = sample.get("hint", "")
        if not hint_path_str:
            logger.warning(f"[{idx}] 'hint' key missing or empty in jsonl sample, skipping")
            continue

        # 클래스별 생성 수 결정
        num_images = args.num_images_per_sample
        if class_num_images:
            match = _CLASS_PATTERN.search(prompt)
            if match:
                cls_id = match.group(1)
                num_images = class_num_images.get(cls_id, args.num_images_per_sample)

        # sample_name 결정 — test_controlnet.py와 동일한 로직
        sample_name = Path(hint_path_str).stem.replace("_hint", "")

        # 이미지 생성 (hint는 conditioning에 미사용)
        generated_images = generate_single(
            pipeline=pipeline,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed + idx,
            device=device,
            num_images=num_images,
            grayscale_postprocess=args.grayscale_postprocess,
            resolution=args.resolution,
        )

        for j, gen_img in enumerate(generated_images):
            save_name = f"{sample_name}_gen{j}.png"
            gen_img.save(generated_dir / save_name)

        results_summary.append({
            "index": idx,
            "sample_name": sample_name,
            "prompt": prompt,
            "hint_path": hint_path_str,  # compose 마스크 추출용 — 생성엔 미사용
            "original_available": False,
            "num_generated": len(generated_images),
        })

    summary_path = output_dir / "generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "total_samples": len(samples),
            "generated": len(results_summary),
            "total_images": sum(r["num_generated"] for r in results_summary),
            "model_path": args.pretrained_model_name_or_path,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "class_num_images": class_num_images,
            "results": results_summary,
        }, f, indent=2)

    total_imgs = sum(r["num_generated"] for r in results_summary)
    logger.info(f"Generation complete: {len(results_summary)}/{len(samples)} samples, "
                f"{total_imgs} total images")
    logger.info(f"Generated images: {generated_dir}")
    logger.info(f"Summary: {summary_path}")


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Vanilla SD v1.5 Baseline Generation (ControlNet 없음)"
    )

    parser.add_argument(
        "--pretrained_model_name_or_path", type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="HuggingFace 모델 ID 또는 로컬 경로",
    )
    parser.add_argument(
        "--jsonl_path", type=str, required=True,
        help="train.jsonl 경로 (batch 생성 입력)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/vanilla_sd",
        help="생성 결과 저장 디렉토리",
    )
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_images_per_sample", type=int, default=1)
    parser.add_argument(
        "--num_images_per_class", type=str, default=None,
        help='클래스별 생성 수 JSON. 예: --num_images_per_class \'{"1":2,"2":10,"3":1,"4":2}\'',
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resolution", type=int, default=512,
        help="생성 해상도 (정사각형). 기본값 512.",
    )
    parser.add_argument(
        "--grayscale_postprocess", action="store_true", default=True,
        help="생성 이미지 grayscale 변환 후 RGB 복제 (SD 1.5 VAE 아티팩트 제거). 기본 활성화.",
    )
    parser.add_argument(
        "--no_grayscale_postprocess", dest="grayscale_postprocess",
        action="store_false",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    pipeline = load_pipeline(args, device)
    generate_from_jsonl(pipeline, args, device)


if __name__ == "__main__":
    main()
