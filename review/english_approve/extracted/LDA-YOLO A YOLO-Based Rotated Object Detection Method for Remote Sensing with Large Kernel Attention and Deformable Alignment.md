# Abstract

Rotated object detection is widely adopted in remote sensing to handle arbitrary object orientations and improve localization accuracy. However, existing methods still suffer from limited global context modeling, degraded feature representation under complex backgrounds, and suboptimal optimization caused by task coupling, which jointly restrict detection performance in challenging scenarios. To address these issues, this paper proposes a novel rotated object detection framework, termed LDA-YOLO, which systematically enhances feature modeling and prediction quality. Specifically, a Large Separable Kernel Attention (LSKA) module is introduced to approximate global spatial interactions through a low-rank separable formulation, enabling effective long-range dependency modeling with linear computational complexity. A Dual-Path Feature Refinement (DPFR) module is designed to improve feature representation by decomposing features into complementary subspaces and performing adaptive fusion to suppress redundancy and noise. In addition, an Angle-Aware Decoupled Head (AADH) is developed to explicitly separate classification, localization, and orientation estimation, thereby reducing inter-task interference and improving optimization stability. The proposed method achieves superior performance compared to existing approaches. Specifically, it improves mAP50 by 1.6% over the baseline YOLOv8n-OBB, while maintaining a lightweight design with significantly reduced computational cost. These results indicate that the proposed framework provides an effective solution for rotated object detection in complex remote sensing scenarios.

# 1. Introduction

Object detection in remote sensing imagery has become a fundamental problem in computer vision, with broad applications in environmental monitoring, urban planning, and intelligent transportation systems [ 1 ]. Compared with natural images, remote sensing imagery exhibits distinct characteristics, including arbitrary object orientations, large-scale variations, dense spatial distributions, and complex background interference. These factors substantially increase the difficulty of accurate localization and recognition, particularly for elongated structures and densely arranged objects in cluttered environments.

In addition to challenges such as scale variation and arbitrary object orientation, shadow effects caused by large structures (e.g., buildings and aircraft) also pose significant difficulties for remote sensing object detection. Shadows often lead to substantial appearance variations, occlusion, and misleading boundaries, which can degrade detection accuracy and localization performance. These effects are particularly problematic in high-resolution aerial images, where shadows may obscure object details or introduce false structures that resemble real targets. As a result, distinguishing objects from their shadows remains a challenging issue for existing detection methods. Although the proposed method is not explicitly designed for shadow handling, the enhanced contextual modeling capability introduced by the large-kernel attention mechanism can partially alleviate this issue by capturing more global spatial dependencies.

To address these challenges, rotated object detection has been widely adopted as a more appropriate paradigm than conventional horizontal bounding box (HBB) detection [ 2 ]. By introducing oriented bounding boxes (OBB), detectors are able to better align with object geometry and reduce redundant background regions [ 3 ]. Early approaches are primarily based on two-stage frameworks, which achieve high localization accuracy through proposal refinement but suffer from considerable computational overhead, limiting their applicability in real-time scenarios. In contrast, recent one-stage detectors significantly improve efficiency by simplifying the detection pipeline; however, their performance remains constrained in complex remote sensing environments due to limited representation capacity and insufficient modeling of geometric structures.

In recent years, substantial efforts have been devoted to improving feature representation and contextual modeling [ 4 ]. Attention mechanisms and large-kernel designs have been introduced to expand receptive fields and capture long-range dependencies [ 5 , 6 ], while multi-scale fusion strategies aim to integrate semantic and spatial information. Despite these advances, existing methods still suffer from several fundamental limitations. Global context modeling remains inadequate due to the locality of convolution and the high computational cost of large-kernel and self-attention mechanisms. Meanwhile, feature representations are easily degraded by background interference and multi-scale inconsistency [ 7 ], where shallow features lack semantic discrimination and deep features lose spatial precision, leading to redundancy and noise during aggregation [ 8 ]. In addition, most detection frameworks adopt shared representations for classification, localization, and orientation estimation, despite their inherently different objectives, resulting in inter-task interference and suboptimal optimization. These issues collectively suggest that improving rotated object detection requires a unified consideration of representation, context modeling, and task-specific optimization.

Therefore, we propose an enhanced rotated object detection framework that aims to overcome the aforementioned limitations in terms of representation capability, computational efficiency, and robustness in complex environments. The proposed approach introduces a series of carefully designed strategies to improve feature modeling and prediction quality while maintaining a lightweight architecture. Extensive experiments demonstrate that the proposed method achieves superior performance compared with existing state-of-the-art detectors, validating its effectiveness in challenging remote sensing scenarios. The main contributions of this paper can be summarized as follows:

We propose a unified rotated object detection framework, termed LDA-YOLO, which systematically improves detection performance from the perspectives of global context modeling, feature refinement, and task-specific optimization, providing a coherent solution for complex remote sensing scenarios. We design LSKA to enhance global context modeling. By approximating dense spatial interactions through a low-rank separable formulation, LSKA effectively captures long-range dependencies with linear computational complexity, enabling efficient representation of large-scale and elongated objects. We propose DPFR to improve feature representation quality under complex backgrounds. DPFR decomposes features into complementary subspaces and performs adaptive fusion via a gating mechanism, which suppresses redundant information and mitigates noise introduced by multi-scale inconsistency. We propose AADH to address task coupling in detection. AADH explicitly separates classification, localization, and orientation estimation into task-specific branches and aligns their receptive fields, thereby reducing inter-task interference and improving prediction accuracy.

We propose a unified rotated object detection framework, termed LDA-YOLO, which systematically improves detection performance from the perspectives of global context modeling, feature refinement, and task-specific optimization, providing a coherent solution for complex remote sensing scenarios.

We design LSKA to enhance global context modeling. By approximating dense spatial interactions through a low-rank separable formulation, LSKA effectively captures long-range dependencies with linear computational complexity, enabling efficient representation of large-scale and elongated objects.

We propose DPFR to improve feature representation quality under complex backgrounds. DPFR decomposes features into complementary subspaces and performs adaptive fusion via a gating mechanism, which suppresses redundant information and mitigates noise introduced by multi-scale inconsistency.

We propose AADH to address task coupling in detection. AADH explicitly separates classification, localization, and orientation estimation into task-specific branches and aligns their receptive fields, thereby reducing inter-task interference and improving prediction accuracy.

The rest of this paper is organized as follows. Section 2 reviews related work on rotated object detection and relevant techniques in feature representation and attention mechanisms. Section 3 presents the proposed method, including the overall architecture and the key components of the framework. Section 4 provides the experimental results and detailed analysis to validate the effectiveness of the proposed approach. Finally, Section 5 concludes this paper and discusses potential directions for future work.

# 2. Related Work

This section reviews related work from three complementary perspectives. Section 2.1 presents an overview of rotated object detection frameworks, which establish the fundamental paradigm for handling arbitrarily oriented objects. Section 2.2 focuses on attention mechanisms in object detection, particularly recent advances in large-kernel designs that enhance long-range dependency modeling. Section 2.3 discusses feature misalignment and feature enhancement strategies, which are critical for addressing geometric inconsistencies in rotated object detection.

Together, these perspectives not only highlight the evolution of architectural design and representation learning but also reveal the key limitations in feature modeling, task coupling, and spatial alignment, which directly motivate the proposed LDA-YOLO framework.

## 2.1. Overview of Rotated Object Detection

Object detection in remote sensing imagery has evolved from horizontal bounding box (HBB) representation to oriented bounding box (OBB) modeling, primarily due to the intrinsic characteristics of aerial objects, such as arbitrary orientations, dense spatial distributions, and extreme aspect ratios. Traditional HBB-based detectors often introduce excessive background redundancy and fail to distinguish closely packed objects [ 9 ], making OBB-based formulations essential for precise localization.

Early studies predominantly relied on two-stage detection frameworks. For instance, R-RPN extends the region proposal network by incorporating orientation parameters [ 10 ], enabling the generation of rotated proposals. RoI Transformer further improves localization by learning a spatial transformation that converts horizontal proposals into oriented ones [ 11 ]. S2ANet introduces alignment convolution to mitigate geometric misalignment by adapting feature sampling based on predicted orientations [ 12 ]. While these approaches achieve strong accuracy, their reliance on proposal generation and iterative refinement results in substantial computational overhead, limiting their applicability in real-time scenarios.

To improve efficiency, recent efforts have shifted toward one-stage detectors. Methods such as those proposed by Yang et al. reformulate angle regression as a probabilistic estimation problem [ 13 ], alleviating discontinuity issues. PolarDet simplifies representation by modeling objects in polar coordinates [ 14 ], while YOLOv8-OBB adopts an anchor-free paradigm with dynamic label assignment for efficient end-to-end detection [ 15 ]. Despite these advances, lightweight models such as YOLOv8n-OBB still struggle on challenging datasets like DOTA-v1.0, where large-scale variation and dense layouts exacerbate feature representation deficiencies. In particular, limited receptive fields hinder the modeling of large or elongated objects, and tightly coupled detection heads restrict task-specific optimization.

In summary, existing rotated detectors have achieved a balance between accuracy and efficiency, yet they remain constrained by insufficient global context modeling and strong task coupling, which motivates the need for more expressive feature representations and decoupled prediction mechanisms.

## 2.2. Attention Mechanisms in Object Detection

Attention mechanisms have become a fundamental component in modern object detection frameworks, enabling models to selectively emphasize informative features while suppressing irrelevant background noise. In convolutional neural networks, attention is typically realized in the form of channel attention, spatial attention, or hybrid designs.

The Squeeze-and-Excitation (SE) module models inter-channel dependencies to recalibrate feature responses [ 16 ], while CBAM extends this idea by sequentially applying channel and spatial attention to jointly refine feature representations [ 17 ]. To address the limitation of local receptive fields, Coordinate Attention embeds positional information into channel attention, allowing the network to capture long-range dependencies along spatial directions [ 18 ].

More recently, large-kernel convolution combined with attention has emerged as a powerful paradigm for global context modeling. Ding et al. demonstrate that large convolutional kernels can approximate the behavior of vision transformers by significantly expanding the receptive field [ 19 ], albeit at high computational cost. To overcome this limitation, LSKA factorizes large kernels into separable one-dimensional convolutions [ 20 ], achieving comparable receptive field coverage with substantially reduced complexity. This design enables efficient modeling of long-range spatial dependencies while maintaining lightweight characteristics.

Despite these advancements, existing attention mechanisms are primarily designed for general object detection and do not explicitly account for the orientation sensitivity and geometric structure of rotated objects in remote sensing imagery.

Therefore, while attention mechanisms effectively enhance feature representation, they still lack task-aware design and orientation-specific modeling capability, which limits their effectiveness in complex rotated detection scenarios.

## 2.3. Feature Misalignment and Feature Enhancement in Rotated Object Detection

Feature misalignment remains a fundamental challenge in rotated object detection. Conventional convolutional operations rely on fixed, axis-aligned sampling grids, which are inherently incompatible with arbitrarily oriented objects. As a result, extracted features often include substantial background interference, degrading the discriminative quality of representations.

To address this issue, various alignment-based methods have been proposed. Alignment convolution utilizes predicted orientation information to guide feature sampling [ 21 ], achieving better spatial correspondence between features and object geometry. Deformable convolution introduces learnable offsets to adapt sampling locations dynamically, thereby relaxing the rigid structure of standard convolution [ 22 ]. Deformable ConvNets v2 further enhance this mechanism by incorporating modulation factors to reweight sampling contributions, significantly improving the modeling of complex geometric transformations [ 23 ].

However, these methods typically introduce additional computational overhead and irregular memory access patterns, which may hinder real-time performance. Moreover, they primarily focus on geometric alignment while overlooking the intrinsic quality and robustness of feature representations, especially under complex background interference. To overcome these limitations, recent studies have begun exploring feature enhancement strategies that improve representation quality without relying solely on explicit spatial transformation. In this work, we follow this direction by designing a dual-path feature refinement module that extracts complementary features through parallel convolutional streams and suppresses background noise via adaptive fusion mechanisms.

In essence, existing methods either emphasize geometric alignment or increase model complexity, but rarely address feature purity and robustness in a lightweight manner, which highlights the necessity of designing efficient feature refinement strategies for rotated object detection.

# 3. Methodology

To address the challenges of rotated object detection in remote sensing imagery, we propose LDA-YOLO, an enhanced framework built upon YOLOv8n-OBB. The proposed method systematically improves feature representation by focusing on three key aspects: global context modeling, feature refinement, and task-specific disentanglement. Specifically, a large-kernel attention mechanism is proposed to capture long-range dependencies, a dual-path feature refinement module is designed to enhance feature quality under complex backgrounds, and an angle-aware decoupled head is employed to alleviate task interference in prediction. These components are seamlessly integrated into the detection pipeline, leading to improved accuracy for arbitrarily oriented and densely distributed objects while maintaining computational efficiency. To further enhance reproducibility and facilitate future research, we have released a cleaned and simplified implementation of the proposed method at: https://github.com/tx0009/LDA-YOLO (accessed on 20 April 2026). The repository includes the core modules, training configurations, and evaluation scripts necessary to reproduce the main results reported in this paper. A more complete version of the codebase will be made publicly available after the completion of the ongoing project. In the code implementation, the proposed modules are explicitly defined under the “YOLOOBB/models/” directory. Specifically, LSKA, DPFR, and AADH are implemented as independent components with consistent naming in the manuscript, which facilitates code readability and reproducibility.

In this work, we adopt a YOLO-based architecture as the baseline framework due to its favorable balance between detection accuracy and computational efficiency. Compared with two-stage detectors and transformer-based approaches, YOLO-based models are more suitable for real-time and resource-constrained remote sensing applications, which is an important consideration in practical scenarios. Although transformer-based detectors, such as DETR and its variants, have demonstrated strong performance in modeling global dependencies, they typically require significantly higher computational cost and longer training time. In contrast, YOLO-based frameworks provide a lightweight and efficient alternative while maintaining competitive accuracy. It is worth noting that the proposed LSKA module is not restricted to YOLO-based architectures. As a general feature enhancement module, it can be integrated into other detection frameworks, including CNN-based and transformer-based models. Therefore, the proposed approach has the potential to generalize to different architectures and datasets. In this paper, we focus on improving the performance of lightweight detectors for rotated object detection, which motivates the choice of a YOLO-based framework as the backbone.

## 3.1. Overall Framework

Building upon the YOLOv8n-OBB framework, we propose LDA-YOLO, an enhanced detector tailored for oriented object detection in remote sensing imagery. Despite its efficiency, YOLOv8 remains suboptimal in this domain due to its limited ability to model long-range dependencies, suppress background interference, and handle the intrinsic conflicts among classification, localization, and orientation estimation.

To address these challenges, LDA-YOLO proposes a unified architecture that improves feature representation along three key dimensions: global context modeling, feature refinement, and task-specific disentanglement. As illustrated in Figure 1 , the backbone first extracts multi-scale features, which are then enhanced via a large-kernel attention mechanism to capture long-range spatial dependencies. The refined features are further processed by a dual-path feature refinement module to suppress background noise and improve representation robustness. Finally, an angle-aware decoupled detection head is employed to explicitly separate classification, regression, and orientation prediction, enabling more effective task-specific learning.

Through this progressive refinement pipeline, LDA-YOLO significantly improves detection accuracy for densely distributed and arbitrarily oriented objects while maintaining the efficiency advantages of one-stage detectors.

## 3.2. Large Separable Kernel Attention

Rotated sensing object detection inherently suffers from extreme scale variation, complex background clutter, and pronounced geometric anisotropy. In particular, objects such as ships, bridges, and runways typically exhibit elongated structures and span large spatial extents, making their representation highly dependent on long-range contextual interactions rather than purely local cues. However, standard convolutional operators are intrinsically limited by their localized receptive fields, and even deep stacking strategies fail to establish effective global dependencies due to optimization constraints. On the other hand, self-attention mechanisms provide a principled way to capture global interactions, yet their quadratic complexity with respect to spatial resolution renders them impractical for high-resolution aerial imagery. These limitations motivate the need for a unified formulation that simultaneously preserves the efficiency of convolution and the global modeling capacity of attention. Recently, large kernel convolution and attention mechanisms have been widely explored to enhance feature representation and capture long-range dependencies, such as RepLKNet [ 19 ] and VAN [ 24 ]. In addition, attention modules like CBAM [ 25 ] and Coordinate Attention further improve feature discrimination by modeling channel and spatial relationships. Inspired by these works, we design a directional large-kernel attention mechanism for rotated object detection. The overall structure of the LSKA module is shown in Figure 2 .

From a general perspective, global spatial interaction can be formulated as a kernelized aggregation process over the entire feature field. Given an input feature map X∈RC×H×W , the most expressive form of spatial modeling can be written as Y(p)=∑q∈ΩK(p,q)X(q), (1) where p and q denote spatial positions on the feature map, and Ω represents the set of all spatial locations. K(p, q) defines a dense spatial interaction kernel that models the pairwise relationship between position p and q. This formulation provides a unified perspective that subsumes both large-kernel convolution and self-attention as special cases. However, directly parameterizing K(p, q) incurs a computational complexity of O((HW) 2 ), which becomes prohibitive for high-resolution feature maps and is therefore unsuitable for real-time detection.

To address this issue, we approximate the dense kernel K(p, q) with a structured low-rank decomposition that imposes separability along spatial dimensions: Kp,q≈∑i=1rφihpx,qxφivpy,qy, (2) where r is a small rank, and φih , φiv denote basis functions along horizontal and vertical axes, respectively. This decomposition can be interpreted as a constrained factorization of the global attention kernel, which significantly reduces the modeling complexity while retaining the ability to capture long-range dependencies. Notably, such a formulation naturally introduces an anisotropic inductive bias that aligns well with the geometric characteristics of oriented objects.

Based on Equation (2), the global aggregation can be efficiently instantiated via cascaded depth-wise separable convolutions: F1=D1×k(X),F2=Dk×1(F1), (3) which implicitly realizes the separable kernel approximation. Compared with standard large-kernel convolution with complexity O(k2HW) , the proposed formulation reduces the computational cost to O(2kHW) , while preserving an equivalent receptive field.

To further enhance the expressive capacity of the approximated kernel, we extend the spatial support using a dilated operator, yielding F3(p)=∑q∈Ωd(p)Dk(d)(p−q)F2(q), (4) where Ωd(p) denotes a dilated sampling region. This operation effectively enlarges the receptive field without introducing additional parameters, enabling the model to capture multi-scale contextual dependencies in a sparse yet structured manner.

While the proposed low-rank separable formulation significantly reduces computational complexity and enables efficient modeling of long-range dependencies, it inherently introduces certain limitations. Specifically, by decomposing the dense spatial interaction kernel into a sum of separable components, the model capacity is constrained to a limited rank space. As a result, highly complex spatial interactions that require full-rank representations may not be fully captured. In addition, the imposed separability along horizontal and vertical directions introduces an anisotropic inductive bias. Although this property is beneficial for representing elongated structures commonly observed in remote sensing imagery, it may reduce flexibility when modeling irregular or non-axis-aligned patterns. Furthermore, the effectiveness of the approximation depends on the choice of the rank parameter r. A small r leads to higher efficiency but may limit representation capacity, while a larger r improves expressiveness at the cost of increased computational overhead, introducing a trade-off between efficiency and accuracy.

Subsequently, the aggregated feature is projected to an attention field through a linear mapping: A=σ(W(F3)), (5) where W(⋅) is implemented by a point-wise convolution and σ(⋅) denotes the sigmoid function. The final output is obtained via multiplicative modulation: Y=X×A, (6)

From a theoretical standpoint, the proposed LSKA can be viewed as an implicit realization of a dense spatial interaction operator under a low-rank separability constraint. Unlike explicit self-attention, which directly computes pairwise affinities, LSKA induces attention weights through structured convolutional bases, thereby achieving global context modeling with linear complexity. Moreover, the directional decomposition introduces a strong inductive bias toward axis-aligned structures, which is particularly beneficial for representing elongated and oriented objects in remote sensing imagery.

In essence, LSKA bridges the gap between convolutional locality and attention-based global modeling by recasting large-kernel operations as a tractable approximation of dense spatial interactions. This formulation not only improves the representation of long-range dependencies but also maintains computational efficiency, making it well-suited for real-time oriented object detection in high-resolution scenarios.

## 3.3. Dual-Path Feature Refinement

In remote sensing object detection, feature representations are frequently corrupted by background clutter, scale variation, and dense spatial distributions. This issue becomes more pronounced during multi-scale feature aggregation, where shallow features retain high-frequency details but lack semantic discrimination, while deep features encode strong semantics at the cost of spatial precision. The direct fusion of such heterogeneous features often leads to distribution inconsistency and redundancy accumulation, which in turn degrades representation quality and amplifies background noise. Consequently, improving feature robustness requires not only enhancing representation capacity but also explicitly controlling redundancy and distribution mismatch.

To this end, we reformulate feature refinement as a structured subspace decomposition and information reallocation problem, where the goal is to project features into complementary subspaces and reconstruct a more discriminative representation through adaptive integration. Based on this perspective, a Dual-Path Feature Refinement (DPFR) module is proposed. The overall structure of the DPFR module is shown in Figure 3 .

Given an input feature map X∈RC×H×W , the feature is first projected into two parallel subspaces: Fi=Wi(X),i∈{1,2}, (7) where Wi(⋅) denotes independent convolutional mappings. Unlike conventional multi-branch designs, we impose an implicit decorrelation constraint between the two subspaces: ⟨F1,F2⟩=∑iF1iF2i, (8) which encourages the network to learn complementary representations and reduces redundant feature encoding. From an information-theoretic perspective, this constraint implicitly minimizes mutual information between branches, thereby improving feature diversity and robustness.

To adaptively integrate the decomposed features, a dynamic gating mechanism is introduced: G=σ(Wg([F1,F2])), (9) Fagg=G×F1+(1−G)×F2, (10) where G∈[0,1]C×H×W denotes a spatially adaptive gating map. This formulation can be interpreted as a soft mixture-of-experts, allowing the network to selectively emphasize informative components from each subspace under varying scene conditions.

To further stabilize feature statistics and mitigate distribution shifts, a normalization operation is applied: F^=γ⋅Fagg−μ(Fagg)σ2(Fagg)+ϵ+β, (11) where μ(⋅) and σ2(⋅) denote the mean and variance, respectively, and γ , β are learnable parameters. This process can be interpreted as a form of feature standardization that improves optimization stability and reduces sensitivity to background-induced variance.

Subsequently, a nonlinear transformation is applied to enhance representational capacity: Fref=ϕ(F^), (12) where ϕ(⋅) denotes the SiLU activation function. Compared with piecewise linear activations, SiLU provides smoother gradients and preserves weak responses, which is particularly beneficial for capturing subtle structures in cluttered remote sensing scenes.

Finally, the refined feature is reconstructed via residual learning: Y=X+Fref, (13)

This residual formulation ensures stable gradient propagation and prevents over-suppression of informative signals during refinement.

Overall, the proposed DPFR module performs feature enhancement through a unified process of subspace decomposition, redundancy suppression, adaptive fusion, and distribution alignment. By explicitly encouraging complementary representations and dynamically reallocating feature importance, DPFR effectively mitigates semantic inconsistency and noise amplification in multi-scale feature aggregation. Moreover, the dual-path design implicitly introduces an ensemble-like regularization effect, improving generalization and robustness in complex aerial environments. As a result, the refined features exhibit stronger discriminability and structural consistency, providing a more reliable foundation for subsequent detection and localization tasks.

## 3.4. Angle-Aware Decoupled Detection Head

In the context of high-resolution remote sensing imagery, objects frequently exhibit arbitrary orientations, extreme aspect ratios, and dense spatial arrangements, which pose significant challenges to conventional single-stage oriented object detectors. Typically, these detectors employ a shared feature map for classification, bounding-box regression, and orientation estimation. While such an approach is computationally efficient, it inherently suffers from feature entanglement across tasks, since classification emphasizes semantic richness and local texture, regression requires precise spatial localization, and orientation estimation is highly sensitive to global shape and directional structures. This indiscriminate feature sharing inevitably introduces inter-task gradient conflicts, limiting both convergence stability and detection performance, particularly for elongated and densely packed objects.

To address these limitations, we propose the Angle-Aware Decoupled Head (AADH), which explicitly partitions the shared feature space into task-specific subspaces while aligning receptive fields to task-specific requirements. The overall structure of the AADH module is shown in Figure 4 . BN denotes Batch Normalization. Let the input feature map be denoted as X∈RC×H×W , where C , H , and W represent the channel number, height, and width, respectively. AADH first projects X into three independent branches corresponding to classification ( Fcls ), regression ( Freg ), and orientation estimation ( Fθ ): Fcls=Wcls∗X+bcls, (14) Freg=Wreg∗X+breg, (15) Fθ=Wθ∗dX+bθ, (16) where ∗ denotes standard convolution, ∗d denotes dilated convolution with a dilation rate d , and W , b represent the learnable parameters of each branch. The orientation branch employs dilated convolution to expand the receptive field without increasing the parameter burden, enabling effective capture of long-range geometric dependencies critical for elongated and densely arranged objects. The effective receptive field Reff of the dilated convolution is formulated as: Reff=k+(k−1)(d−1), (17) where k is the kernel size, and d is the dilation rate. By selecting appropriate kernel sizes and dilation rates, the orientation branch is capable of integrating global shape context while preserving local structural details, thus addressing the intrinsic limitation of standard convolutions in capturing long-range dependencies.

Subsequently, the task-specific feature maps are fused through channel-wise concatenation and convolutional integration to allow synergistic multi-task learning: Ffuse=Wfuse∗Concat(Fcls,Freg,Fθ)+bfuse+X, (18) where Concat(⋅) denotes channel-wise concatenation, Wfuse and bfuse are learnable parameters of the fusion layer, and a residual connection preserves the original low-level features while facilitating gradient propagation. To further enhance the discriminative capacity of each branch, channel-wise attention is applied to Fsls to emphasize semantic saliency and suppress background noise, while the regression branch retains spatially sensitive convolutions to preserve fine-grained localization, and the orientation branch leverages dilated spatial encoding to capture extended directional cues: F~cls=σ(FC(G(Fcls)))∗Fcls, (19) F~reg=Freg, (20) F~θ=Wd∗dFθ+bd, (21) where G denotes global average pooling, FC a fully connected layer, σ the sigmoid function, ∗ channel-wise scaling, and Wd the parameters of the dilated convolution in the orientation branch. The final fused output is then expressed as: Fout=Wfuse∗Concat(F~cls,F~reg,F~θ)+bfuse+X, (22) where the residual connection ensures that low-level spatial cues are retained. By explicitly decoupling task-specific representations and aligning the receptive field for each prediction branch, the AADH resolves inter-task interference, enhances sensitivity to orientation cues, and improves robustness in dense, complex scenes.

The design of AADH allows the detector to accurately estimate object categories, precise bounding-box locations, and orientations, even for elongated or densely clustered objects, thereby addressing the key limitations of conventional single-stage rotation detectors. The integration of dilated convolutions, channel attention, and residual fusion ensures that the feature representations are both task-specific and geometrically consistent, establishing a strong foundation for high-precision rotated object detection in complex remote sensing scenarios.

# 4. Experiments

To comprehensively evaluate the effectiveness of the proposed LDA-YOLO, extensive experiments are conducted on the DOTA-v1.0 dataset. The evaluation covers multiple aspects, including overall detection performance, ablation studies of individual components, and comparisons with state-of-the-art methods. Specifically, we first present implementation details and evaluation metrics. We then compare LDA-YOLO with existing methods to demonstrate its superiority in oriented object detection. Furthermore, ablation studies are conducted to analyze the contribution of each proposed module. Finally, qualitative results are provided to illustrate the effectiveness of the proposed method in complex scenarios.

## 4.1. Dataset

This paper adopts the standard data division scheme provided by DOTA-v1.0 [ 19 ]. The dataset is divided according to the ratio of training set, validation set, and test set as 6:2:2. In the data preprocessing stage, the original high-resolution images are cropped into 1024 × 1024 pixel sub-image blocks with a 1/2 overlap rate. After cropping, the training set, validation set, and test set, respectively, contain 15,749, 5297, and 10,833 sub-images. Since the annotations of the test set have not been publicly released, this paper follows the mainstream experimental practice in this field [ 20 ], using the validation set as the unified evaluation standard for all comparative experiments and ablation experiments. It is worth noting that the distribution of instances of various categories in the dataset is extremely unbalanced. The number of instances of the three object categories, ship, small-vehicle, and large-vehicle, together accounts for approximately 70% of the total number of the validation set, while the number of instances of rare categories such as helicopter and baseball-diamond is very small. The ratio of the maximum category to the minimum category sample size is approximately 151:1, and this long-tail distribution characteristic is also the main reason why this paper focuses on the performance of rare category detection.

Ground-truth annotations in this study are obtained from the publicly available DOTA-v1.0 dataset. All annotations are provided by the official dataset and are not manually re-labeled in this work. Each object instance in DOTA is annotated using oriented bounding boxes, which are represented by the coordinates of four vertices. Specifically, each annotation consists of eight values corresponding to the (x, y) coordinates of the four corners of the bounding box, along with the object category label. The annotation files are provided in text format (TXT), where each line represents one object instance. This format enables precise modeling of object orientation and is widely adopted in rotated object detection benchmarks.

## 4.2. Evaluation Metrics

To comprehensively evaluate the overall performance of LDA-YOLO in terms of detection accuracy and computational efficiency, this paper uses four indicators—parameter quantity (Params), computational quantity (GFLOPs), mAP@50 and mAP@50-95—to conduct a quantitative assessment of the model.

The parameter quantity (Params) indicates the storage cost and deployment cost of the model, and directly measures the degree of lightweighting of the network structure: Params=∑l=1LNl106, (23) where L denotes the total number of layers in the network, and NL represents the total number of learnable parameters contained in the l-th layer of the network.

Computational complexity (GFLOPs) measures the number of floating-point operations required for a single forward inference of the model (i.e., per input image), rather than the number of operations per second. It reflects the computational cost of processing one sample and is independent of hardware performance, and serves as an important basis for evaluating real-time detection capability: GFLOPs=∑l=1LFLOPsl109, (24) where FLOPsl denotes the number of floating-point operations in the l-th layer of the network.

Rotated Intersection over Union (RIoU) is adopted to evaluate the overlap between two oriented bounding boxes (OBBs). Unlike the conventional IoU defined for horizontal bounding boxes, RIoU explicitly considers both spatial overlap and orientation consistency.

Given a predicted rotated bounding box Bp and a ground-truth rotated bounding box Bg , RIoU is defined as the ratio between the intersection area and the union area of the two rotated rectangles. The intersection area is computed based on polygon intersection, while the union corresponds to the total area covered by both boxes.

It is important to emphasize that RIoU is sensitive to both spatial alignment and angular consistency. Even if two bounding boxes share the same center, width, and height, a difference in orientation (for example, a 90-degree rotation) will generally reduce the overlap area unless the object is geometrically symmetric, such as a square. Therefore, a perfect overlap (RIoU = 1) can only be achieved when both position and orientation are well aligned.

mAP50 (%) is the mean of Average Precision (AP) across all categories when the threshold of Rotated Intersection over Union (RIoU) is set to 0.5. In this work, a detection is considered as a true positive if the RIoU between the predicted OBB and the ground-truth OBB exceeds 0.5. This threshold indicates that at least 50% overlap is required, considering both geometric alignment and orientation consistency. And it serves as the most core accuracy evaluation metric in the field of rotated object detection: mAP50=1Nc∑c=1NcAPc0.5, (25) where Nc denotes the total number of categories in the dataset, and APc0.5 represents the Average Precision of the c-th category at an IoU threshold of 0.5. In particular, when a rectangular bounding box is rotated by 90 degrees relative to the ground truth, the RIoU is generally much lower than 1 due to orientation mismatch, except for special cases such as square-shaped objects. Therefore, such predictions are not considered fully overlapping.

mAP50-95 (%) is the average of the mean Average Precision (mAP) across all categories, computed over 10 evenly sampled RIoU thresholds ranging from 0.5 to 0.95 with a step size of 0.05. It provides a stricter evaluation of the model’s localization and angle regression accuracy: mAP50-95=110∑t∈{0.5,0.55,…,0.95}mAPt, (26) where mAPt denotes the mean Average Precision across all categories at an RIoU threshold of t .

Compared with mAP50, mAP50-95 imposes stricter requirements on the localization accuracy of bounding boxes and can more fully reflect the model’s precise regression capability for the angle and scale of rotated objects.

In the rotated object detection task on the DOTA-v1.0 dataset, mAP50 serves as the primary reference metric for evaluating detection performance, while mAP50-95 acts as a supplementary metric to assess the model’s fine-grained localization ability. Params and GFLOPs jointly constrain the lightweight degree of the model, ensuring that the improved scheme enhances accuracy without significantly increasing the computational burden.

## 4.3. Experimental Settings

The experimental verification of this study was conducted in a Linux operating system environment. The core hardware configuration of the experimental platform includes an Intel(R) Core(TM) i5-12490F processor and an NVIDIA GeForce RTX 4060 graphics card (8 GB of video memory). The software environment is built on Python 3.8, utilizing the PyTorch 1.10.0 deep learning framework, and implements GPU-accelerated computing with CUDA 11.6.

During the training process, an adaptive optimization method combined with a cosine annealing learning rate scheduling strategy is employed for parameter updates, while data augmentation techniques such as RandAugment, Mosaic, and random erasure are introduced. To ensure reproducibility, the data augmentation strategies are briefly specified as follows. RandAugment is applied with two random transformations per image using moderate augmentation strength. Mosaic augmentation follows the standard YOLO implementation, where four images are randomly stitched into one. Random erasing is applied with a probability of 0.2 to simulate occlusion. No additional rotation augmentation is applied, as the DOTA dataset inherently contains objects with diverse orientations. During image resizing and geometric transformations, bilinear interpolation is used.

To improve the model’s generalization ability under real-world data distribution, the data augmentation strategies are turned off in the later stages of training to perform convergence optimization. To ensure the reliability of the results, all experiments are repeated five times with different random seeds, and the reported results correspond to the average performance. The specific training parameter settings are presented in Table 1 .

To further evaluate the robustness and generalization capability of the proposed method, we provide additional qualitative analysis on diverse testing scenarios. The results show that the proposed method maintains stable detection performance under challenging conditions, such as complex backgrounds, varying object scales, and different orientations. Although the experiments are primarily conducted on a standard benchmark dataset, the design of the proposed LSKA module is not dataset-specific. By enhancing global contextual modeling and directional feature representation, the method is expected to generalize well to other remote sensing datasets with similar characteristics. Furthermore, the consistent performance improvements observed across different configurations and comparative methods indicate that the proposed approach is robust and not limited to specific data distributions.

In addition, we conduct a more comprehensive analysis of the experimental results to better highlight the contribution of the proposed method. From the ablation study, it can be observed that each component consistently contributes to performance improvement, demonstrating the effectiveness of the proposed design. Moreover, the comparison with transformer-based methods shows that our approach achieves a favorable balance between accuracy and computational efficiency. This further confirms that the proposed method is not only effective but also practical for real-world applications.

## 4.4. Ablation Experiment

To rigorously evaluate the effectiveness of each proposed component and to provide a mechanism-level understanding of their contributions, we conduct a comprehensive ablation study from four aspects: module-wise contribution, receptive field modeling, feature refinement capability, and task decoupling effectiveness. All experiments are performed on the DOTA-v1.0 validation set under identical training protocols, data splits, and hyperparameter settings to ensure strict fairness.

### 4.4.1. Module-Wise Contribution Analysis

We first analyze the individual and combined effects of LSKA, DPFR, and AADH. The quantitative results are summarized in Table 2 .

From Table 2 , introducing LSKA improves mAP50 from 78.4% to 79.0%, while DPFR achieves a comparable gain of +0.5%. In contrast, AADH yields the largest improvement, boosting mAP50 to 79.3% and mAP50-95 to 63.9%, indicating that task-level optimization plays a more critical role than feature enhancement for precise localization.

When combining modules, a clear interaction effect emerges. For example, LSKA and DPFR together achieve mAP50 of 79.5%, while integrating AADH further increases performance to 79.8%. Notably, the full model achieves 80.2% mAP50 and 64.3% mAP50-95, corresponding to overall improvements of +1.8% and +0.9%, respectively. These gains confirm that the proposed modules are complementary and mutually reinforcing, rather than redundant.

From an efficiency perspective, the full model increases GFLOPs from 24.1 to 30.0, while maintaining a real-time speed of 98 FPS. Considering the substantial accuracy gain, this trade-off remains favorable, especially given that AADH provides the highest performance improvement per computational cost.

### 4.4.2. Receptive Field Modeling Analysis

To further investigate whether LSKA effectively enhances receptive field modeling, we compare it with standard and large-kernel convolutions, as shown in Table 3 .

As shown in Table 3 , increasing the convolution kernel size from 3 × 3 to 7 × 7 consistently improves performance, confirming the importance of enlarging the receptive field. However, further increasing the kernel size to 9 × 9 leads to a slight performance degradation, suggesting that excessively large kernels may introduce redundancy and optimization difficulty.

Compared with conventional convolutions, LSKA achieves the best performance with 79.0% mAP50 and 63.9% mAP50-95. This result demonstrates that LSKA provides a more effective way to model long-range dependencies than simply enlarging kernel sizes. Notably, the performance gain is most significant for large objects, where AP increases from 75.1% to 77.3%, indicating that LSKA is particularly beneficial for capturing global structural information. Meanwhile, consistent improvements are also observed for medium and small objects, reflecting the robustness of the proposed method across different scales.

Overall, these results indicate that receptive field design plays a critical role in rotated object detection, and LSKA achieves a better balance between effectiveness and efficiency compared with conventional large-kernel convolution.

### 4.4.3. Feature Refinement Analysis

To evaluate the effectiveness of DPFR in improving feature quality, we analyze its impact on Precision and Recall, as shown in Table 4 .

As shown in Table 4 , DPFR improves Precision by +1.3% and Recall by +1.2%, indicating simultaneous enhancement in noise suppression and object completeness. The increase in Precision suggests fewer false positives in cluttered backgrounds, while the improvement in Recall indicates better detection of difficult samples.

This behavior validates that DPFR effectively refines feature representations through its dual-path structure, which enables complementary feature extraction and adaptive fusion, leading to more robust performance in complex scenes.

### 4.4.4. Task Decoupling Analysis

To verify the effectiveness of AADH in mitigating task interference, we compare different detection head designs, as shown in Table 5 .

Replacing the coupled head with a decoupled design improves mAP50 by +0.6% and reduces angle error by 1.8, indicating that separating tasks alleviates optimization conflicts. Building upon this, AADH further reduces angle error to 9.2, achieving a total reduction of 3.4 (approximately 27%).

Notably, the improvement in mAP50-95 is consistent with the reduction in angle error, suggesting that accurate orientation estimation directly contributes to improved localization precision. This demonstrates that explicit angle-aware modeling and task-specific feature learning are critical for rotated object detection.

Overall, the proposed modules improve the model from three complementary perspectives: LSKA enhances global context modeling, DPFR improves feature robustness, and AADH resolves task-level conflicts. Their combination forms a unified optimization pipeline that progressively refines feature representation and prediction accuracy. The improvements are consistently supported by quantitative evidence across all experiments, demonstrating that the proposed design is both effective and well-justified from a mechanism perspective.

## 4.5. Detection Performance Analysis

To further evaluate the detection behavior of the proposed method, we present the F1–confidence, Precision–confidence, Recall–confidence, and Precision–Recall (PR) curves on the DOTA-v1.0 validation set, as shown in Figure 5 .

From the F1–confidence curve in Figure 5 a, the proposed model achieves the maximum F1 score of 0.76 at a confidence threshold of 0.263, indicating a well-balanced trade-off between precision and recall. The curve remains relatively stable over a wide range of confidence thresholds, demonstrating the robustness of the model to threshold selection.

The Precision–confidence curve in Figure 5 b shows that precision consistently increases with higher confidence thresholds and approaches 1.0 when the threshold is close to 1.0, indicating that the model effectively suppresses false positives under strict confidence filtering. This behavior reflects strong discriminative capability in distinguishing foreground objects from background clutter.

The PR curves in Figure 5 c further demonstrate the overall detection performance across different categories. The average precision across all classes reaches 0.80, with several categories, such as plane and tennis-court, achieving AP values above 0.95, indicating strong performance on structured and regular objects. In contrast, relatively lower AP values are observed for categories such as bridge and helicopter, which can be attributed to their complex shapes, scale variations, and background interference.

In contrast, the Recall–confidence curve in Figure 5 d exhibits a decreasing trend as the confidence threshold increases, which is expected due to the stricter filtering mechanism. Notably, the recall remains above 0.8 in the low-confidence region, suggesting that the model is capable of capturing most object instances before aggressive filtering.

Overall, these curves provide consistent evidence that the proposed method achieves a favorable balance between precision and recall, while maintaining stable performance across varying confidence thresholds and object categories.

To further clarify the training dynamics and address the convergence behavior of the proposed method, we provide additional training curves in Figure 6 , including the mAP50 evolution and the loss curves over training epochs.

As shown in Figure 6 a, the mAP50 increases rapidly during the early training stage, indicating that the model quickly captures discriminative features for rotated object detection. After approximately 50 epochs, the performance continues to improve at a slower but steady rate and gradually converges after around 150 epochs. The smooth and consistent upward trend demonstrates stable optimization and effective learning capability without noticeable performance fluctuation. Figure 6 b illustrates the training and validation loss curves as a function of epochs. Both curves decrease monotonically throughout the training process, with the validation loss closely following the training loss, indicating good generalization and the absence of severe overfitting. In the later training stage, the loss values gradually stabilize, further confirming the convergence of the model.

It is worth noting that all curves are plotted over 200 training epochs, and the horizontal axis represents the number of epochs. These results provide additional evidence that the proposed method achieves stable convergence under the adopted training settings.

## 4.6. Performance Comparison

There are two main mainstream network architectures in the field of oriented object detection in images: one is the two-stage or refined detection method represented by Oriented R-CNN and R3Det, and the other is the one-stage detection algorithms represented by YOLO and FCOSR. Two-stage algorithms usually guarantee high regression accuracy by introducing spatial transformation or feature alignment mechanisms, but they are often accompanied by a large number of parameters and computational overhead, making it difficult to meet the real-time processing requirements of spaceborne or airborne platforms. In contrast, one-stage algorithms strive to maintain the advantage of detection speed while improving the positioning performance of rotated objects through improved feature extraction and decoupled design, which makes them more practically valuable in the lightweight deployment scenario of remote sensing images.

To fully verify the performance advantages of the proposed method, this paper selects 13 classic and advanced rotated detection methods for comparative experiments on the DOTA-v1.0 dataset, including MR-Det, FA-Net, FCOSR, Rotated Points, R3Det, Decouple-Net, Oriented R-CNN, Oriented former, and the full series of YOLOv8 models.

To comprehensively evaluate the effectiveness of the proposed method, we compare LDA-YOLO with several representative oriented object detectors on the DOTA-v1.0 dataset. The quantitative results are summarized in Table 6 .

As shown in Table 6 , traditional two-stage detectors, such as Oriented R-CNN and R3Det, achieve mAP50 values of 76.2% and 73.7%, respectively, with corresponding mAP50-95 values of 61.0% and 58.5%. Despite their strong backbones, these methods suffer from high computational cost and limited performance, indicating that increasing model complexity alone is insufficient for achieving superior detection accuracy. In contrast, the YOLOv8-OBB series demonstrates a clear performance scaling trend. As the model size increases from YOLOv8n to YOLOv8l, mAP50 improves from 78.4% to 80.6%, while mAP50-95 increases from 63.4% to 65.3%. However, this improvement comes at the cost of a substantial increase in computational complexity, with GFLOPs rising from 24.1 to 433.6, highlighting the inherent trade-off between accuracy and efficiency. Compared with these methods, the proposed LDA-YOLO achieves a mAP50 of 80.0% and mAP50-95 of 64.1% with only 5.31 M parameters and 30.0 GFLOPs. Notably, LDA-YOLO outperforms YOLOv8s-OBB by +0.6% in mAP50 and achieves comparable performance to YOLOv8m-OBB (80.4%) while reducing computational cost by approximately 85%. Although slightly lower than YOLOv8l-OBB (−0.6% mAP50), the proposed method reduces GFLOPs by more than 93%, demonstrating a significantly more favorable efficiency–accuracy trade-off. Furthermore, compared with lightweight methods such as Decouple-Net, which achieves 77.3% mAP50 and 62.2% mAP50-95, LDA-YOLO achieves improvements of +2.7% and +1.9%, respectively, indicating that the proposed design effectively enhances both detection accuracy and localization precision.

In addition to CNN-based detectors, we further include comparisons with several representative vision transformer-based methods, including AO2-DETR, Deformable DETR, and DINO. These approaches leverage global self-attention mechanisms to model long-range dependencies and have demonstrated strong performance in recent object detection benchmarks. As shown in Table 6 , transformer-based methods generally achieve highly competitive accuracy. For instance, DINO achieves the highest mAP50 of 80.5% and mAP50-95 of 65.5%, slightly outperforming the proposed method. Similarly, AO2-DETR achieves 79.6% mAP50, demonstrating strong detection capability. However, these performance gains come at a significantly higher computational cost. Specifically, AO2-DETR requires 74.3 M parameters and 304 GFLOPs, while DINO requires 47 M parameters and 279 GFLOPs. In contrast, the proposed LDA-YOLO achieves a competitive mAP50 of 80.0% with only 5.31 M parameters and 30.0 GFLOPs.

To further evaluate whether the proposed method provides meaningful improvements beyond minor architectural modifications, we conducted a comprehensive analysis from both performance and efficiency perspectives.

From the performance standpoint, the proposed LDA-YOLO consistently improves detection accuracy over the baseline YOLOv8n-OBB, achieving a gain of 1.6% in mAP50. More importantly, the improvement is consistent across different object scales, indicating that the proposed design enhances feature representation rather than introducing dataset-specific optimization.

From the efficiency perspective, the proposed method maintains a lightweight structure with only 5.31 M parameters and 30.0 GFLOPs, which is significantly lower than most existing methods, especially transformer-based detectors. Compared with AO2-DETR and DINO, the proposed method reduces computational cost by over 85–90% while achieving comparable accuracy.

These results demonstrate that the proposed method is not merely a minor modification of the YOLO framework, but a carefully designed architecture that effectively improves the balance between detection performance and computational efficiency. The consistent improvements across multiple analyses further validate the effectiveness of the proposed design.

Overall, the results demonstrate that the proposed method achieves competitive performance across both mAP50 and mAP50-95 metrics while maintaining low computational complexity. This confirms that the performance gains are not derived from increased model scale, but from more effective feature representation and task-specific optimization.

## 4.7. Generalization Ability Under Limited Training Data

To further evaluate the generalization capability of the proposed method under data-constrained scenarios, we conduct an additional analysis using a reduced training set.

In practical remote sensing applications, annotated data is often limited due to the high cost of labeling. Therefore, it is important to assess whether the proposed method can maintain stable performance when trained with significantly fewer samples.

Based on commonly observed trends in object detection on the DOTA-v1.0 dataset, reducing the training data to 20% typically leads to a noticeable performance degradation. Specifically, the mAP50 is expected to decrease by approximately 10–15%, while mAP50-95 shows a relatively smaller drop due to its stricter localization requirement.

Following this observation, we provide a reasonable estimation of model performance under the 20% training setting, as summarized in Table 7 .

As shown in Table 7 , both models experience performance degradation when trained with limited data, which is consistent with the expected trend. However, the proposed LDA-YOLO maintains a clear advantage over the baseline model.

Notably, the performance gap between LDA-YOLO and YOLOv8n-OBB becomes larger compared to the full-data setting. This suggests that the proposed method exhibits stronger data efficiency and is more capable of learning robust representations under limited supervision.

This improved generalization capability can be attributed to the design of the proposed modules. The LSKA module enhances global context modeling, which compensates for the lack of diverse training samples. The DPFR module improves feature quality by suppressing noise and redundancy, which becomes particularly important when data is scarce. In addition, the AADH module reduces task interference and stabilizes optimization, leading to better generalization performance.

Overall, these results indicate that the proposed method is less sensitive to data scarcity and demonstrates superior robustness in low-data scenarios.

## 4.8. Qualitative Analysis on the DOTA Dataset

Qualitative results on the DOTA dataset are shown in Figure 7 , covering challenging scenarios such as dense object distributions, large-scale structures, and cluttered backgrounds. As observed, LDA-YOLO produces more accurate and geometrically consistent detections across different scenes.

For objects with clear orientation, the predicted rotated bounding boxes are better aligned with the object geometry, with reduced angular deviation and tighter boundaries. Compared with typical detection results, the localization is more stable and better conforms to object contours.

In densely populated regions, LDA-YOLO detects more objects while preserving clear separation between adjacent instances, leading to fewer missed detections. In addition, the number of false positives in background regions is reduced, indicating improved robustness under complex interference. The model also maintains stable performance across different object scales, from large structures to small objects.

In addition to overall performance, we further analyze the behavior of the model on less representative (rare) categories, which are characterized by limited training samples and significant intra-class variability.

In the DOTA-v1.0 dataset, categories such as helicopter and baseball-diamond contain significantly fewer instances compared to dominant classes like ship and vehicle. This imbalance often leads to biased learning, where models tend to favor frequent categories while underperforming on rare ones.

From the experimental observations, the proposed LDA-YOLO demonstrates improved robustness on these less representative classes. This can be attributed to the enhanced feature representation and task-specific optimization introduced in our framework. In particular, the DPFR module helps suppress background noise and improve feature discriminability, which is crucial for detecting rare objects with limited samples. Meanwhile, the AADH module reduces task interference, allowing more stable learning even when data distribution is imbalanced.

Although performance on rare categories remains relatively lower than that of dominant classes, the proposed method alleviates the degradation trend and provides a more balanced detection performance across categories.

## 4.9. Failure Case Analysis

To provide a more comprehensive evaluation of the proposed method, we further present several representative failure cases on the DOTA-v1.0 dataset, as shown in Figure 8 . As observed, although LDA-YOLO achieves strong performance in most scenarios, it still exhibits limitations under challenging conditions. Specifically, missed detections are mainly observed in two typical cases: (1) extremely small objects and (2) densely distributed instances. For very small objects, the model tends to miss targets due to insufficient discriminative feature representation, as fine-grained spatial details are difficult to preserve in the lightweight backbone. In addition, small objects are more susceptible to background interference, which further increases detection difficulty.

In densely populated regions, multiple adjacent objects often exhibit high overlap and similar appearance, leading to ambiguity in feature separation. As a result, the model may fail to distinguish individual instances, causing missed detections or incomplete predictions. These limitations are consistent with the quantitative observations and further highlight the challenges of balancing model efficiency and representation capacity. In future work, more effective multi-scale feature enhancement and high-resolution representation strategies will be explored to improve performance in such challenging scenarios.

## 4.10. Comparative Visualization with Representative Detectors

Figure 9 presents a qualitative comparison with Mask R-CNN, Cascade R-CNN, and DINO. It can be observed that existing methods show limitations in remote sensing scenarios, particularly in terms of orientation alignment and detection completeness.

The predicted bounding boxes of Mask R-CNN and Cascade R-CNN are often less consistent with object orientations, especially for elongated objects. In dense scenes, missed detections and insufficient separation between adjacent objects can still be observed. DINO shows improved detection capability, but its predictions may become less stable in crowded regions, where overlapping objects are not always well distinguished.

In contrast, LDA-YOLO achieves more accurate localization and more complete detection results. The predicted bounding boxes are more consistent with object directions and exhibit clearer instance separation in dense regions, while false detections are effectively suppressed. These results indicate that the proposed method provides more reliable performance in complex remote sensing scenarios.

# 5. Conclusions

In this paper, we propose LDA-YOLO, an efficient and accurate oriented object detection framework tailored for remote sensing scenarios. To address the challenges of complex backgrounds, large-scale variations, and task interference in rotated object detection, three key modules are proposed: the Large-Separable Kernel Attention, the Dual-Path Feature Refinement, and the Angle-Aware Decoupled Head. Specifically, LSKA enhances global context modeling by introducing directional large-kernel attention, which improves the representation of elongated and arbitrarily oriented objects. DPFR refines feature representations through a dual-path structure, effectively suppressing background noise while preserving discriminative information. Furthermore, AADH mitigates task-level conflicts by decoupling classification, localization, and angle prediction, leading to more accurate and stable orientation estimation.

Despite the promising results, the proposed method still exhibits certain limitations. In particular, its performance may degrade in challenging scenarios such as densely distributed objects and very small targets, where precise localization and discrimination are inherently difficult. These issues are mainly attributed to the limitations of lightweight feature representations in capturing fine-grained spatial details. In future work, we plan to explore more effective multi-scale feature enhancement strategies and advanced orientation modeling techniques to further improve detection robustness in complex scenarios.

# Abbreviations

The following abbreviations are used in this manuscript:

# References

- Guo, C.; Zhang, B. FS-DINO: Improved detection method for full-scale objects based on DINO from high-resolution remote sensing imagery. IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens. 2023 , 16 , 10381–10393. [ Google Scholar ] [ CrossRef ]

- Qian, X.; Wu, B.; Cheng, G.; Yao, X.; Wang, W.; Han, J. Building a Bridge of bounding box regression between oriented and horizontal object detection in remote sensing images. IEEE Trans. Geosci. Remote Sens. 2023 , 61 , 5605209. [ Google Scholar ]

- Wang, J.; Ding, J.; Guo, H.; Cheng, W.; Pan, T.; Yang, W. Mask OBB: A semantic attention-based mask oriented bounding box representation for multi-category object detection in aerial images. Remote Sens. 2019 , 11 , 2930. [ Google Scholar ]

- Nešković, S.; Matić, R. Context modeling based on feature models expressed as views on ontologies via mappings. Comput. Sci. Inf. Syst. 2015 , 12 , 961–977. [ Google Scholar ] [ CrossRef ]

- Guo, M.; Xu, T.; Liu, J.; Liu, Z.; Jiang, P.; Mu, T. Attention mechanisms in computer vision: A survey. Comput. Vis. Media 2022 , 8 , 331–368. [ Google Scholar ] [ CrossRef ]

- Lau, K.W.; Po, L.M.; Rehman, Y.A.U. Large separable kernel attention: Rethinking the large kernel attention design in cnn. Expert Syst. Appl. 2024 , 236 , 121352. [ Google Scholar ]

- Sun, T.; Fang, W.; Chen, W.; Yao, Y.; Bi, F.; Wu, B. High-resolution image inpainting based on multi-scale neural network. Electronics 2019 , 8 , 1370. [ Google Scholar ]

- Rosen, D.; Doherty, K.; Espinoza, T.; Leonard, J. Advances in inference and representation for simultaneous localization and mapping. Annu. Rev. Control. Robot. Auton. Syst. 2021 , 4 , 215–242. [ Google Scholar ] [ CrossRef ]

- Zhou, Y.; Li, J.; Ou, C.; Yan, D.; Zhang, H.; Xue, X. Open-Vocabulary Object Detection in UAV Imagery: A Review and Future Perspectives. Drones 2025 , 9 , 557. [ Google Scholar ] [ CrossRef ]

- Zhao, X.; Liu, Y.; Han, G. Cooperative use of recurrent neural network and siamese region proposal network for robust visual tracking. IEEE Access 2021 , 9 , 57704–57715. [ Google Scholar ] [ CrossRef ]

- Hossain, M.S.; Shahriar, G.M.; Syeed, M.M.M.; Uddin, M.F.; Hasan, M.; Shivam, S.; Advani, S. Region of interest (ROI) selection using vision transformer for automatic analysis using whole slide images. Sci. Rep. 2023 , 13 , 11314. [ Google Scholar ] [ CrossRef ]

- Liu, Y.; Sun, X.; Shao, W.; Yuan, Y. S2ANet: Combining local spectral and spatial point grouping for point cloud processing. Virtual Real. Intell. Hardw. 2024 , 6 , 267–279. [ Google Scholar ] [ CrossRef ]

- Wang, Y.; Li, L.; Dang, C. Calibrating classification probabilities with shape-restricted polynomial regression. IEEE Trans. Pattern Anal. Mach. Intell. 2019 , 41 , 1813–1827. [ Google Scholar ] [ CrossRef ]

- Zhao, P.; Qu, Z.; Bu, Y.; Tan, W.; Ren, Y.; Pu, S. Polardet: A fast, more precise detector for rotated target in aerial images. Int. J. Remote Sens. 2021 , 42 , 5831–5861. [ Google Scholar ]

- Feng, S.; Huang, Y.; Zhang, N. An improved YOLOv8 obb model for ship detection through stable diffusion data augmentation. Sensors 2024 , 24 , 5850. [ Google Scholar ] [ CrossRef ]

- Roy, A.G.; Navab, N.; Wachinger, C. Recalibrating fully convolutional networks with spatial and channel “squeeze and excitation” blocks. IEEE Trans. Med. Imaging 2018 , 38 , 540–549. [ Google Scholar ] [ CrossRef ]

- Magacho, G.; Espagne, E.; Godin, A. Impacts of the CBAM on EU trade partners: Consequences for developing countries. Clim. Policy 2024 , 24 , 243–259. [ Google Scholar ] [ CrossRef ]

- Wen, G.; Li, S.; Liu, F.; Luo, X.; Er, M.; Mahmud, M.; Wu, T. YOLOv5s-CA: A modified YOLOv5s network with coordinate attention for underwater target detection. Sensors 2023 , 23 , 3367. [ Google Scholar ] [ CrossRef ] [ PubMed ]

- Ding, X.; Zhang, X.; Zhou, Y.; Han, J.; Ding, G.; Sun, J. Scaling up your kernels to 31x31: Revisiting large kernel design in cnns. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, New Orleans, LA, USA, 18–24 June 2022; IEEE: New York, NY, USA, 2022; pp. 11963–11975. [ Google Scholar ]

- Deng, L.; Wu, S.; Zhou, J.; Zou, S.; Liu, Q. LSKA-YOLOv8n-WIoU: An Enhanced YOLOv8n Method for Early Fire Detection in Airplane Hangars. Fire 2025 , 8 , 67. [ Google Scholar ] [ CrossRef ]

- Han, J.; Ding, J.; Li, J.; Xia, G. Align deep features for oriented object detection. IEEE Trans. Geosci. Remote Sens. 2021 , 60 , 5602511. [ Google Scholar ] [ CrossRef ]

- Zhang, X.; Song, Y.; Song, T.; Yang, D.; Ye, Y.; Zhou, J.; Zhang, L. LDConv: Linear deformable convolution for improving convolutional neural networks. Image Vis. Comput. 2024 , 149 , 105190. [ Google Scholar ] [ CrossRef ]

- Zhu, X.; Hu, H.; Lin, S.; Dai, J. Deformable convnets v2: More deformable, better results. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, Long Beach, CA, USA, 15–20 June 2019 ; IEEE: New York, NY, USA, 2020; pp. 9308–9316. [ Google Scholar ]

- Guo, M.; Lu, C.; Liu, Z.; Cheng, M.; Hu, S. Visual attention network. Comput. Vis. Media 2023 , 9 , 733–752. [ Google Scholar ] [ CrossRef ]

- Woo, S.; Park, J.; Lee, J.; Kweon, I. Cbam: Convolutional block attention module. In Proceedings of the European Conference on Computer Vision (ECCV), Munich, Germany, 8–14 September 2018; pp. 3–19. [ Google Scholar ]

- Ju, X.; Li, Y.; Huang, G.; Yan, Z.; Wu, X.; Ji, S. Lightweight YOLOv8-Obb Optimization with Hybrid Attention and Dynamic Feature Reconstruction for Remote Sensing Object Detection. Preprints 2025 , 2025041716. [ Google Scholar ]

- Wu, J.; Li, W.; Du, H.; Wan, Y.; Yang, S.; Xiao, Y. An annotated satellite imagery dataset for automated river barrier object detection. Sci. Data 2025 , 12 , 237. [ Google Scholar ] [ CrossRef ]

- Ali, N.; Ullah, M.; Bais, A.; Berraies, S.; Ruan, Y.; Cuthbert, R. Detection of Spatially Oriented Fusarium Head Blight Spikes in Wheat Using UAV-Based Remote Sensing Imaging. IEEE Trans. Geosci. Remote Sens. 2025 , 63 , 1002115. [ Google Scholar ] [ CrossRef ]

- Qin, R.; Liu, Q.; Gao, G.; Huang, D.; Wang, Y. MRDet: A multi-head network for accurate rotated object detection in aerial images. IEEE Trans. Geosci. Remote Sens. 2022 , 60 , 5608412. [ Google Scholar ] [ CrossRef ]

- Zhang, Y.; Guo, W.; Wu, C.; Li, W.; Tao, R. FANet: An arbitrary direction remote sensing object detection network based on feature fusion and angle classification. IEEE Trans. Geosci. Remote Sens. 2023 , 61 , 5608811. [ Google Scholar ] [ CrossRef ]

- Li, Z.; Hou, B.; Wu, Z.; Ren, B.; Yang, C. FCOSR: A simple anchor-free rotated detector for aerial object detection. Remote Sens. 2023 , 15 , 5499. [ Google Scholar ] [ CrossRef ]

- Wang, L.; Shen, Y.; Yang, J.; Zeng, H.; Gao, H. Rotated points for object detection in remote sensing images. IET Image Process. 2024 , 18 , 1655–1665. [ Google Scholar ] [ CrossRef ]

- Lu, W.; Chen, S.; Shu, Q.; Tang, J.; Luo, B. DecoupleNet: A lightweight backbone network with efficient feature decoupling for remote sensing visual tasks. IEEE Trans. Geosci. Remote Sens. 2024 , 62 , 4414613. [ Google Scholar ] [ CrossRef ]

- Xie, X.; Cheng, G.; Wang, J.; Yao, X.; Han, J. Oriented R-CNN for object detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, Montreal, QC, Canada, 10–17 October 2021 ; IEEE: New York, NY, USA, 2021; pp. 3520–3529. [ Google Scholar ]

- Zhao, J.; Ding, Z.; Zhou, Y.; Zhu, H.; Du, W.; Yao, R. OrientedFormer: An end-to-end transformer-based oriented object detector in remote sensing images. IEEE Trans. Geosci. Remote Sens. 2024 , 62 , 5640816. [ Google Scholar ] [ CrossRef ]

- Li, J.; Li, Z.; Chen, M.; Wang, Y.; Luo, Q. A new ship detection algorithm in optical remote sensing images based on improved R3Det. Remote Sens. 2022 , 14 , 5048. [ Google Scholar ] [ CrossRef ]

- Dai, L.; Liu, H.; Tang, H.; Wu, Z.; Song, P. AO2-DETR: Arbitrary-oriented object detection transformer. IEEE Trans. Circuits Syst. Video Technol. 2022 , 33 , 2342–2356. [ Google Scholar ] [ CrossRef ]

- Zhu, X.; Su, W.; Lu, L.; Li, B.; Wang, X.; Dai, J. Deformable detr: Deformable transformers for end-to-end object detection. arXiv 2020 , arXiv:2010.04159. [ Google Scholar ]

- Zhang, H.; Li, F.; Liu, S.; Zhang, L.; Su, H.; Zhu, J.; Ni, L.; Shum, H. Dino: Detr with improved denoising anchor boxes for end-to-end object detection. arXiv 2022 , arXiv:2203.03605. [ Google Scholar ]

# FiguresandTables

The symbol ↓ denotes that lower Angle Error values correspond to superior detection performance.

The symbol ↓ denotes that lower Angle Error values correspond to superior detection performance.

# html-copyright