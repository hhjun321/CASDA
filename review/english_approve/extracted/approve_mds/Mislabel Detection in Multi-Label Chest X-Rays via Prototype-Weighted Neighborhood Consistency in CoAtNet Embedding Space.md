# Abstract

Large-scale chest X-ray (CXR) datasets often rely on report-derived or weak labels, introducing missing and incorrect annotations that can degrade downstream models and limit trust. We study training-free mislabel detection in multi-label CXRs by scoring neighborhood label consistency in a fixed embedding space. Using the NIH Chest X-ray Kaggle sample (5606 CXRs), we extract intermediate CoAtNet features and obtain 64-dimensional embeddings with a frozen CoAtNet backbone and a lightweight refinement head. On top of these embeddings, we compare kNN consistency baselines with distance weighting and label-set similarity against LPV-DW-CS, clustered prototype voting weighted by distance and cluster support. We evaluate three synthetic label-noise regimes with review budgets matched to the corruption rate: random single-label (5% and 20%), boundary-noise (20% corruption within the lowest-density 20% subset), and disjoint-label replacement (20% within that subset). LPV-DW-CS yields the highest downstream macro-AUROC after filtering top-ranked samples (up to 0.8860), while kNN variants achieve higher Recall@budget at the same review rates (up to 99.44%). An image-only expert Likert review of top-ranked real samples finds substantial label-set inconsistencies (54.1% for LPV-DW-CS-280-A; 60.5% for KNN-DW-LSS), supporting neighborhood-consistency ranking as a practical, training-free tool for targeted dataset auditing.

# 1. Introduction

Deep learning systems for medical imaging increasingly rely on large-scale datasets whose labels are obtained from radiology reports, weak supervision, or limited expert annotation. In chest radiography, widely used resources such as NIH Chest X-rays and CheXpert have enabled progress at scale, but they also expose a central challenge: labels are often uncertain, incomplete, or noisy [ 1 , 2 , 3 ]. In multi-label settings, a single image can present multiple co-occurring findings, and the absence of a label is not necessarily evidence of a true negative. This mismatch between recorded labels and underlying pathology can degrade not only predictive performance, but also calibration, interpretability, and ultimately clinical trust [ 4 ]. Label noise in CXR datasets appears in several forms, including missing positives (under-labeling), spurious positives arising from ambiguous language, and systematic confusions among clinically related conditions. A recurring (and practically important) point of divergence is whether these discrepancies should be treated as annotation errors to be corrected, or as epistemic uncertainty that should be modeled explicitly [ 2 , 3 ]. Robust-learning methods aim to mitigate noise during training (e.g., via loss correction, reweighting, or sample selection), whereas data-centric approaches prioritize auditing and curation so that downstream models learn from cleaner supervision [ 3 , 5 ].

In practice, suspicious samples can be corrected or simply removed from training depending on the application and review budget, motivating complementary tools that directly flag candidates for targeted inspection [ 5 ]. In this work, we study training-free mislabel detection using neighborhood consistency in a high-quality embedding space. Recent training-free methods suggest that, given high-quality representations, local label agreement among nearest neighbors can effectively surface mislabeled instances [ 6 , 7 ]. Building on these ideas, we extract fixed visual representations from a CoAtNet backbone [ 8 ], the best-performing individual model reported in SynthEnsemble [ 9 ].

We score each image based on how consistent its label set is with those of nearby images. This setting is explicitly data-centric. The goal is not to retrain a noise-robust model, but to generate actionable rankings that prioritize expert time. It also yields training-free consistency signals that can be used alongside embeddings in downstream pipelines.

This study originated from a diversity-oriented medical case-retrieval pipeline for differential diagnosis: visually similar chest X-rays were intentionally retrieved across different labels, and expert inspection suggested that many disagreements reflected under-labeling rather than true clinical dissimilarity. This observation motivates a curation-oriented objective: rank images for which local neighborhood evidence contradicts the recorded labels, thereby focusing review on samples that are most likely to improve dataset reliability if corrected or removed.

Unlike robust-learning and training-signal-based noisy-label methods, which rely on additional optimization, loss dynamics, or relabeling mechanisms during training [ 10 , 11 , 12 , 13 ], our approach performs suspicious-sample ranking directly on fixed embeddings without training a dedicated noise-robust predictor for scoring. Relative to recent training-free embedding-space methods [ 6 , 7 , 14 ], we focus on the multi-label chest X-ray setting, where label-set overlap, under-labeling, and “No Finding” cases make neighborhood-consistency scoring more challenging. We further study both direct kNN neighborhood-consistency variants and a prototype-based method family derived from our retrieval pipeline, and we assess the resulting rankings not only by recovery under controlled synthetic corruptions, but also by downstream macro-AUROC after filtering and an image-only expert review on real samples.

## Contributions

Our main contributions are as follows:

(i) Study of two training-free method families for multi-label CXRs: (a) kNN + clustering methods derived from our retrieval pipeline that summarize each neighborhood via representative prototypes (varying k and the number of clusters) and combine distance and representativeness weighting; (b) Pure kNN neighborhood-consistency approaches based on signed consistency scoring, distance-weighted voting, and label-set (Jaccard) similarity weighting. (ii) Evaluation under multiple synthetic noise mechanisms: we evaluate the methods under random relabeling, boundary-focused noise, and disjoint-label replacements on the NIH Chest X-ray Kaggle sample, using Recall@budget and downstream macro-AUROC after filtering. (iii) Expert medical review of top-ranked real samples: we complement the synthetic study with an expert medical review of top-ranked candidates from one representative method per family using an image-only Likert protocol, reporting confirmation rates and label-wise patterns among the reviewed cases.

Study of two training-free method families for multi-label CXRs:

(a) kNN + clustering methods derived from our retrieval pipeline that summarize each neighborhood via representative prototypes (varying k and the number of clusters) and combine distance and representativeness weighting; (b) Pure kNN neighborhood-consistency approaches based on signed consistency scoring, distance-weighted voting, and label-set (Jaccard) similarity weighting.

kNN + clustering methods derived from our retrieval pipeline that summarize each neighborhood via representative prototypes (varying k and the number of clusters) and combine distance and representativeness weighting;

Pure kNN neighborhood-consistency approaches based on signed consistency scoring, distance-weighted voting, and label-set (Jaccard) similarity weighting.

Evaluation under multiple synthetic noise mechanisms: we evaluate the methods under random relabeling, boundary-focused noise, and disjoint-label replacements on the NIH Chest X-ray Kaggle sample, using Recall@budget and downstream macro-AUROC after filtering.

Expert medical review of top-ranked real samples: we complement the synthetic study with an expert medical review of top-ranked candidates from one representative method per family using an image-only Likert protocol, reporting confirmation rates and label-wise patterns among the reviewed cases.

# 2. Related Work

## 2.1. Label Noise and Uncertainty in Chest Radiography

Large-scale CXR datasets commonly derive labels from radiology reports or weak supervision, which introduces ambiguity, incompleteness, and label noise [ 1 , 2 , 3 ]. In multi-label settings, co-occurring findings are frequent and the absence of a label is not equivalent to a confirmed negative, making under-labeling particularly prevalent. Beyond reducing predictive performance, such discrepancies can harm calibration and interpretability and ultimately erode clinical trust when model outputs are treated as reliable evidence [ 4 , 15 ]. A practical point of divergence is whether these discrepancies should be treated primarily as annotation errors to be corrected or as epistemic uncertainty that should be modeled explicitly [ 2 , 3 ].

## 2.2. Robust Learning Versus Data-Centric Curation

The substantial literature addresses noisy supervision via robust objectives, loss correction, and sample selection during training [ 10 , 11 , 12 ]. Recent surveys in medical image analysis provide updated overviews of label-noise learning and organize the field into broad methodological categories including regularization, loss redesign, loss reweighting, label correction, sample selection, and hybrid approaches [ 16 ]. Within this broader landscape, recent noisy-label detection work has sought to address the limitations of prior methods that rely on distinguishable training signals, such as training loss, to differentiate between clean and noisy labels. DynaCor, for example, distinguishes incorrectly labeled instances from correctly labeled ones based on the dynamics of these training signals [ 13 ]. To cope with the absence of supervision for clean and noisy labels, it introduces a label-corruption strategy that augments the original dataset with intentionally corrupted labels, thereby enabling indirect simulation of the model’s behavior on noisy labels. It then identifies clean and noisy instances by inducing distinguishable clusters from latent representations of training dynamics.

Complementarily, data-centric approaches emphasize auditing datasets to identify suspicious samples for expert review and curation [ 5 ]. In practice, flagged samples may be corrected when feasible or removed when review resources are limited or when the downstream application prioritizes conservative supervision over completeness [ 5 ]. This motivates lightweight tools that surface high-value candidates for inspection without requiring repeated model retraining.

## 2.3. Training-Free Mislabel Detection in Embedding Space

Recent training-free methods show that, given high-quality representations, neighborhood label agreement among nearest neighbors can effectively surface corrupted labels and under-labeling patterns [ 6 , 7 ]. Importantly, these works also emphasize that the resulting neighborhood-consistency signals can be reused beyond pure dataset auditing: for instance, training-free detection can act as a lightweight preprocessing module to prepare data for subsequent pipelines such as semi-supervised learning [ 6 ]. More broadly, embedding-space approaches advocate shifting part of the learning burden from end-to-end training to interpretable, training-free schemes on top of high-quality pretrained features, and show that neighborhood-derived reliability signals can even be appended to the embedding and exploited by simple downstream classifiers [ 7 ]. More recent work has extended this line through FastSimiFeat [ 14 ]. It retains the training-free use of k -NN over pretrained features, but introduces a confusion-matrix-based noise ratio estimator, an adaptive number of k -NN cycles based on the estimated noise level, and label-correction steps to improve efficiency and noisy-data handling.

Within this broader landscape, recent surveys in medical image analysis show that most label-noise methods remain centered on robust training, correction, or sample-selection strategies during optimization [ 16 ]. In contrast, our emphasis is on a representation-first, data-centric setting in which the mislabel scoring stage itself remains training-free on top of fixed embeddings. Our work follows this representation-first paradigm in the multi-label CXR setting and studies two method families: (i) kNN + clustering variants derived from our retrieval pipeline that summarize neighborhoods via representative prototypes and incorporate distance and cluster-support weighting; and (ii) pure kNN consistency baselines including binary voting, signed scoring, distance weighting, and label-set (Jaccard) similarity weighting. All methods operate on fixed CoAtNet embeddings [ 8 ], yielding actionable rankings for data curation without training a dedicated noise-robust predictor.

# 3. Materials and Methods

## 3.1. Dataset

We use the Random Sample of the NIH Chest X-ray Dataset [ 1 ], a randomly selected subset of the full NIH Chest X-ray Dataset. This sample contains 5606 frontal chest radiographs and their corresponding class labels over 15 categories (14 findings plus “No Finding”). Each image is associated with a binary label vector yi∈{0,1}L , where L=15 . We denote by Yi={ℓ|yiℓ=1} the set of positive labels for sample i . For downstream classifier training we follow common practice and predict the 14 disease labels; “No Finding” is retained for mislabel scoring because it influences neighborhood consistency.

## 3.2. Patient-Disjoint Splitting

To avoid patient leakage, we split data by patient ID extracted from image filenames [ 17 ]. We create a 20% patient-disjoint test set. The remaining 80% is split into train and validation groups using group-aware splitting by patient ID, yielding approximately 70% train and 10% validation overall.

## 3.3. Embedding Space Construction

We extract a 2048-dimensional intermediate embedding from a frozen CoAtNet backbone, using the pretrained weights from SynthEnsemble [ 8 , 9 ]. In SynthEnsemble, the upstream model is trained for multi-label prediction of the 14 thoracic findings rather than with an explicit “No Finding” output. As a result, while the backbone provides strong disease-discriminative visual features, it does not explicitly encourage “No Finding” studies to form a compact region in feature space. In our extracted backbone features, “No Finding” cases appeared empirically more dispersed than many pathology-specific neighborhoods, which is undesirable for neighborhood-consistency scoring.

We therefore introduce a lightweight refinement head to obtain a more compact and audit-oriented embedding space. The refinement head maps the 2048-dimensional backbone features to a 64-dimensional representation through hidden layers of 512, 128, and 64 units with ReLU activations, followed by a sigmoid output over the pathology labels. The frozen CoAtNet backbone contains approximately 73.9 M parameters, while the refinement head contains approximately 1.12 M trainable parameters.

The head is trained with a false-positive-weighted multi-label loss: L(y,y^)=1L′∑ℓ=1L′BCE(yℓ,y^ℓ)+λ·(1−yℓ)·y^ℓ, (1) where L′=14 , y∈{0,1}L′ and y^∈[0,1]L′ denote the multi-label targets and predicted probabilities, respectively, and “No Finding” is represented implicitly as the all-zero disease vector. In our implementation, we set λ=2.0 to increase the penalty for confident false positives on negative labels.

The refinement head is trained once before any synthetic label corruption is injected for the auditing experiments. We use Adam with learning rate 1×10−3 , batch size 32, weight decay 0, and train for up to 50 epochs with early stopping (patience = 5). The refinement head was trained using the same optimization setup adopted for the CoAtNet-based training stage in SynthEnsemble, so as to remain consistent with the representation pipeline from which the frozen backbone features were obtained. After training, the head is frozen and its 64-dimensional hidden representation is used as the final embedding vector zi in the main experiments.

All mislabel scoring methods described below operate in a training-free manner on top of these fixed embeddings. To assess the effect of this supervised refinement stage, we compare the performance obtained with the refined 64-dimensional embeddings against that obtained with the raw frozen 2048-dimensional CoAtNet features under the disjoint-label replacement setting in the synthetic evaluation. Nearest neighbors are computed under Euclidean distance in the corresponding embedding space.

## 3.4. Neighborhood-Consistency Scores

Let dmax denote the global maximum pairwise distance in embedding space. dmax=maxp,q∈{1,…,N}d(zp,zq) (2)

We define the globally normalized distance weight as d˜ij=d(zi,zj)dmax,wij=1−d˜ij. (3) so that d˜ij∈[0,1] and wij∈[0,1] .

Let Nk1(i) be the set of k1 nearest neighbors of sample i under Euclidean distance in embedding space. We measure agreement between a sample i and a neighbor j with a signed vote vij=+1,if|Yi∩Yj|>0,−1,otherwise. (4)

We then compute a weighted neighborhood-consistency score Ci=1k1∑j∈Nk1(i)wijvij,Ci∈[−1,1], (5) and define a sample-level suspiciousness score as Si=1−Ci2,Si∈[0,1], (6) so larger Si indicates stronger disagreement with the local neighborhood and thus higher suspicion of mislabeling.

We compare the following weighting schemes (with globally normalized Euclidean distances in embedding space):

KNN: Uniform weights wij=1 . KNN-DW: Distance weights wij=1−d˜ij . KNN-DW-LSS: Distance weights modulated by label-set similarity (LSS). For multi-label agreement, we use Jaccard similarity between label sets Yi and Yj : LSS(i,j)=|Yi∩Yj||Yi∪Yj|,wij=(1−d˜ij)·LSS(i,j),if|Yi∩Yj|>0,(1−d˜ij),if|Yi∩Yj|=0. (7)

KNN: Uniform weights wij=1 .

KNN-DW: Distance weights wij=1−d˜ij .

KNN-DW-LSS: Distance weights modulated by label-set similarity (LSS). For multi-label agreement, we use Jaccard similarity between label sets Yi and Yj : LSS(i,j)=|Yi∩Yj||Yi∪Yj|,wij=(1−d˜ij)·LSS(i,j),if|Yi∩Yj|>0,(1−d˜ij),if|Yi∩Yj|=0. (7)

In our experiments, the kNN baselines use k1=10 to preserve local neighborhood structure while maintaining stable agreement estimates, consistent with prior training-free corrupted-label detection work [ 6 ].

## 3.5. LPV-DW-CS: Clustered Prototype Voting with Cluster Support

For compactness, we denote LPV-DW-CS variants as LPV-DW-CS- k1 - k2 , where the first suffix indicates the retrieved neighborhood size and the second indicates the clustering configuration. We use A to denote the adaptive setting in which k2 equals the number of unique label sets observed in the neighborhood.

LPV-DW-CS replaces the full neighbor set Nk1(i) with a set of prototypes obtained by clustering the neighborhood with KMeans [ 18 ]. Concretely, for each query sample i :

1. Retrieve a larger neighborhood Nk1(i) (we use k1∈{280,512} ). 2. Cluster {zj:j∈Nk1(i)} into k2 clusters. We consider fixed k2∈{5,15} and an adaptive setting where k2 equals the number of unique label sets observed in the neighborhood. 3. For each cluster c , select one representative (prototype) rc as the neighbor closest to the query sample i within that cluster. 4. Compute a distance weight for each prototype using the global distance normalization defined above: wicdist=1−d(zi,zrc)dmax. (8) 5. Compute a cluster-support (representativity) factor, CSc=1|c|∑j∈cIYj∩Yrc≠∅, (9) which measures how consistently the prototype’s label set is supported by samples in its own cluster. 6. Compute a prototype vote sign vic=+1 if |Yi∩Yrc|>0 and vic=−1 otherwise, and compute Ci=1k2∑c=1k2wicdist·CScvic,Si=1−Ci2. (10)

Retrieve a larger neighborhood Nk1(i) (we use k1∈{280,512} ).

Cluster {zj:j∈Nk1(i)} into k2 clusters. We consider fixed k2∈{5,15} and an adaptive setting where k2 equals the number of unique label sets observed in the neighborhood.

For each cluster c , select one representative (prototype) rc as the neighbor closest to the query sample i within that cluster.

Compute a distance weight for each prototype using the global distance normalization defined above: wicdist=1−d(zi,zrc)dmax. (8)

Compute a cluster-support (representativity) factor, CSc=1|c|∑j∈cIYj∩Yrc≠∅, (9) which measures how consistently the prototype’s label set is supported by samples in its own cluster.

Compute a prototype vote sign vic=+1 if |Yi∩Yrc|>0 and vic=−1 otherwise, and compute Ci=1k2∑c=1k2wicdist·CScvic,Si=1−Ci2. (10)

# 4. Evaluation

We conduct two complementary evaluations. First, we inject synthetic label noise under multiple regimes, allowing evaluation against a known injected corruption set. Second, we complement this with an expert medical review on real samples, which is particularly relevant because large CXR datasets may contain native label noise from report-derived weak supervision and ambiguity in clinical language.

The expert inspects each case using a lightweight protocol in which only the CXR image and its dataset-provided label set are shown, without access to the radiology report or additional clinical context.

## 4.1. Synthetic Label-Noise Evaluation

We inject synthetic label noise into the train+val pool while keeping the patient-disjoint test set unchanged. This yields a known injected corruption set for evaluation across three regimes of increasing difficulty, from random relabeling to locally plausible disjoint-label replacements. We then measure recovery at a fixed review budget and the downstream impact of filtering the most suspicious samples.

In addition, we conduct a representation-space ablation under the disjoint-label replacement regime (20%, low-density subset), using the same patient-disjoint split, review budget, and evaluation metrics to compare raw frozen 2048-dimensional CoAtNet features against the refined 64-dimensional embeddings.

### 4.1.1. Noise Injection Regimes

We consider three corruption regimes of increasing difficulty:

1. Random relabeling (5% and 20%): A fraction p of samples are selected uniformly at random. For each selected sample, we corrupt the annotation by replacing it with a single positive label that was not present in the original label set. 2. Boundary-noise (20%, low-density subset): We use the mean Euclidean distance to the k=10 nearest neighbors as an inverse-density metric in embedding space (higher mean distance indicates lower density). We then select the 20% lowest-density samples and apply the same single-label corruption as in random relabeling. 3. Disjoint-label replacement (20%, low-density subset): Using the same 20% lowest-density subset, we replace each selected sample’s label set with that of its nearest neighbor whose label set is disjoint (empty intersection). This produces harder, label-plausible corruptions that are more difficult to distinguish from clean annotations using local evidence alone.

Random relabeling (5% and 20%): A fraction p of samples are selected uniformly at random. For each selected sample, we corrupt the annotation by replacing it with a single positive label that was not present in the original label set.

Boundary-noise (20%, low-density subset): We use the mean Euclidean distance to the k=10 nearest neighbors as an inverse-density metric in embedding space (higher mean distance indicates lower density). We then select the 20% lowest-density samples and apply the same single-label corruption as in random relabeling.

Disjoint-label replacement (20%, low-density subset): Using the same 20% lowest-density subset, we replace each selected sample’s label set with that of its nearest neighbor whose label set is disjoint (empty intersection). This produces harder, label-plausible corruptions that are more difficult to distinguish from clean annotations using local evidence alone.

### 4.1.2. Metrics

We report two complementary metrics, capturing recovery under a fixed review budget and the downstream impact of filtering:

Recall@budget: the fraction of injected corruptions recovered within the top- p% most suspicious ranked list, where p matches the injected noise rate. Downstream macro-AUROC after filtering: we remove the top- p% most suspicious samples from the train+val pool, train a multi-label classifier on the remaining data, and evaluate downstream macro-AUROC on the patient-disjoint test set.

Recall@budget: the fraction of injected corruptions recovered within the top- p% most suspicious ranked list, where p matches the injected noise rate.

Downstream macro-AUROC after filtering: we remove the top- p% most suspicious samples from the train+val pool, train a multi-label classifier on the remaining data, and evaluate downstream macro-AUROC on the patient-disjoint test set.

When a single run is reported, we use seed 85. When repeated runs are used to estimate variability, we use the seed set [85,37,82,14,59] , with one run per seed, and report mean ± std across the resulting values. In particular, downstream macro-AUROC results are reported over these five seeds, whereas single-run analyses use seed 85 unless otherwise stated. Recall@budget is computed with respect to the injected corruption set.

### 4.1.3. Distance Normalization Ablation

We compare the global distance normalization used in the main experiments against a local alternative computed within each query neighborhood. This ablation is evaluated under disjoint-label replacement using the same patient-disjoint split and the same downstream and ranking metrics as in the main experiments. The goal is to assess whether scoring based on neighborhood distances benefits more from a global distance scale or from neighborhood-specific rescaling.

### 4.1.4. Neighborhood and Clustering Sensitivity

We also study the sensitivity of the ranking methods to their neighborhood and clustering parameters under disjoint-label replacement. For KNN-DW-LSS, we vary the neighborhood size k1 . For LPV-DW-CS, we vary both the retrieved neighborhood size k1 and the number of clusters k2 . Because these additional parameter sweeps were evaluated only with Recall@budget, they are used to characterize ranking sensitivity rather than to replace the main results, which report both ranking recovery and downstream macro-AUROC after filtering.

### 4.1.5. Computational Cost Evaluation

To assess practical deployment cost, we benchmark the ranking stage on the full dataset ( N=5606 ) using precomputed embeddings. For each method, we measure the wall-clock time required to produce a complete suspiciousness ranking over all samples, after one warm-up run and over five measured runs. We also report the average per-query time and the one-time cost of precomputing the global maximum distance dmax .

All benchmarks were run in Google Colab on Linux (6.6.113+), using Python 3.12.13 on an Intel Xeon CPU @ 2.20 GHz with 12.67 GB RAM and no GPU. The benchmark was executed with thread pools limited to one thread to improve run-to-run consistency and comparability across methods, using NumPy 2.0.2, scikit-learn 1.6.1, and SciPy 1.16.3. The reported runtimes correspond to the ranking stage on precomputed embeddings and therefore do not include image loading, backbone feature extraction, or refinement-head forward passes.

## 4.2. Expert Medical Review (Likert Validation)

Real-world label noise in large CXR datasets is often dominated by under-labeling and report-derived uncertainty. To complement the synthetic study, we perform a targeted expert review on real (non-synthetically corrupted) data.

### 4.2.1. Sample Selection

We construct suspiciousness rankings using two representative methods: LPV-DW-CS-280-A (with k1=280 and adaptive k2 ) and KNN-DW-LSS. From each method’s top-1% most suspicious samples, we randomly select a subset for expert assessment (37 samples from LPV-DW-CS-280-A and 38 samples from KNN-DW-LSS), keeping the review workload manageable while focusing on cases with the strongest predicted label inconsistencies.

### 4.2.2. Review Protocol

A medical expert is shown only the suspicious chest X-ray image and its associated multi-label annotations (no radiology report and no additional clinical context). The expert rates the level of agreement with the provided label set using a four-point Likert scale: Strongly agree, Agree, Disagree and Strongly disagree.

Optionally, the expert may provide brief notes explaining the rating.

### 4.2.3. Operational Definition of Mislabeling

To keep the expert review lightweight given limited clinician time, we do not ask the reviewer to adjudicate each label independently despite the multi-label nature of the dataset. Instead, the expert provides a global agreement rating for the entire label set. Under this operational protocol, any clinically relevant mismatch is treated as a mislabeling signal: missing findings (under-labeling) and/or false-positive findings (over-labeling) are both considered evidence of an incorrect annotation.

### 4.2.4. Binarization of Likert Responses

For analysis, we map the four-point Likert responses into a binary indicator of suspected mislabeling. We treat Disagree and Strongly disagree as mislabeled (label set judged inconsistent with the image) and Agree and Strongly agree as non-mislabeled (label set judged broadly consistent).

# 5. Results

## 5.1. Synthetic Label-Noise Results

Table 1 reports the representation-space ablation under disjoint-label replacement. Table 2 , Table 3 , Table 4 and Table 5 summarize mislabel-detection performance across the three synthetic noise regimes. We report (i) downstream macro-AUROC (mean ± std over five seeds), computed after removing the top- p% most suspicious samples from the train + val pool (with p matched to the injected noise rate) and evaluating on the patient-disjoint test set, and (ii) Recall@budget, measuring how many synthetically corrupted samples are recovered within that same top- p% review budget.

### 5.1.1. Representation-Space Ablation Under Disjoint-Label Replacement

Table 1 compares mislabel detection using the raw frozen 2048-dimensional CoAtNet features and the refined 64-dimensional embeddings under the same disjoint-label replacement regime (20%, low-density subset), patient-disjoint split, and top-20% review budget.

The refined representation improves both downstream macro-AUROC and Recall@budget for the two method families under this evaluated setting. For KNN-DW-LSS, downstream macro-AUROC increases from 0.8772 ± 0.0080 to 0.8792 ± 0.0043, while Recall@budget rises from 55.63% to 98.44%. The gain is larger for LPV-DW-CS-280-A, whose downstream macro-AUROC increases from 0.8662 ± 0.0021 to 0.8840 ± 0.0024 and whose Recall@budget rises from 42.47% to 89.19%.

Notably, LPV-DW-CS-280-A underperforms KNN-DW-LSS on the raw backbone features, but becomes the strongest method in downstream macro-AUROC under this refinement setting. Table 1. Representation-space ablation under disjoint-label replacement (20%, low-density subset). We compare raw frozen 2048-dimensional CoAtNet features and refined 64-dimensional embeddings using the same patient-disjoint split, top-20% review budget, and evaluation protocol. Higher values are better (↑). Best values are shown in bold. Table 1. Representation-space ablation under disjoint-label replacement (20%, low-density subset). We compare raw frozen 2048-dimensional CoAtNet features and refined 64-dimensional embeddings using the same patient-disjoint split, top-20% review budget, and evaluation protocol. Higher values are better (↑). Best values are shown in bold.

### 5.1.2. Random Relabeling

Under random single-label replacement at 5% noise, Table 2 shows that LPV-DW-CS achieves the highest downstream macro-AUROC in this regime ( 0.8860±0.0039 ), while the kNN baselines achieve the highest recovery at the fixed 5% budget, with the best recall of 98.66% for KNN-DW-LSS. Table 2. Performance under random relabeling 5% (224 corrupted samples, selection budget = top-5%). We report downstream macro-AUROC (mean ± std over five seeds) and recall at the budget. Higher values are better (↑). Best values are shown in bold. Table 2. Performance under random relabeling 5% (224 corrupted samples, selection budget = top-5%). We report downstream macro-AUROC (mean ± std over five seeds) and recall at the budget. Higher values are better (↑). Best values are shown in bold.

Table 3 reports the 20% random-noise regime and confirms the same pattern. LPV-DW-CS yields the strongest downstream macro-AUROC ( 0.8829±0.0032 ), whereas kNN variants recover almost all corrupted samples at the review budget, reaching 99.44% recall for KNN and KNN-DW. Table 3. Performance under random relabeling 20% (897 corrupted samples, selection budget = top-20%). We report downstream macro-AUROC (mean ± std over five seeds) and recall at the budget.Higher values are better (↑). Best values are shown in bold. Table 3. Performance under random relabeling 20% (897 corrupted samples, selection budget = top-20%). We report downstream macro-AUROC (mean ± std over five seeds) and recall at the budget.Higher values are better (↑). Best values are shown in bold.

### 5.1.3. Boundary-Focused Noise

For boundary-noise targeting the lowest-density 20% of samples, Table 4 reports that LPV-DW-CS achieves the best downstream macro-AUROC in this setting ( 0.8854±0.0020 ). In contrast, the kNN baselines achieve higher Recall@budget, reaching up to 98.55% , which indicates stronger recovery of corrupted samples at the fixed review rate. Table 4. Performance under boundary-noise 20% (897 corrupted samples, selection budget = top-20%). We report downstream macro-AUROC (mean ± std over five seeds) and recall at the budget. Higher values are better (↑). Best values are shown in bold. Table 4. Performance under boundary-noise 20% (897 corrupted samples, selection budget = top-20%). We report downstream macro-AUROC (mean ± std over five seeds) and recall at the budget. Higher values are better (↑). Best values are shown in bold.

### 5.1.4. Disjoint-Label Replacement

In the disjoint-label replacement regime on the same boundary subset, LPV-DW-CS attains the strongest downstream macro-AUROC after filtering in this setting ( 0.8859±0.0019 ), as shown in Table 5 . Among the kNN baselines, KNN-DW yields the highest Recall@budget ( 87.29% ) while achieving downstream macro-AUROC 0.8785±0.0075 .

Overall, across Table 2 , Table 3 , Table 4 and Table 5 , LPV-DW-CS consistently delivers higher downstream macro-AUROC, whereas the kNN variants remain the most effective at recovering corrupted samples within the fixed review budget. Table 5. Performance under disjoint-label replacement 20% (boundary subset) (897 corrupted samples, selection budget = top-20%). We report downstream macro-AUROC (mean ± std over five seeds) and recall at the budget. Higher values are better (↑). Best values are shown in bold. Table 5. Performance under disjoint-label replacement 20% (boundary subset) (897 corrupted samples, selection budget = top-20%). We report downstream macro-AUROC (mean ± std over five seeds) and recall at the budget. Higher values are better (↑). Best values are shown in bold.

### 5.1.5. Distance Normalization Ablation Results

Table 6 compares local and global distance normalization under disjoint-label replacement. Overall, global normalization yields substantially stronger Recall@budget across both the kNN and LPV-DW-CS families, and also produces markedly better downstream macro-AUROC for the LPV-DW-CS variants.

For the direct kNN baselines, the effect of normalization is most pronounced in ranking recovery. For example, KNN improves from 86.06% to 98.55% Recall@budget when moving from local to global normalization, and KNN-DW-LSS improves from 86.29% to 98.44%. Changes in downstream macro-AUROC are smaller in this family, with KNN-DW-LSS increasing from 0.8781±0.0084 to 0.8792±0.0043 , while KNN-DW shows similar macro-AUROC under both normalizations.

For LPV-DW-CS, the benefit of global normalization is much stronger and is reflected in both evaluation criteria. For instance, LPV-DW-CS-280-A improves from 0.8678±0.0066 to 0.8840±0.0024 in downstream macro-AUROC and from 57.86% to 89.19% in Recall@budget, while LPV-DW-CS-280-15 improves from 0.8654±0.0125 to 0.8833±0.0037 and from 60.42% to 87.29%, respectively.

These results support the use of global distance normalization throughout the paper. Empirically, a global distance scale provides substantially better recovery of corrupted samples and, for LPV-DW-CS, a much stronger downstream filtering effect than neighborhood-specific rescaling. Table 6. Distance normalization ablation under disjoint-label replacement. Global normalization consistently improves Recall@budget and is especially beneficial for LPV-DW-CS in downstream macro-AUROC. Higher values are better (↑). Table 6. Distance normalization ablation under disjoint-label replacement. Global normalization consistently improves Recall@budget and is especially beneficial for LPV-DW-CS in downstream macro-AUROC. Higher values are better (↑).

### 5.1.6. Neighborhood and Clustering Sensitivity Results

Table 7 and Table 8 summarize the parameter-sensitivity analysis under disjoint-label replacement using Recall@budget. For KNN-DW-LSS, performance remained high across a broad range of neighborhood sizes, increasing from 95.65% at k1=5 to 98.44% at k1=10 , peaking at 99.00% for k1∈{20,40} , and then decreasing moderately for larger neighborhoods. This indicates that the direct kNN ranking is relatively stable across moderate values of k1 , with only modest gains beyond the main setting k1=10 .

For LPV-DW-CS, Recall@budget showed a stronger dependence on both k1 and k2 . In these recall-only sweeps, smaller retrieved neighborhoods and larger numbers of clusters consistently improved recovery of corrupted samples. The best LPV-DW-CS result in this analysis was obtained at k1=64 and k2=20 , reaching 95.99% Recall@budget. For fixed k1 , increasing k2 generally improved recall, whereas increasing k1 from 64 to 280 and 512 led to progressively lower recovery.

Because these additional parameter sweeps were evaluated only with Recall@budget, they should be interpreted as a characterization of ranking sensitivity rather than as a replacement for the main results, which report both ranking recovery and downstream macro-AUROC after filtering. Table 7. Neighborhood-size sensitivity for KNN-DW-LSS under disjoint-label replacement (20%, low-density subset), reported as Recall@budget (%) at a top-20% review budget. Higher values are better. Best values are shown in bold. Table 7. Neighborhood-size sensitivity for KNN-DW-LSS under disjoint-label replacement (20%, low-density subset), reported as Recall@budget (%) at a top-20% review budget. Higher values are better. Best values are shown in bold. Table 8. Neighborhood and clustering sensitivity for LPV-DW-CS under disjoint-label replacement (20%, low-density subset), reported as Recall@budget (%) at a top-20% review budget. Rows vary the retrieved neighborhood size k1 , and columns vary the number of clusters k2 . Higher values are better. Best value is shown in bold. Table 8. Neighborhood and clustering sensitivity for LPV-DW-CS under disjoint-label replacement (20%, low-density subset), reported as Recall@budget (%) at a top-20% review budget. Rows vary the retrieved neighborhood size k1 , and columns vary the number of clusters k2 . Higher values are better. Best value is shown in bold.

### 5.1.7. Computational Cost Results

Table 9 summarizes the computational cost of the ranking stage on the full dataset ( N=5606 ) using precomputed embeddings. Precomputing the global maximum distance dmax required 0.6369 s.

Among the evaluated methods, KNN-DW-LSS is the fastest, requiring 9.5282±0.9210 s to produce a full ranking, corresponding to 1.6996 ms per query. LPV-DW-CS-280-5 increases runtime to 34.6800±0.7679 s (6.1862 ms/query), LPV-DW-CS-280-15 to 55.3906±0.3079 s (9.8806 ms/query), and LPV-DW-CS-280-A to 58.9891±0.4794 s (10.5225 ms/query).

Thus, the direct kNN baseline provides the lowest ranking cost, while the prototype-based methods introduce a clear runtime overhead due to neighborhood clustering and prototype selection. Nevertheless, even the slowest evaluated variant remains below one minute for a full ranking over all 5606 samples in this benchmark setting. Table 9. Computational cost of the ranking stage on the full dataset ( N=5606 ) using precomputed embeddings. We report mean wall-clock time over 5 runs after one warm-up run, together with the average time per query. Lower values are better (↓). Best values are shown in bold. Table 9. Computational cost of the ranking stage on the full dataset ( N=5606 ) using precomputed embeddings. We report mean wall-clock time over 5 runs after one warm-up run, together with the average time per query. Lower values are better (↓). Best values are shown in bold.

## 5.2. Expert Medical Review

We report expert-review outcomes for two representative ranking methods, one from each family. LPV-DW-CS-280-A represents the kNN + clustering approach, and KNN-DW-LSS represents the direct kNN baseline. The reviewed sample lists from the two methods overlap in 17 cases. Therefore, the expert reviewed 58 unique images in total across both lists, while the 17 overlapping cases contribute to the per-method counts for both methods.

### 5.2.1. Likert Distribution

Using the four-point Likert protocol described in Section 4.2 , we binarize responses by treating Disagree and Strongly disagree as mislabeled and Agree and Strongly agree as non-mislabeled. Under this mapping, LPV-DW-CS-280-A achieves an expert-confirmed precision of 54.1% with 20 mislabels out of 37 reviewed cases, whereas KNN-DW-LSS achieves 60.5% with 23 mislabels out of 38 reviewed cases. Thus, KNN-DW-LSS yields a higher confirmed mislabel rate on the reviewed subset, while both methods identify a substantial fraction of label-set disagreements under this lightweight, image-only review protocol. The full Likert response distribution for both methods is shown in Figure 1 .

### 5.2.2. Cumulative Precision Along the Ranking

To assess how confirmation quality evolves as the review budget increases, Figure 2 and Figure 3 report cumulative Precision@k over the reviewed cases, in ranking order. Both methods exhibit larger fluctuations in the earliest ranks and stabilize as k increases. KNN-DW-LSS maintains a consistently higher cumulative precision over most of the reviewed range and reaches a higher peak in the mid-ranking segment, indicating that its cumulative precision remains higher over most of the reviewed range as the review budget increases. LPV-DW-CS-280-A shows smaller oscillations around its mean, indicating a steadier but lower-yield ranking on this subset.

### 5.2.3. Mislabel Confirmation Rate by Label

We further stratify expert-confirmed mislabel outcomes by pathology to identify which labels are most frequently implicated among the reviewed candidates. Figure 4 and Figure 5 report the expert-confirmed mislabel rate per pathology for KNN-DW-LSS and LPV-DW-CS-280-A, respectively. We restrict the analysis to labels with at least three reviewed cases per method to reduce small-sample artifacts.

Across both rankings, Emphysema reaches 100% confirmation, with n=3 in each method. For LPV-DW-CS-280-A, Pneumonia also reaches 100% with n=4 , whereas KNN-DW-LSS attains 62.5% with n=8 . KNN-DW-LSS additionally surfaces high-confirmation candidates for Mass at 75.0% with n=4 and Pneumothorax at 66.7% with n=3 . More frequent labels such as Fibrosis show intermediate confirmation rates, 50.0% with n=10 for LPV-DW-CS-280-A and 54.5% with n=11 for KNN-DW-LSS, consistent with a mixture of under-labeling and over-labeling patterns that remain challenging to adjudicate under an image-only protocol.

Because per-label sample sizes differ across methods, these label-wise rates should be interpreted as qualitative signals rather than definitive comparative evidence. Their purpose is not to estimate a reliable pathology-wise distribution of mislabels, but to illustrate that the reviewed suspicious cases are not concentrated in a single label and instead span multiple pathologies.

# 6. Discussion

Across random relabeling regimes, LPV-DW-CS attains the highest downstream macro-AUROC after filtering, indicating that removing its top- p% flagged samples yields the largest improvement in test performance. At the same time, the kNN baselines achieve the best recall at the fixed review budget (top- p% ), which is often the operational objective when expert time is limited. In addition to auditing, such scores can be reused downstream as training-free reliability cues in embedding-based pipelines.

In boundary-noise settings, LPV-DW-CS again yields the strongest post-filtering macro-AUROC in our experiments, while kNN variants consistently produce higher Recall@budget at the same filtering rate. This indicates that prototype compression may spread suspicion more evenly across the ranked list, reducing the concentration of corrupted samples at the very top.

In the disjoint-label replacement regime, LPV-DW-CS achieves the best downstream macro-AUROC after filtering ( 0.8859±0.0019 ) but lower Recall@budget (80.94%). In contrast, the kNN baselines concentrate more corrupted samples within the fixed review budget, with KNN-DW reaching the highest recall (87.29%) at a lower downstream macro-AUROC ( 0.8785±0.0075 ).

The representation-space ablation confirms the rationale for introducing the refinement head. Its purpose is not merely dimensionality reduction, but the construction of an embedding space whose local geometry is better aligned with label-set consistency scoring, especially under harder, locally plausible corruptions such as disjoint-label replacement. This motivation is particularly relevant because the frozen CoAtNet backbone was trained to predict the 14 thoracic findings without an explicit “No Finding” output, and therefore does not explicitly encourage “No Finding” studies to form compact neighborhoods in the original feature space. The refinement stage was introduced to obtain a more compact and audit-oriented representation, particularly for cases that are not naturally organized by the original 14-label multi-label classification task. This is most clearly reflected in LPV-DW-CS-280-A, whose performance deteriorates much more strongly on the raw 2048-dimensional backbone features than that of KNN-DW-LSS. Whereas LPV-DW-CS-280-A underperforms KNN-DW-LSS in the raw backbone space, it becomes the strongest method in downstream macro-AUROC after refinement. This behavior is consistent with the intended role of the refinement stage and indicates that prototype-based summarization is more dependent on embedding quality than direct kNN scoring.

The distance-normalization ablation further shows that the choice of normalization affects both method families, but not in the same way. For the direct kNN baselines, global normalization mainly improves Recall@budget, while downstream macro-AUROC changes are comparatively small. For LPV-DW-CS, by contrast, global normalization improves both Recall@budget and downstream macro-AUROC much more strongly. This indicates that prototype-based weighting is more sensitive than direct kNN scoring to how distances are scaled across neighborhoods, and supports the use of a global distance scale in the main experiments.

The neighborhood and clustering sensitivity analysis shows that Recall@budget is relatively stable for KNN-DW-LSS across moderate neighborhood sizes, whereas LPV-DW-CS is more sensitive to both k1 and k2 . Because this additional analysis was evaluated only with Recall@budget, it should be interpreted as a characterization of ranking sensitivity rather than as a basis for broader conclusions about downstream behavior.

The computational-cost benchmark should be interpreted together with the evaluation metric of interest. Direct kNN scoring remains the most efficient option, producing a complete ranking in under 10 s on the full dataset, whereas the LPV-DW-CS variants incur additional overhead from neighborhood clustering and prototype selection. This additional cost does not translate into uniformly better rankings: in our experiments, the kNN baselines remain stronger for Recall@budget, whereas LPV-DW-CS is more favorable when the objective is downstream macro-AUROC after filtering. Accordingly, the computational overhead of LPV-DW-CS should be understood relative to this more specific downstream benefit, rather than as a general improvement in ranking quality.

The current implementations were developed primarily to study ranking behavior and mislabel-detection performance rather than to optimize computational efficiency. Accordingly, the reported runtimes should be interpreted as representative of the present research implementation, not as optimized lower bounds for deployment.

Finally, the expert medical review complements the synthetic-noise evaluation by measuring how often top-ranked candidates correspond to expert-confirmed label-set disagreements under the image-only Likert protocol. Under the binarized mapping (Disagree/Strongly disagree as mislabeled), KNN-DW-LSS attains a higher expert-confirmed precision (60.5%; 23/38) than LPV-DW-CS-280-A (54.1%; 20/37), while both methods surface a substantial fraction of cases that the expert judged as mislabeled. The Likert distributions and cumulative Precision@k curves further characterize how confirmation yield evolves along each ranking.

These expert-review results should be interpreted with caution because the review used an image-only protocol. A disagreement with the dataset labels does not necessarily imply a true labeling error: the finding may be subtle or not clearly visible on the radiograph, and the original report may have been written with access to additional clinical information (e.g., patient history, prior studies, or other context) that was not available to the reviewer in this validation. Therefore, the expert-confirmed rates should be interpreted as a practical signal of auditing utility rather than a definitive estimate of annotation error prevalence.

# 7. Conclusions

We studied training-free mislabel detection for multi-label chest X-rays in a fixed CoAtNet-based embedding space. Neighborhood-consistency scores provide a practical ranking for expert review, without training a dedicated mislabel detector. We compared two training-free families: (i) kNN + clustering methods derived from our retrieval pipeline and (ii) direct kNN neighborhood-consistency baselines, including distance weighting and a Jaccard-based label-set similarity term for the multi-label setting. In the synthetic label-noise study, LPV-DW-CS consistently attains the highest downstream macro-AUROC after filtering, while kNN baselines more reliably maximize Recall@budget at the same review rate, highlighting a trade-off between maximizing recovery under a fixed audit budget and maximizing downstream test performance after removal.

Beyond ranking suspicious samples for targeted auditing, the results also suggest that filtering potentially corrupted samples can improve downstream performance even after removing part of the training data, indicating that a smaller but more reliable training subset may be preferable to retaining all available labels under noise. The representation-space ablation, distance-normalization ablation, neighborhood and clustering sensitivity analysis, and computational-cost benchmark further clarify how embedding quality, distance scaling, and method configuration affects both ranking behavior and downstream utility.

We also complemented the synthetic evaluation with an expert medical review under an image-only Likert protocol. Under the binarized agreement mapping, both method families surfaced substantial label-set disagreements among their top-ranked candidates (54.1% for LPV-DW-CS-280-A and 60.5% for KNN-DW-LSS on the reviewed subsets), supporting the practical utility of neighborhood-consistency rankings for targeted dataset auditing in multi-label CXR datasets.

Future work will explore adaptive selection of k1 and k2 , alternative prototype selection strategies, and stronger comparisons beyond neighborhood-based methods by incorporating non-kNN baselines for mislabel detection. Although such methods typically require additional training and therefore operate under different assumptions than our training-free scoring stage on fixed embeddings, they would provide an important external reference for contextualizing the practical value of neighborhood-consistency ranking in multi-label CXR auditing.

We also plan to validate these findings at larger scale on the full NIH Chest X-ray dataset and to run complementary sanity-check experiments on simpler benchmarks (e.g., CIFAR-style datasets) to better isolate failure modes of neighborhood-consistency scoring, even when the target task is not multi-label. Finally, we will explore and compare strategies for handling suspicious samples beyond removal, including automatic or semi-automatic relabeling, as well as methods for adaptively selecting the filtering threshold. We will also investigate hierarchy-aware extensions that exploit clinical label taxonomies to score disagreements at different semantic levels, potentially improving robustness when confusions occur between closely related pathologies.

# Acknowledgments

This work was carried out during the period of an internal doctoral scholarship within the Ph.D. Program in Electronic Engineering at Universidad Técnica Federico Santa María (UTFSM) and was supported by the Dirección de Postgrado (DP) at UTFSM. The authors also acknowledge support from ANID AC3E (CIA250006).

# Abbreviations

The following abbreviations are used in this manuscript:

# References

- Wang, X.; Peng, Y.; Lu, L.; Lu, Z.; Bagheri, M.; Summers, R.M. Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition ; IEEE: Piscataway, NJ, USA, 2017; pp. 2097–2106. [ Google Scholar ]

- Irvin, J.; Rajpurkar, P.; Ko, M.; Yu, Y.; Ciurea-Ilcus, S.; Chute, C.; Marklund, H.; Haghgoo, B.; Ball, R.; Shpanskaya, K.; et al. Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison. In Proceedings of the AAAI Conference on Artificial Intelligence ; AAAI Press: Palo Alto, CA, USA, 2019; Volume 33, pp. 590–597. [ Google Scholar ] [ CrossRef ]

- Karimi, D.; Dou, H.; Warfield, S.K.; Gholipour, A. Deep learning with noisy labels: Exploring techniques and remedies in medical image analysis. Med. Image Anal. 2020 , 65 , 101759. [ Google Scholar ] [ CrossRef ] [ PubMed ]

- Kompa, B.; Snoek, J.; Beam, A.L. Second opinion needed: Communicating uncertainty in medical machine learning. NPJ Digit. Med. 2021 , 4 , 4. [ Google Scholar ] [ CrossRef ] [ PubMed ]

- Northcutt, C.; Jiang, L.; Chuang, I. Confident learning: Estimating uncertainty in dataset labels. J. Artif. Intell. Res. 2021 , 70 , 1373–1411. [ Google Scholar ] [ CrossRef ]

- Zhu, Z.; Dong, Z.; Liu, Y. Detecting Corrupted Labels Without Training a Model to Predict. In Proceedings of the 39th International Conference on Machine Learning, 17–23 July 2022, Baltimore, MD, USA ; Chaudhuri, K., Jegelka, S., Song, L., Szepesvari, C., Niu, G., Sabato, S., Eds.; Proceedings of Machine Learning Research: Cambridge, MA, USA, 2022; Volume 162, pp. 27412–27427. [ Google Scholar ]

- Salvo, F.D.; Doerrich, S.; Rieger, I.; Ledig, C. An Embedding is Worth a Thousand Noisy Labels. Transactions on Machine Learning Research. arXiv 2025 , arXiv:2408.14358. [ Google Scholar ]

- Dai, Z.; Liu, H.; Le, Q.V.; Tan, M. Coatnet: Marrying convolution and attention for all data sizes. Adv. Neural Inf. Process. Syst. 2021 , 34 , 3965–3977. [ Google Scholar ]

- Ashraf, S.N.; Mamun, M.A.; Abdullah, H.M.; Alam, M.G.R. SynthEnsemble: A fusion of CNN, vision transformer, and Hybrid models for multi-label chest X-ray classification. In Proceedings of the 2023 26th International Conference on Computer and Information Technology (ICCIT) ; IEEE: Piscataway, NJ, USA, 2023; pp. 1–6. [ Google Scholar ]

- Patrini, G.; Rozza, A.; Krishna Menon, A.; Nock, R.; Qu, L. Making deep neural networks robust to label noise: A loss correction approach. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition ; IEEE: Piscataway, NJ, USA, 2017; pp. 1944–1952. [ Google Scholar ]

- Han, B.; Yao, Q.; Yu, X.; Niu, G.; Xu, M.; Hu, W.; Tsang, I.; Sugiyama, M. Co-teaching: Robust training of deep neural networks with extremely noisy labels. Adv. Neural Inf. Process. Syst. 2018 , 31 , 8527–8537. [ Google Scholar ]

- Arazo, E.; Ortego, D.; Albert, P.; O’Connor, N.; McGuinness, K. Unsupervised label noise modeling and loss correction. In Proceedings of the International Conference on Machine Learning ; Proceedings of Machine Learning Research: Cambridge, MA, USA, 2019; pp. 312–321. [ Google Scholar ]

- Kim, S.; Lee, D.; Kang, S.; Chae, S.; Jang, S.; Yu, H. Learning Discriminative Dynamics with Label Corruption for Noisy Label Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) ; IEEE: Piscataway, NJ, USA, 2024; pp. 22477–22487. [ Google Scholar ] [ CrossRef ]

- Lee, J.; Park, H.; Kim, M.; Yoon, J.; Yoo, K.; Byun, S.J. FastSimiFeat: A Fast and Generalized Approach Utilizing k-NN for Noisy Data Handling. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, New York, NY, USA, 21–25 October 2024; CIKM ’24. pp. 1143–1152. [ Google Scholar ] [ CrossRef ]

- Oakden-Rayner, L.; Dunnmon, J.; Carneiro, G.; Ré, C. Hidden stratification causes clinically meaningful failures in machine learning for medical imaging. In Proceedings of the ACM Conference on Health, Inference, and Learning ; Association for Computing Machinery: New York, NY, USA, 2020; pp. 151–159. [ Google Scholar ]

- Shi, J.; Zhang, K.; Guo, C.; Yang, Y.; Xu, Y.; Wu, J. A survey of label-noise deep learning for medical image analysis. Med. Image Anal. 2024 , 95 , 103166. [ Google Scholar ] [ CrossRef ] [ PubMed ]

- Guendel, S.; Grbic, S.; Georgescu, B.; Liu, S.; Maier, A.; Comaniciu, D. Learning to recognize abnormalities in chest x-rays with location-aware dense networks. In Proceedings of the Iberoamerican Congress on Pattern Recognition ; Springer: Berlin/Heidelberg, Germany, 2018; pp. 757–765. [ Google Scholar ]

- Lloyd, S. Least squares quantization in PCM. IEEE Trans. Inf. Theory 1982 , 28 , 129–137. [ Google Scholar ] [ CrossRef ]

# FiguresandTables

# html-copyright