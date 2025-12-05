


**Research Topic:** xLSTM Architecture's For Recommendations 

**Research questions:**

This report aims to answer the following four research questions:

RQ1: How does xLSTM’s performance scale with dataset size compared to established architectures like BERT4Rec and SAS4Rec ?.

RQ2: How do sequence length and embedding size influence model performance across different item-popularity levels, and do larger sequences or embeddings improve a model’s ability to make accurate long-tail (less popular) recommendations ?.

RQ3: What trade-offs exist between recommendation accuracy and computational cost as sequence length and model complexity increase?.

RQ4: Embedding Saturation and Utilization: How do different model architectures make effective use of their embedding representations, and does embedding dimensionality lead to better spatial distribution, representation diversity, or improved predictive performance ?.

The primary objective is to evaluate the effectiveness of the xLSTM model across multiple datasets and benchmark it against state-of-the-art baselines using established ranking metrics.


**Data Flow Pipeline**

<img width="1013" height="767" alt="image" src="https://github.com/user-attachments/assets/ad9043b6-06a0-4398-8754-796d0f5e2b94" />

This script trains a sequential recommender system on a **user-specified/customizable MovieLens dataset** (100K, 1M, 10M, or 20M). It preprocesses the data, maps user/item IDs, and splits interactions with train/validation/test sequences. Users can select among **four AI models: standard LSTM, xLSTM, BERT4REC, SAS4REC** variant with configurable parameters. The selected model is trained using PyTorch with evaluation metrics like Recall@10, MRR, Hit Rate and NDCG. After training, the best model is used to predict and display top-10 movie recommendations based on user history.

**Training script that integrates:**

A. Dynamic dataset selection (100K, 1M, 10M, 20M) | B. Multiple model choices (LSTM, xLSTM, BERT4Rec, SASRec) | C. Dataset-specific hyperparameters (xlstm_params, dataloader_params)  |  D. TensorBoard logging  |  E. GPU monitoring  |  F. Evaluation metrics (Recall@10, MRR, NDCG)  | G. Early stopping + best model saving  | H. Easy-readable prediction logging with movie titles


**Initial Setup Requirements**

A. Install Necessary Packages (in quite mode) | B. Triton Activation For GPU Acceleration (To make sure Triton and GPU Accerleration, to speed up the training process) |  C. Select the necessary model and datasets (Model)  | D. Run all would work, to change the model and datasets, adjust the variable in the main script. | E. Script is mainly desinged for Colab Environment, for A100 GPU. Single click solution.

**Methodology**

**Experimental Setup::**

1. Datasets: MovieLens (100K, 1M, 10M, Steam)

2. Models Evaluated: xLSTM, BERT4Rec, SASRec

3. Custom configs for each dataset/model

4. Configuration: Custom hyperparameters tuned for each dataset-model combination

**Systems Features:**

1. GPU-accelerated training (NVIDIA A100, Triton-backed kernels for xLSTM), Comprehensive TensorBoard Logging.

2. Early stopping (patience = 3 epochs) with best model checkpointing

3. Real-time Top-K recommendation outputs with movie title

**Training Workflow:**
1. User and Item ID remapping for compact indexing
2. Temporal sequence splitting (Train/Validation/Test) 
3. Random seeds applied (42, 123, 2023) to ensure statistical reproducibility 
4. Early stopping triggered based on Recall@10 Improvments


**Requirements:** mlstm_kernels: 2.0.0 | xlstm: 2.0.4 | torch: 2.7.1 | torchvision: 0.22.1 | torchaudio: 2.7.1

**Total Training Hours**: 200 Hours (A100 GPU - 84 Experiments)

**Parameters**:

![image](https://github.com/user-attachments/assets/de0a6173-3172-44c6-88ed-f8fd4b403a1a)

**Folder: Best Models** ( Contains best model for inferencing, v4 is latest)

**Folder: Runs** (Contains all the recent Run History with 8 different performance attributes (Recall, Hit Rate, GPU Performance, Epoch Run Time, Total Parameters etc.)

**Model Results:**

<img width="757" height="638" alt="image" src="https://github.com/user-attachments/assets/9aa5c873-d542-41c1-a20a-3ae71b1dac1d" />

<img width="1116" height="736" alt="image" src="https://github.com/user-attachments/assets/3760389c-fe54-42c2-923b-3a2a6c28fd4f" />


**Overall Research Findings:**

RQ1 — Performance Scaling Across Dataset Sizes:  xLSTM demonstrates a clear positive scaling trend. While performance on the smallest dataset (ML 100K) drags the Transformer models, xLSTM significantly improves as the interaction histories emerge. On MovieLens 10M, xLSTM reaches Recall@10 values around 31.8 percent, converging closely with BERT4Rec, indicating that its gating mechanisms and the enhanced memory structures leverage medium scale datasets effectively. 

RQ2 — Effects of Sequence Length and Embedding Size: Experiments across sequence lengths of 32, 64, and 128 show that xLSTM exhibits increasing performance variance as sequences grow longer, reflecting its sensitivity to temporal window size. Unlike Transformer baselines, which often compress older interactions into dominant embedding directions, xLSTM maintains stronger temporal fidelity in long sequences due to its recurrent gating structure. This results in improved handling of long-term dependencies and enhanced differentiation of long tail items. Larger embedding dimensions further strengthen this effect on the large datasets, while offering limited benefit in sparse or in the short history domains (Table 5.1).

RQ3 — Accuracy vs. Computational Efficiency Trade-offs: xLSTM introduces a measured trade-off between accuracy and computational cost. Training times are typically 1.5×–2× longer than other baselines, and inference speed is moderate—faster than deep Transformer architectures yet slower than lightweight recurrent models. However, xLSTM avoids the quadratic attention bottleneck of Transformers, offering more predictable scaling in long sequences and large catalog sizes. Overall, xLSTM provides balanced accuracy efficiency characteristics across the datasets (Table 5.1).

RQ4 — Embedding Utilization, Saturation, and Representational Diversity: Embedding geometry analyses reveal that xLSTM makes substantially more effective use of embedding space than Transformer baselines. While BERT4Rec and SAS4Rec exhibits the anisotropic embedding structures driven by the popularity bias, xLSTM produces nearly isotropic embeddings with lower hubness, higher intrinsic dimensionality, and more uniform variance distribution. CKA similarity studies further show that the xLSTM learns fundamentally different, sequence-oriented embedding structures rather than compressing items along global similarity axes.

Overall, xLSTM demonstrates strong scaling behavior (RQ1), clear sensitivity to sequence length and embedding size (RQ2), meaningful efficiency‑accuracy trade-offs (RQ3), and superior embedding utilization compared to Transformer (RQ4). 



**Comprehensive Embedding Geometry Analysis for Sequential Recommenders:**

**L2 Norm:**

<img width="457" height="132" alt="image" src="https://github.com/user-attachments/assets/33c5f041-03d5-4a5b-822b-62f1d87fd85b" />


<img width="1088" height="692" alt="image" src="https://github.com/user-attachments/assets/53cf2f06-4f52-4b13-8e2a-1261640ec51e" />



{
  "BERT4Rec": {
    "vocab_size": 10678,
    "embedding_dim": 256,
    "num_items": 10677,
    "key": "bert.embeddings.word_embeddings.weight"
  },
  "SASRec": {
    "vocab_size": 10678,
    "embedding_dim": 256,
    "num_items": 10677,
    "key": "item_embedding.weight"
  },
  "xLSTM": {
    "vocab_size": 10678,
    "embedding_dim": 256,
    "num_items": 10677,
    "key": "embedding.weight"
  }
}

=== Intrinsic Dim / Anisotropy ===
BERT4Rec:  ID≈181.97 | mean cos=0.0154 (±0.0760)
SASRec:  ID≈204.45 | mean cos=0.0167 (±0.0720)
xLSTM:  ID≈249.88 | mean cos=0.0004 (±0.0625)

=== Hubness (k=10) ===
BERT4Rec: Gini=0.357  max=147  mean=10.00  std=7.56
SASRec: Gini=0.353  max=158  mean=10.00  std=7.82
xLSTM: Gini=0.179  max=24  mean=10.00  std=3.20

=== Cross-Model Similarity ===
BERT4Rec vs SASRec:  Jaccard@10=0.095  Jaccard@50=0.105  CosStructCorr=0.469  CKA=0.413  ProcErr=0.881
BERT4Rec vs xLSTM:  Jaccard@10=0.001  Jaccard@50=0.004  CosStructCorr=0.008  CKA=0.030  ProcErr=0.990
SASRec vs xLSTM:  Jaccard@10=0.001  Jaccard@50=0.003  CosStructCorr=0.006  CKA=0.029  ProcErr=1.018



		A. Embedding Spectrum Analysis 
		B. Variance Distribution & Intrinsic Dimension Study 
		C. Hubness and Popularity Bias Evaluation 
		D. t-SNE Embedding Space Visualization 
		E. Cross-Model Representation Geometry Comparison 
		F. Anisotropy and Isotropy Assessment 
		G. Neighborhood Structure Stability Analysis 
		H. Item Similarity Manifold Exploration


<img width="915" height="881" alt="image" src="https://github.com/user-attachments/assets/70a65ff3-0991-4c5d-a3b9-4c38d4e7477f" />


**Row 1 – Eigenvalue Decay (“spectrum”)**

The strength of each principal component in the embedding covariance.

Interpretation:

		1. BERT4Rec / SASRec decay very steeply → a few dominant directions → anisotropic space (information compressed in few axes).
		2. xLSTM’s curve is much flatter → variance spread across many dimensions → higher intrinsic dimension and better coverage of the vector space.
		3. Flat tail means embeddings retain more independent features.
		4. In Transformers, sharp decay often correlates with popularity or frequency bias.
		5. xLSTM therefore encodes items more uniformly and with richer latent diversity.

**Row 2 – Cumulative Explained Variance**

How many components are needed to explain total variance.

Interpretation:
		
		1. BERT4Rec and SASRec reach ≈ 90 % variance by ~50 dims → heavy redundancy.
		2. xLSTM needs ~200 dims for the same → more distributed information.
		3. A gentle slope indicates broader feature usage and less rank collapse.
		4. This confirms the intrinsic-dimension metrics (≈ 180 / 204 / 250).
		5. In summary, xLSTM = highest representational capacity, BERT4Rec/SASRec = more compact, redundant embeddings.

**Row 3 – Hubness Histograms (k = 10)**

How many times each item appears in other items’ top-10 nearest neighbors.

Interpretation:
		
		1. BERT4Rec / SASRec distributions are extremely right-skewed — a few movies appear hundreds of times ⇒ hub items dominate similarity space.
		2. xLSTM histogram is almost symmetric and much narrower — most items appear roughly equally often.
		3. Lower hubness (Gini ≈ 0.18) ⇒ better fairness and long-tail coverage.
		4. Transformer embeddings likely overfit to popular items.
		5. xLSTM yields a flatter similarity graph, enhancing diversity and mitigating popularity bias.


**Row 4 – t-SNE Projections**

A 2-D nonlinear projection of the 256-D embeddings (cosine distances).

Interpretation:
		
		1. BERT4Rec and SASRec form dense, elliptical blobs — embeddings crowd near a center → again anisotropy and hub formation.
		2. xLSTM plot is more evenly filled, points occupy a ring-like or diffuse shape → isotropy and balanced similarity.
		3. Fewer tight clusters means less genre-specific collapse; features are smoothly spread.
		4. Visually, xLSTM’s space is broader and more uniform.
		5. This geometry supports more stable neighbor retrieval across item types.

**Summary**

		A. BERT4Rec & SASRec: classic Transformer geometry — sharp spectral drop-off, anisotropy, hub dominance, overlapping t-SNE blob.
		B. xLSTM: near-isotropic, high-rank space with uniform neighbor frequency.
		C. xLSTM’s balanced variance explains its better diversity metrics and potentially more robust generalization.
		D. The difference in t-SNE and spectrum shapes shows fundamentally different inductive biases: attention models compress; xLSTM expands.
		F. Combining xLSTM with either Transformer (ensemble) could yield complementary strengths — one captures high-level correlations, the other preserves fine-grained variety.

Overall, Transformers (BERT4Rec, SASRec) learn narrow, popularity-biased manifolds; xLSTM learns a broad, isotropic embedding landscape — richer, fairer, and geometrically independent.


**A. Embedding anisotropy visualization:**

The Anisotropy Index (AI) measures how uniformly embeddings are distributed in space — it’s the mean cosine similarity between random pairs of vectors.
A low AI (≈0) means embeddings are evenly spread (isotropic), while a high AI (>0.05) means they point in similar directions (anisotropic), indicating reduced geometric diversity.

<img width="848" height="600" alt="image" src="https://github.com/user-attachments/assets/12fd0c69-fc86-4d50-81e4-2198fbbcf89d" />

BERT4Rec (AI = 0.0163) and SASRec (AI = 0.0164) show mild anisotropy — their movie embeddings tend to align toward a common direction, meaning popular movies cluster together in the same region.
In contrast, xLSTM (AI = 0.00035) produces an almost perfectly isotropic space, where movie vectors are well-spread and orthogonal.
Thus, xLSTM captures sequence-dependent uniqueness, representing each film (e.g., The Matrix, Titanic, Toy Story) in distinct, uncorrelated directions rather than emphasizing overall popularity.
This geometric diversity allows xLSTM to model temporal order and recency more effectively, explaining its superior recall despite higher embedding norms.


**B. Cosine structure correlation and CKA similarity:**

BERT4Rec vs SASRec | Corr=0.471 | CKA=0.427

BERT4Rec vs xLSTM | Corr=0.010 | CKA=0.030

SASRec vs xLSTM | Corr=0.008 | CKA=0.030

<img width="497" height="392" alt="image" src="https://github.com/user-attachments/assets/0cbea92a-720e-422d-98a1-126859451a66" />

<img width="1160" height="451" alt="image" src="https://github.com/user-attachments/assets/9f306212-39f4-440f-96b1-a4a634f6dbae" />

In the cross-model similarity study, BERT4Rec ↔ SASRec showed a cosine structure correlation of 0.471 and a CKA similarity of 0.427, indicating a strong geometric overlap. Both are Transformer-based models, meaning they learn comparable contextual movie embeddings where distances reflect semantic or co-occurrence similarity (e.g., similar genres or viewing patterns).

In contrast, BERT4Rec ↔ xLSTM (0.010, 0.030) and SASRec ↔ xLSTM (0.008, 0.030) revealed minimal structural similarity. The xLSTM embedding space is organized very differently — it doesn’t rely on global movie similarity but rather captures temporal and sequential dependencies. Thus, xLSTM represents movies based on their order and recency in user histories, not just shared context, which explains its distinct geometry despite strong recall performance.


**Data Sampling Techniques:**
				
				A. Random Sampling → Select a fraction p of rows uniformly (e.g., USING SAMPLE 1% in DuckDB); fast but may fragment user timelines.
				B. Stratified Sampling → Sample proportionally within groups (e.g., per user/item category); implemented with groupby + sample() in Pandas/Polars.
				C. Systematic Sampling → Pick every k-th record after a random offset; efficient for ordered files but risky if patterns exist.
				D. Time-based Sampling → Filter interactions by a timestamp window (e.g., WHERE ts >= NOW() - INTERVAL '90 days'); preserves temporal recency.
				E. User-based Hash Sampling → Deterministic subset of users via hash (e.g., hash(user_id) % 100 = 0); keeps complete histories of selected users.
				F. Per-user Last-K Sampling → Take last K events per user using window functions (ROW_NUMBER() OVER (PARTITION BY user ORDER BY ts DESC)); reduces data but preserves recency.
				G. Storage formats → For scale, write sampled subsets to Parquet/ZSTD (columnar, compressed) for fast reloads vs. raw TSV/CSV.
				H. Best practice → Use hash or last-K sampling for recommender research, time-based for evaluation splits, and combine with Parquet for speed.


**Data Sources:** Here, we will be leveraging RecBole libraries to explore various models and to develop more customizable one. 


-------------------------------------------------------------------------------------------------------------------------------------

**Model 1: Bert4Rec** Datasets: MovieLENS, Music4All, Amazon Software

**Model Architecture:**
1. The model is a modified version of BERT (Bidirectional Encoder Representations from Transformers).
2. It encodes the input sequence using multiple transformer layers:
      A. Each movie ID is converted into embedding vectors.
      B. Self-attention mechanism understands contextual relationships between movies.
      C. Outputs a dense layer that predicts probabilities for all possible movies.

**Making Predictions:**
During inference:
    The last few movies a user watched are fed into the model.
    The model predicts the most likely next movie (or multiple next movies).
Example:
    Input (last watched movies): [101, 102, 103]
    Output (predictions): 104 (Movie D), 105 (Movie E), 106 (Movie F)
    This means the model suggests that Movie D, E, or F are likely to be watched next.

**Generating Top-K Recommendations:**
After training, the model can generate recommendations for a given user:
Takes user’s past watched sequence.
Masks the last movie (or predicts the next).
The model outputs probabilities for all movies.
Top-K movies with the highest probability are recommended.

**Evaluation Metrics:** To evaluate the model accuracy Recall 5, 10, Precision, NDCG will be used. 

Recall = How many relevant items recommended/Total No. of relevant items **available** (measures the relevance. )

Precision: How many relevant items recommended//Total No. of items **recommended** (measures the accuracy.)

Normalized Discounted Combined Gain (NDGC): For Ranking Quality [DGC/IDGC].


**Cold Start Problem** :

Some of the commonly used approaches were:
1. Clustering Approach,
2. Profile Based (Meta Data) Approach,
3. Hierarchical approach, and
4. Novalty or Randomness Approach

**Performance Optimization:**

A. Implement Leave-One-Out Splitting, B. Integrate Negative Sampling.

**Logit Score:** Direct Score, before applying any activation funtions, non bounded ( can be larger and can go larger negative values). Higher the logit score, better the prediction is.

**Probability:** Derived from logit score after applying softmax function (always between 0 to 1), probability is calculated across all the Items in the list, so it might seem to be less, distributed across all of them. 

--------------------------------------------------------------------------------------------------------------------------------------

Model 2: xLSTM (Datasets: MovieLENS100-K, MovieLENS1M, MovieLENS20M, Music4All Datasets, Amazon Software Datasets)

Section 1: Model Parameters:

![image](https://github.com/user-attachments/assets/6b456f6f-c137-4782-a333-6a76fd5b0d58)


**Model Architecture:** 

xLSTM:

![image](https://github.com/user-attachments/assets/d442320b-f9a9-414a-87e6-e46a3f920e94)


**Model Architecture Hyperparameters at a glance:** (xLSTMLargeConfig — Advanced Configuration with Theoretical Justifications)

1. embedding_dim=128
	• What it does: Specifies the dimensionality of learned vector representations for discrete input tokens (e.g., movie IDs).
	• Why: A higher embedding dimension increases representational capacity, enabling the model to capture more latent semantic features. The embedding layer projects sparse one-hot input vectors into a continuous, dense space where semantic similarity correlates with vector proximity.
	• Common alternatives:
		○ 64: Lower capacity, faster convergence.
		○ 256/512: Useful in large item vocabularies to prevent underfitting.
	• When to use: Scale with dataset complexity. Use 128–256 for medium-size datasets with rich item metadata.
	Recommended: MovieLens 100K → 64 or 128, MovieLens 1M → 128 or 256, MovieLens 20M → 256 or 512
	

2. num_heads=2
	• What it does: Defines the number of parallel attention heads in multi-head attention modules within the xLSTM blocks.
	• Why: Multi-head attention decomposes the representation space into subspaces, allowing the model to attend to information from multiple perspectives simultaneously. This increases its ability to capture heterogeneous temporal dependencies.
	• Common alternatives:
		○ 1: Deactivates multi-head decomposition, reducing model complexity.
		○ 4/8: Enables modeling finer-grained patterns across modalities or positional contexts.
	• When to use: Increase when sequences are long or contain multiple intertwined dependencies (e.g., genre + recency + popularity).
	Recommended: MovieLens 100K → 1 or 2, MovieLens 1M → 2 or 4, MovieLens 20M → 4 or 8
	

3. num_blocks=2
	• What it does: Sets the number of stacked xLSTM layers.
	• Why: Deeper architectures allow hierarchical learning where lower layers capture local dependencies and higher layers model abstract, long-range patterns. This improves generalization and capacity to capture complex sequence dynamics.
	• Common alternatives:
		○ 1: Suitable for shallow tasks or small data regimes.
		○ 3+: Improves abstraction, suitable for deep sequence modeling like session-based or hierarchical recommendation.
	• When to use: Start with 2. Increase depth if the model underfits or fails to capture long-term user behavior trends.
	Recommended: MovieLens 100K → 1 or 2, MovieLens 1M → 2 or 3,  MovieLens 20M → 3 or 4
	

4. vocab_size=num_items + 1
	• What it does: Defines the size of the input vocabulary (items) including padding.
	• Why: Essential for allocating the correct size of embedding and output matrices. The +1 accounts for a sentinel token (e.g., <PAD>), critical for batching variable-length sequences.
	• When to use: Always match to dataset; padding index typically uses ID 0.
	

5. return_last_states=True
	• What it does: Returns only the final hidden state from each sequence.
	• Why: In next-item prediction, only the final timestep matters — intermediate states are irrelevant. Returning only the last state reduces memory and computational overhead during inference.
	• Alternative:
		○ False: Needed for token-level tasks or for attention over the whole sequence in downstream layers.
	• When to use: True for sequence-to-one settings; False for sequence-to-sequence or explainability requirements.
	Recommended: All MovieLens versions → True
	

6. mode="inference"
	• What it does: Controls the internal operation flags — disables dropout, gradient tracking, etc.
	• Why: Reduces unnecessary stochasticity and overhead during evaluation. Ensures deterministic behavior, important for reproducibility and deployment.
	• Alternative:
		○ "training": Activates regularization components like dropout.
	• When to use: Set to "inference" during evaluation or production deployment.
	Recommended: Use "training" when fitting the model. Use "inference" during evaluation or deployments.
	
7. chunkwise_kernel="chunkwise--triton_xl_chunk"
	• What it does: Specifies the backend kernel used for chunk-based sequence processing in the xLSTM.
	• Why: Chunking enables parallelism over sub-segments of the sequence, reducing latency and memory usage while preserving local context. Triton provides a high-performance, GPU-optimized kernel for this.
	• Alternatives:
		○ "chunkwise--native": CPU-friendly but slower and less parallelized.
	• When to use: Always prefer Triton if targeting GPU execution and high throughput.
	• All MovieLens versions → "chunkwise--triton_xl_chunk"
	

8. sequence_kernel="native_sequence__triton"
	• What it does: Determines the kernel used for processing full sequences end-to-end.
	• Why: In sequence modeling, kernel efficiency dictates overall throughput. Triton kernels can fuse operations and minimize memory transfers on GPUs.
	• Alternatives:
		○ "native_sequence__torch": More debuggable but less performant.
	• When to use: Triton for production/research; Torch for debugging and CPU contexts.
	• All MovieLens versions → "native_sequence__triton"
	

9. step_kernel="triton"
	• What it does: Kernel for token-by-token (autoregressive) prediction.
	• Why: In online inference, the model predicts one step at a time. Efficient step kernels minimize latency and memory reuse overhead.
	• Alternatives:
		○ "torch": Simplified fallback, better suited for testing and interpretability.
	• When to use: Triton in real-time systems or batch decoding tasks.


**To accelarate GPU Training Process:**

**1. AMP (Automatic Mixed Precision)**: It's a feature in PyTorch that enables training using a mix of:

		float16 (FP16) — faster, uses less memory
		
		float32 (FP32) — for stable parts of the model


**Training Objective + Optimizer + Scheduler Breakdown at a glance**

Step 1: criterion = nn.CrossEntropyLoss()
	• What it does: Defines the loss function used to measure how well the model’s predictions match the ground truth.
	• Why: CrossEntropyLoss is mathematically equivalent to maximizing the log-likelihood of the true class (movie index) in a multi-class classification setting. It's standard for categorical prediction tasks where only one true label exists.
	• How it works:
		○ Applies log(softmax(logits)) internally.
		○ Penalizes the model if the predicted probability for the true label is low.
	• Alternatives:
		○ nn.NLLLoss: Use with explicit log_softmax output.
		○ FocalLoss: For class-imbalance-sensitive training.
	• Recommended for MovieLens:
		○ MovieLens 100K → CrossEntropyLoss (default, reliable).
		○ MovieLens 1M / 20M → Still effective. Consider FocalLoss if popularity imbalance is extreme.

Step 2: optimizer = optim.Adam(model.parameters(), lr=0.001)
	• What it does: Specifies the optimizer that updates model weights based on computed gradients.
	• Why: Adam (Adaptive Moment Estimation) uses first- and second-order moments to adjust the learning rate per parameter. It converges faster and more stably than SGD in many cases.
	• How it works:
		○ Tracks moving averages of gradients and squared gradients.
		○ Adapts learning rate per parameter dynamically.
	• Alternatives:
		○ SGD: Simpler, requires more tuning.
		○ AdamW: Weight-decay decoupled Adam, more robust for regularization.
		○ RMSProp: Useful in recurrent networks, though less common now.
	• Recommended:
		○ MovieLens 100K → Adam(lr=1e-3)
		○ MovieLens 1M → AdamW(lr=3e-4)
		○ MovieLens 20M → AdamW(lr=1e-4) or scheduled warm-up

Step 3: scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
	• What it does: Decays the learning rate every 5 epochs by multiplying it by 0.5.
	• Why: Learning rate scheduling helps escape local minima early and encourages fine-tuning as training progresses. Reducing LR gradually allows stable convergence.
	• How it works:
		○ Epochs 1–5: LR = 0.001
		○ Epochs 6–10: LR = 0.0005
		○ ... continues halving every step_size
	• Alternatives:
		○ CosineAnnealingLR: Smoothly decays LR to a minimum.
		○ ReduceLROnPlateau: Adaptive decay based on validation loss.
		○ OneCycleLR: Aggressive LR scheduling, good for fast convergence.
	• Recommended:
		○ MovieLens 100K → StepLR(step_size=5, gamma=0.5) 
		○ MovieLens 1M → ReduceLROnPlateau(patience=3) or CosineAnnealing
		○ MovieLens 20M → OneCycleLR for faster training with controlled generalization

Step 4: recall_list, mrr_list, ndcg_list = [], [], []
	• What it does: Initializes lists to store evaluation metrics per epoch for validation and test sets..
	• Why: Tracking Recall@K, MRR@K, and NDCG@K helps monitor ranking quality and ensure model performance is improving.
	• How it works:
		○ After each epoch, predictions are collected.
		○ Top-k metrics are computed and stored.
	• Alternatives:
		○ Store in a dict or log with wandb, TensorBoard, etc.

Section 3: Evaluation Results and Predictions for MovieLENS10M:

<img width="1275" height="317" alt="image" src="https://github.com/user-attachments/assets/fc9064dc-3a23-4def-9026-bb60d411be3b" />


**Popularity bias:**

<img width="796" height="550" alt="image" src="https://github.com/user-attachments/assets/7b485c09-559c-42d2-aa8a-877763082e49" />

<img width="734" height="228" alt="image" src="https://github.com/user-attachments/assets/2465b706-bf99-44aa-b8ae-b4a69f2881a6" />

-----------------------------------------------------------------------------------------------

**General Classification of Recommender Systems:**

Sequential Recommendation (SR):- SR focuses on next-item prediction by modeling the temporal ordering of user interactions. These models utilize sequential data to capture evolving user preferences. RNN-based and transformer-based models are generally included in this category, and this is the primary research focus in this thesis work.

General Recommendation (GR):-  These models rely solely on user–item interaction data, typically in the form of implicit feedback. Implicit feedback includes signals that indirectly indicate user preferences, such as clicks, add-to-cart events, purchases, time spent, or interaction frequency.

Content-Aware Recommendation:- These models incorporate additional side information, such as user or item features. They are often applied in click-through rate (CTR) prediction tasks, using explicit feedback and binary classification evaluation. As feature-based methods, they often go beyond raw user–item interactions by including information about users, items, or context.

Knowledge-Based Recommendation:- Utilizes external knowledge graphs to add semantic or structural context beyond interactions.

**References:**

[1] xLSTM: Extended Long Short-Term Memory: https://arxiv.org/pdf/2405.04517

[2] xLSTM-Mixer: Multivariate Time Series Forecasting by Mixing via Scalar Memories: https://doi.org/10.48550/arXiv.2410.16928

[3] Amazon Science: https://github.com/amazon-science

[4] xLSTM Time : Long-term Time Series Forecasting With xLSTM: https://doi.org/10.48550/arXiv.2407.10240

[5] Quaternion Transformer4Rec: Quaternion numbers-based Transformer for recommendation: https://github.com/vanzytay/QuaternionTransformers

[6] Recommender Systems: A Primer: https://doi.org/10.48550/arXiv.2302.02579

[7] Exploring the Impact of Large Language Models on Recommender Systems: An Extensive Review: https://arxiv.org/pdf/2402.18590

[8] Recommender Systems with Generative Retrieval: https://openreview.net/pdf?id=BJ0fQUU32w

[9] Attention Is All You Need: https://arxiv.org/abs/1706.03762

[10] Recbole: https://recbole.io

[11] Group Lens: https://grouplens.org/datasets/movielens/100k/

[12] OpenAI: https://openai.com/

[13] Hugging Face: https://huggingface.co/docs/hub/en/models-the-hub

[14] Kreutz, C.K., Schenkel, R. Scientific paper recommendation systems: a literature review of recent publications. Int J Digit Libr 23, 335–369 (2022). https://doi.org/10.1007/s00799-022-00339-w

[15] Recommendation Systems: Algorithms, Challenges, Metrics, and Business Opportunities https://doi.org/10.3390/app10217748

[16] Roy, D., Dutta, M. A systematic review and research perspective on recommender systems. J Big Data 9, 59 (2022). https://doi.org/10.1186/s40537-022-00592-5

[17] Music4All — A Large-Scale Multi-Faceted Content-Centric Recommendation Dataset

[18] A Comprehensive Review of Recommender Systems: Transitioning from Theory to Practice https://doi.org/10.48550/arXiv.2407.13699

[19] Lost in Sequence: Do Large Language Models Understand Sequential Recommendation?: https://arxiv.org/pdf/2502.13909

----------------------------------------

Other Supporting Information:

**Scope** For General Recommendation Algorithm's In Various Sectors:-
1. **Energy Sectors**, (Energy Saving Programs, Substations, CO2 Emission, Solar, Grid Automation, Sensor Meters, Generators, Turbines, Smart Buildings, Smart Cities, Electrical Products and HVAC transmission)
2. **Healthcare and Pharmaceutical**, (Reconstruction kernels, AI post-processing algorithms, Preset Selection, clinical workflows, operational, diagnostic, device-focused, Imaging Modality, Multi-Model Protocol Recommendations, drug targets, clinical trial candidates, and molecule designs)
3. **Aerospace and Transportation**, (Test Procedure Recommendations Wind Tunnel, Engine Testing, Component & Subsystem Design Recommendations, Recommended temperature/pressure cycles for composites curing, Predictive Maintenance Recommendation, Quality, PQ Testing, Supply Chain and OEM Stocks recommendations)
4. Banking and Fintech sectors, (Wealth Management, Customized Credit Products, Investment Portfolio's, Equities, and Insurance plan recommendations)
5. Technology and Service sectors, (Ecommerce, products, content, or services, boosting engagement and customer satisfaction.) and Other Specialized Sectors. 

**List of GPU's Availability:** (For Model Training)

![image](https://github.com/user-attachments/assets/c19f0af2-17d4-49af-82e0-31900cb9ac12)






   
