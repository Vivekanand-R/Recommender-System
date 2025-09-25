
**Research Topic:** xLSTM Architecture's For Recommendations [Streaming Applications]

This script trains a sequential recommender system on a **user-specified/customizable MovieLens dataset** (100K, 1M, 10M, or 20M).

It preprocesses the data, maps user/item IDs, and splits interactions with train/validation/test sequences.

Users can select among **four AI models: standard LSTM, xLSTM, BERT4REC, SAS4REC** variant with configurable parameters.

The selected model is trained using PyTorch with evaluation metrics like Recall@10, MRR, Hit Rate and NDCG.

After training, the best model is used to predict and display top-10 movie recommendations based on user history.


**Data Flow Pipeline**

<img width="1013" height="767" alt="image" src="https://github.com/user-attachments/assets/ad9043b6-06a0-4398-8754-796d0f5e2b94" />


**Training script that integrates:**

A. Dynamic dataset selection (100K, 1M, 10M, 20M)

B. Multiple model choices (LSTM, xLSTM, BERT4Rec, SASRec)

C. Dataset-specific hyperparameters (xlstm_params, dataloader_params)

D. TensorBoard logging

E. GPU monitoring

F. Evaluation metrics (Recall@10, MRR, NDCG)

G. Early stopping + best model saving

H. Easy-readable prediction logging with movie titles


**Few Research Focus Areas/Questions:**

1. Embedding Saturation and Utilization: Are larger embeddings really helping the model learn better user/item relationships, or are they underutilized?

2. Gradient Stability / Exploding Gradients: Do longer sequences introduce more instability or gradient explosion?

3. Computation-Time vs Performance Trade-off: At what point does longer sequence input hurt speed more than it helps accuracy?

4. Effective Sequence Length vs. Truncation: How much of the input sequence is actually contributing to predictions?

5. Overfitting Signals: Do longer sequences encourage memorization rather than generalization?

6. Top-k Diversity / Coverage: Does sequence length affect recommendation diversity or item popularity bias?

7. Token Usage Heatmap: Where in the sequence is the model focusing? More recent items or early ones?

8. Ablation Logging: How much performance drop occurs when certain features are turned off?


**Initial Setup Requirements**

A. Install Necessary Packages (in quite mode)

B. Triton Activation For GPU Acceleration (To make sure Triton and GPU Accerleration, to speed up the training process)

C. Select the necessary model and datasets (Model)

D. Run all would work, to change the model and datasets, adjust the variable in the main script.

E. Script is mainly desinged for Colab Environment, for A100 GPU. Single click solution.

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


**Requirements:**
mlstm_kernels: 2.0.0
xlstm: 2.0.4
torch: 2.7.1
torchvision: 0.22.1
torchaudio: 2.7.1

**Total Training Hours**: 200 Hours (A100 GPU - 84 Experiments)

**Parameters**:

![image](https://github.com/user-attachments/assets/de0a6173-3172-44c6-88ed-f8fd4b403a1a)

**Folder: Best Models** ( Contains best model for inferencing, v4 is latest)

**Folder: Runs** (Contains all the recent Run History with 8 different performance attributes (Recall, Hit Rate, GPU Performance, Epoch Run Time, Total Parameters etc.)

**Model Results:**

<img width="811" height="562" alt="image" src="https://github.com/user-attachments/assets/2ced07c9-dfa2-40f6-ba76-1d003a190a99" />


**Conclusion from Final Results:**

A. xLSTM evaluated under a novel configuration for Sequencial recommenders; observed performance on various conditions.

B. Performance Scaling (RQ1): xLSTM matches BERT4Rec's Recall@10 (~26-27%) on the 1M dataset, indicating scalability with richer interaction histories. Performance converges  as dataset size grows.

C. Sequence Sensitivity (RQ2): Standard deviation increases with sequence length, underscoring sensitivity to input length variations.

D. Trade-offs (RQ3): xLSTM achieves competitive accuracy on large datasets but incurs higher computational costs, especially in smaller-scale scenarios.

**E. From 10M Datasets:**

	🔸 xLSTM: 
 	Dominates all three major metrics — this model is clearly the best choice.
	For longer sequences (e.g., 128+), xLSTM maintains advantage over transformer-based models.
	Performs well at all sequence lengths, Slightly best at 64, but stable performance overall

	🔸 BERT4Rec:
	In 10M datasets, Solid performance, but consistently lower than xLSTM. 
	Shows a performance dip at Seq Len = 64:
	Recall@10 drops to 0.2744 (vs. 0.3112 and 0.3171 at 32 and 128)
	Best at Seq Len = 128, but still behind xLSTM
	
	🔸 SAS4Rec:
	Performance drops drastically as sequence length increases:
	Recall@10 drops from 0.2143 → 0.1271 → 0.0727
	This trend suggests SAS4Rec is not able to capture long-term dependencies effectively
	Also shows lower parameter count, but at a cost of worse accuracy


---------------------------------------

**Data Flow (At High Level)**

<img width="660" height="838" alt="image" src="https://github.com/user-attachments/assets/6bf85359-c8ed-400d-99ab-7fd04e323cdc" />

---------------------------------------

# Recommender Systems: (In Detail)

Audio Podcast Version: https://www.dropbox.com/scl/fi/zv511ysp0ecdaqbo9nskp/Recommender-Systems_-Architectures-Applications-and-Market-Analysis.wav?rlkey=3u9za3bbogvc0506ubxohxe2w&st=gy3ekapc&dl=0

Estimated Read Time: 10-15 Minutes

Introduction to Recommender Systems:

At its core, a **recommendation engine** uses computer algorithms to predict and suggest items of interest to users based on their past behaviors and contextual data. On the deep learning front, xLSTM (extended LSTM) and Transformers have been evolved with latest architectures in recent years and plays a very important role in Large Language Models. 

Since recommenders have became an essential part of our daily digital experiences, in this paper we will be leveraging XLSTM architecture based recommenders and will be comparing the results with other modern architectures like Autotransformers, Recurring Neural Networks (RNN) and other Matrix Factorization Methods. xLSTM incorporates architectural enhancements like attention mechanism, gating improvments and bidirectional capabilities. It will be impactful due to several unique aspects when to compared to the tradional successful methods. 

**Existing Approaches:**
**Transformers Based:** Uses the self-attention mechanism from transformer architectures (like in GPT or BERT) to model the user-item interactions. It also captures the complex sequential patterns in user behavior (e.g., purchase history or clicks) and finally makes personalized recommendations by understanding the contextual relationships between items and users. Unlike, RNN which process information sequencially (one step at a time), transformer process information in parallel utilizing the self attentio mechanism which results in faster computation precerving long term dependencies.

A transformer-based recommender system uses an embedding layer to convert the users and items into dense vector representations. It applies self-attention layers to capture complex dependencies and sequential relationships in user-item interactions, enhanced by positional encodings to preserve and keep the order of actions. The final output layer computes scores or rankings to predict the most relevant items for each user.

**Matrix Factorization Based:** It decompose a user-item interaction matrix into two smaller matrices: one representing users and the other representing items. These matrices capture latent factors (hidden patterns) that explain the user preferences and item characteristics. By reconstructing the original matrix, the system predicts how much a user might like an unseen item. Some techniques include SIngular Value Decomposition (SVD), Non-Negative Matrix Factorization (NMF) and Probabilistic Matrix Factorization (PMF).

**Proposed Hybrid Methods & Noval Approaches:** xLSTM (extended LSTM) incorporates architectural enhancements like attention mechanism, gating improvments and bidirectional capabilities. It will be impactful due to several unique aspects when to compared to the traditional successful methods. 

Value Proposition: The core concept and unique benefit is still the same **recommend the best similar product/service to the end user** and to help the users. 

It is also estimated that the **market scope** for recommender system is expected to be approx 20-28 billion by 2030, currently in 2024 valued approx 6 billion US dollars. 

Below are the two major types in Recommenders:
1. Collaborative Filtering (User to User), and
2. Content Based Filtering (Product to Product).

Different Simple methods to identify user similarities:
1. Correlations, 2. Cosine Similarities, 3. Jaccard Similarities, 4. Euclidean Distance, 5. Hamming Distance, 6. Manhatten Distance, 7. Bhattachryya Distance, 8. Neural Network Embeddings (Collaborative Filtering), 9. Kullback Leibler divergence, 10. Embeddings and Latent Features, 11. Sequence-Based Similarity, 12. Deep Collaborative Filtering with Embeddings (via Neural Networks), 13. Transformer Models for Sequential Recommendations (e.g. BERT4Rec), and 14. Other Hybrid Approches.


**Datasets:**

1. Amazon Software data usually refers to the large-scale Amazon Product Review datasets (reviews, ratings, timestamps, and product metadata) widely used for recommender system research. They capture user–item interactions across millions of products and enable benchmarking of collaborative and content-based recommendation models.

2. MUSIK4all (JKU) is a massive music interaction dataset (≈228M events, 119,140 users, TSV File) containing user–track play counts and timestamps. It is designed for music recommender systems, supporting temporal modeling, user history analysis, and large-scale evaluation.

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

List of Different Varieties of Datasets: (Ref: https://recbole.io/)

![image](https://github.com/user-attachments/assets/e842adf0-6eaa-48b7-9ffa-68312db0788e)

-------------------------------------------------------------------------------------------------------------------------------------

**Model 1: Bert4Rec** Datasets: MovieLENS, Music4All, Amazon Software

**Model Architecture:**
1. The model is a modified version of BERT (Bidirectional Encoder Representations from Transformers).
2. It encodes the input sequence using multiple transformer layers:
      A. Each movie ID is converted into embedding vectors.
      B. Self-attention mechanism understands contextual relationships between movies.
      C. Outputs a dense layer that predicts probabilities for all possible movies.

![image](https://github.com/user-attachments/assets/64fbe331-7fec-4787-84d5-dbfd55f9ba95)

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

Instead of Top-K, Sequencial Recommendation needs to be done, considering the timestamp.    

**Performance Optimization:** Comparision and Performance Results For All Approaches:
1st set of results (To be fine tuned further)

![image](https://github.com/user-attachments/assets/6f6ab94b-4b13-4fc4-a7ff-9fab78e2baa1)

**Interpretations:**

1. Recall@10 = 0.2932, Meaning: In ~29.3% of cases, the true next item appears in the top 10 predictions. Solid result for a baseline xLSTM model. It means if we recommend 10 movies, ~3 times out of 10, the correct one will be in that list.

2. MRR@10 = 0.1266, Mean Reciprocal Rank: Measures the position of the first correct item in the top-10 list. On average, the correct item appears around position 8 (1/0.1266 ≈ 7.9). Higher MRR means better ranking of correct items.

3. NDCG@10 = 0.1655, Normalized Discounted Cumulative Gain: Measures both relevance and position in the ranked list. A higher value means the model is not just retrieving the right item, but ranking it closer to the top. At ~16.5% of the optimal ranking gain. Good enough for an initial run,better results when its finetunes over 200 epochs. 

**Evaluation Metrics:** To evaluate the model accuracy Recall 5, 10, Precision, NDCG will be used mainly. 

Recall = How many relevant items recommended/Total No. of relevant items **available** (measures the relevance. )

Precision: How many relevant items recommended//Total No. of items **recommended** (measures the accuracy.)

**Normalized Discounted Combined Gain (NDGC):** For Ranking.

**Epochs:** How many times we process our complete data until we reach final/optimum goal. 

**Learning rate:**, How fast did we adjust our weights to reach that optimum level.

**Cold Start Problem** (For new users when we don't have data, 192 users): 
Some of the commonly used approaches were:
1. Clustering Approach,
2. Profile Based (Meta Data) Approach,
3. Hierarchical approach, and
4. Novalty or Randomness Approach


**Feature Focus:**

A. User-level features (e.g., user age, gender, occupation)
B. Item-level features (e.g., genres, title metadata)
C. Time or position embeddings beyond fixed positional indices
D. Temporal dynamics (e.g., timestamp-based recency)

**Performance Optimization:**

A. Implement Leave-One-Out Splitting, B. Integrate Negative Sampling.

   
List of Models:

![image](https://github.com/user-attachments/assets/d157ae60-54c2-41e8-9bf3-e79e1250bc1b)

Sample Output:
![image](https://github.com/user-attachments/assets/ae41618e-a913-44c1-80d6-76d15a3faba3)


**Logit Score:** Direct Score, before applying any activation funtions, non bounded ( can be larger and can go larger negative values). Higher the logit score, better the prediction is.

**Probability:** Derived from logit score after applying softmax function (always between 0 to 1), probability is calculated across all the Items in the list, so it might seem to be less, distributed across all of them. 

--------------------------------------------------------------------------------------------------------------------------------------

Model 2: xLSTM (Datasets: MovieLENS100-K, MovieLENS1M, MovieLENS20M, Music4All Datasets, Amazon Software Datasets)

Section 1: Model Parameters:

![image](https://github.com/user-attachments/assets/6b456f6f-c137-4782-a333-6a76fd5b0d58)

![image](https://github.com/user-attachments/assets/10382bb0-d290-4994-a9f3-2906e38f19d8)

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
	Recommended: Use "training" when fitting the model. Use "inference" during evaluation or deployment.
	
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


**Which movies dominate the top-10 predictions across the test set?**

To understand:

# Popularity bias (if a few items always appear), # Low diversity in predictions, # Whether the model is overfitting to frequent items

**First, To study the popularity bias:**

<img width="796" height="550" alt="image" src="https://github.com/user-attachments/assets/7b485c09-559c-42d2-aa8a-877763082e49" />

100K: Models are strong on head (popular) items but underperform on long-tail (diverse) items. For recommendation systems, this can mean: Poor personalization, Repetition of already-known items, Missed opportunities in user engagement.

Datasets: Can be used: The Amazon software dataset is a dataset for the Amazon software product segment, similar to the Amazon dataset, which also contains user purchase and review information. [https://arxiv.org/html/2504.05323v1]

**Baseline Model Comparision:**

![image](https://github.com/user-attachments/assets/0c85e244-aedd-4423-8138-46ef2494ca84)

-----------------------------------------------------------------------------------------------

**Recbole - Major Classifications:**

Four Classifications: 

1. General Recommendation (GR): Netflix use case which discussed above. The interaction of users and items is the only data that can be used by model. Trained on implicit feedback data and evaluated using top-n recommendation. Collaborative filter (CF) based models are classified here. 

2. Content-aware Recommendation: Amazon use case. Click-through rate prediction, CTR prediction. The dataset is explicit and contains label field. Evaluation conducted by binary classification.

3. Sequential Recommendation: Spotify, similar to time series problem, which we discussed earlier. The task of SR (next-item recommendation) is the same as GR which sorts a list of items according to preference. History interactions are organized in sequences and the model tends to characterize the sequential data. Session-based recommendation are also included here.

4. Knowledge-based Recommendation: Knowledge-based recommendation introduces the external knowledge graph to enhance general or sequential recommendation.

SEO (Search Engine Optimization) and SEM techniques may also be merged, along with Google Adsense and adwords, to improve user experience further. 

**Scope** For Recommendation Engines In Various Sectors:
1. Energy Sectors, (Energy Saving Programs, Substations, CO2 Emission, Solar, Grid Automation, Sensor Meters, Smart Buildings, Smart Cities, Electrical Products and HVAC transmission)
2. Banking and Fintech sectors, (Wealth Management, Customized Credit Products, Investment Portfolio's, Equities, and Insurance plan recommendations)
3. Technology and Service sectors, (Ecommerce Products)
4. Entertainment and Gamification, (Custom Localization and Immersive Experience)
5. Food, Beverages & Agriculture Industry (AI-based recommendations for improved crop yield, agricultural products, custom fertilizers, supply and demand forecasting, as well as weather and climate change insights using satellite data.)
6. Healthcare and Pharmaceutical, 
7. Aviation and Transportation, and 
8. Other Specialized Sectors. 


**Industrial Applications:**

<img width="722" height="247" alt="image" src="https://github.com/user-attachments/assets/11539db4-9604-4a0e-957c-0f460d5d8eb6" />


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


----------------------------------------

**List of GPU's Availability:**

![image](https://github.com/user-attachments/assets/c19f0af2-17d4-49af-82e0-31900cb9ac12)








   
