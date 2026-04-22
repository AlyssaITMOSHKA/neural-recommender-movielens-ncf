# Movie recommendation system

## Project overview

This project implements and evaluates a movie recommendation system using the MovieLens 1M dataset.   It compares with a deep learning-based model (neural collaborative filtering) with a classical collaborative filtering approaches. The goal is to understand how neural architectures improve recommendation quality over traditional baselines.

---

## Dataset

I used the MovieLens 1M dataset, which contains:

- 6,040 users  
- 3,900 movies  
- 1,000,209 ratings  

### Data sources:
- `movies.dat` — movie metadata  
- `users.dat` — user profiles  
- `ratings.dat` — user-item interactions  

Additionally, we use the NCF benchmark split:
- `train.rating`
- `test.rating`
- `test.negative`

Each test sample contains:
- 1 positive item  
- 99 negative sampled items  

---

## Models

### 1. Neural collaborative filtering (NCF)

I implemented the NeuMF architecture, combining:

- 🔹 GMF (Generalized Matrix Factorization)
- 🔹 MLP (Multi-Layer Perceptron)

#### Architecture highlights:

- User & item embeddings
- Non-linear feature interaction via deep MLP
- Concatenation of GMF + MLP representations
- Final prediction layer with sigmoid/logit output

#### Training details:
- Loss: `BCEWithLogitsLoss`
- Optimizer: `Adam`
- Embedding size: 32
- Layers: 3-layer MLP
- Negative sampling: 1 negative per positive

---

### 2. Factorization Machine Baseline

A classical baseline using:

- Sparse feature encoding via `DictVectorizer`
- Linear model trained with SGD (`log-loss`)

This model captures pairwise feature interactions only.

---

## Evaluation Protocol

We evaluate models using Top-K recommendation metrics:

### Metrics:

- Hit Ratio@K (HR@K)
  Measures whether the true item is in the top-K list

- NDCG@K
  Measures ranking quality (position-aware relevance)

---

## Results

| Model                | HR@10  | NDCG@10 |
|---------------------|--------|---------|
| Factorization Machine (SGDClassifier) | 0.0215 | 0.0105 |
| Neural Collaborative Filtering (NCF)  | 0.6702 | 0.3966 |

---

## Key findings

- NCF significantly outperforms classical FM baseline
- Deep non-linear interactions greatly improve ranking quality
- Classical linear models struggle with sparse user-item interactions
- Neural embeddings capture richer user preferences

---

## Training Behavior

- NCF shows stable convergence across epochs
- Loss decreases consistently with negative sampling
- Training monitored per epoch with loss visualization

---

## Visualizations

The project includes rating distribution analysis, item popularity (long-tail effect), training loss curves per epoch.

