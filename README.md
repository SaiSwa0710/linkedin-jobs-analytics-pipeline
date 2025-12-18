# 📊 LinkedIn Jobs Analytics — End-to-End Data Engineering & Machine Learning Pipeline

**Dataset Size:** ~1.3 Million LinkedIn Job Postings  
**Primary Stack:** Apache Spark (PySpark), Hadoop, SBERT, PCA, PyTorch, GPU (NVIDIA T4)

---

## 1. Project Overview

This repository presents a **single, end-to-end data engineering and machine learning pipeline** built on large-scale LinkedIn job posting data.

The project intentionally spans **the full lifecycle** of a real-world data system:

- Distributed ingestion, cleaning, and EDA using **Apache Spark**
- GPU-accelerated **NLP embedding generation**
- Feature engineering and dimensionality reduction
- **Three distinct ML use cases**, evaluated honestly
- A rigorous analysis of **where ML works well — and where it fundamentally does not**

Rather than optimizing for leaderboard-style metrics, this project emphasizes:

> **Scalability, correctness, interpretability, and principled evaluation of model limits.**

This README is the **single curated entry point** for the entire project and subsumes all Phase-2 documentation.

---

## 2. Dataset Summary

The raw dataset consists of ~1.3M LinkedIn job postings with highly heterogeneous structure:

**Available fields include:**
- Job titles
- Company names
- Locations
- Long-form job descriptions
- Free-text skill lists
- Sparse and inconsistently formatted salary information

**Key challenges:**
- Large scale (cannot fit in memory)
- Noisy, semi-structured text
- Extremely sparse salary labels
- Millions of redundant skill variants

All heavy transformations are performed in Spark, and the outputs are stored as **ML-ready parquet datasets**, enabling downstream experimentation without repeatedly reprocessing raw data.

---

## 3. Repository Structure

```text
.
├── phase2.ipynb
│   └── Distributed Spark ETL, cleaning, and EDA
│
├── Dataset/
│   ├── raw/
│   │   └── Original LinkedIn job posting files
│   └── processed/
│       └── Cleaned, Spark-processed parquet outputs
│
├── embeddings/
│   ├── job_summary_embeddings_pca32.parquet
│   ├── job_title_embeddings_pca32.parquet
│   └── jobs_skills_canonical.parquet
│
├── notebooks/
│   ├── SBERT_extract_job_summary_embeddings.ipynb
│   ├── SBERT_skill_reduction.ipynb
│   ├── skill_prediction.ipynb
│   ├── seniority_prediction.ipynb
│   └── salary_regression.ipynb
│
├── models/
│   ├── skill_model.pt
│   ├── seniority_model.pt
│   └── salary_model_stable.pt
│
└── README.md
```

## 4. Phase 2 — Distributed Data Engineering with Spark

This phase focuses on building a **scalable, fault-tolerant data engineering pipeline** capable of processing ~1.3 million LinkedIn job postings. All heavy transformations and exploratory analysis are executed using **Apache Spark (PySpark)** to ensure performance and reproducibility.


### 4.1 Spark Setup & Environment

- Apache Spark (PySpark) running locally on Windows
- Hadoop-compatible filesystem layout
- JVM configuration aligned to avoid Spark driver/executor instability
- Spark UI used extensively to inspect DAGs, stages, and shuffle behavior

Spark is deliberately used **only where it provides leverage**: large-scale joins, aggregations, and EDA over wide datasets.


### 4.2 Spark ETL & Cleaning Pipeline

All core data transformations are executed in Spark to handle scale and schema variability:

- Schema normalization across heterogeneous raw input files
- HTML tag removal and text normalization
- Handling missing values and malformed rows
- Deduplication across job postings
- Flattening and exploding nested skill arrays
- Type-safe casting for downstream ML compatibility

The output of this stage is a **single, clean, joined Spark DataFrame** that serves as the system of record for all downstream embedding and ML tasks.

---

### 4.3 Performance Optimization Techniques

To ensure Spark remains performant at scale, the following optimizations are applied:

- Strategic use of `.cache()` and `.persist(MEMORY_AND_DISK)`
- Repartitioning to reduce data skew
- Avoidance of unnecessary wide shuffles
- Iterative Spark DAG inspection via Spark UI
- Disabling adaptive query execution where it caused instability

These choices significantly reduce recomputation and improve interactive EDA latency.

---

### 4.4 Exploratory Data Analysis (EDA)

EDA is performed using Spark SQL and DataFrame aggregations:

- Job count distribution by geography
- Role and seniority prevalence
- Long-tail analysis of skill frequencies
- Salary field sparsity and availability analysis

A key insight emerges during EDA:

> Salary information is extremely sparse and weakly correlated with job text.

This observation directly informs later ML design decisions.

---

## 5. Embedding Generation — GPU-Based NLP

Spark is not well-suited for deep NLP workloads. Therefore, **all embedding generation is performed outside Spark** using GPU-backed environments (Google Colab with NVIDIA T4).

Only **minimal, clean parquet outputs** from Spark are transferred to the GPU environment to reduce memory and I/O overhead.

---

### 5.1 Job Summary Embeddings

- **Model:** `paraphrase-MiniLM-L6-v2`
- **Input:** Full job descriptions
- **Original Dimensionality:** 384
- **Dimensionality Reduction:** PCA → 32
- **Output File:** `job_summary_embeddings_pca32.parquet`

PCA is used to retain semantic variance while enabling efficient downstream training.

---

### 5.2 Job Title Embeddings

- **Model:** `all-MiniLM-L6-v2`
- Optimized for short text phrases
- **Original Dimensionality:** 384
- **PCA Reduction:** 32
- **Output File:** `job_title_embeddings_pca32.parquet`

These embeddings capture role semantics and seniority cues effectively.

---

## 6. Skill Canonicalization Pipeline

Raw skill data contained **millions of noisy and redundant variants**, making direct ML infeasible.

### Canonicalization Steps

1. Flatten ~3M raw skill strings
2. Generate SBERT embeddings
3. Reduce dimensionality using PCA (32 dimensions)
4. Apply KMeans clustering (2000 clusters)
5. Select the skill closest to each cluster centroid
6. Assign this as the **canonical skill label**

### Output

- File: `jobs_skills_canonical.parquet`
- ~2000 standardized skills
- Structured, ML-ready skill representation

This step transforms unstructured skill noise into a usable semantic taxonomy.

---

## 7. Machine Learning Use Cases

Three ML problems are explored to evaluate the strengths and limits of text embeddings.

---

### 7.1 Use Case 1 — Canonical Skill Prediction  
**(Multi-Label Classification)**

**Objective:**  
Predict missing canonical skills for job postings with empty or incomplete skill lists.

**Input Features:**
- Job summary embeddings (PCA-32)
- Job title embeddings (PCA-32)

**Model Architecture:**
- PyTorch MLP
- Sigmoid outputs over ~2000 skills
- `BCEWithLogitsLoss`
- GPU training with automatic mixed precision (AMP)

**Outcome:**  
The model successfully learns semantic associations between job text and skills.

**Key Insight:**  
Text embeddings are well-suited for **taxonomy completion and data enrichment tasks**.

---

### 7.2 Use Case 2 — Seniority Prediction  
**(Multi-Class Classification)**

**Objective:**  
Predict job seniority levels (`intern`, `junior`, `mid`, `senior`) from job text.

**Why this works well:**
- Seniority is explicitly encoded in job titles and descriptions
- Keywords such as *Senior*, *Lead*, and *Entry-level* are preserved by SBERT
- PCA compression retains this semantic signal

**Outcome:**  
The model significantly outperforms baseline classifiers.

**Conclusion:**  
Job text is a strong predictor of seniority.

---

### 7.3 Use Case 3 — Salary Prediction  
**(Regression — Negative Result)**

**Objective:**  
Predict annual salary using job title and summary embeddings.

**Models Attempted:**
- Linear regression
- MLP regression
- Log-scaled targets
- Huber loss
- Dropout and regularization
- Output clamping for numerical stability

**Outcome:**  
All models perform **worse than predicting the mean salary** (negative R²).

This is a **consistent and reproducible negative result**.

---

## 8. Why Salary Prediction Failed — Critical Insight

This failure reflects a **data limitation**, not a modeling error.

### Salary vs Text Signal

| Aspect                       | Seniority | Salary |
|-----------------------------|-----------|--------|
| Explicitly stated in text   | High      | Low    |
| Captured by embeddings      | Strong    | Weak   |
| Depends on external factors | Low       | Very High |
| Label noise                 | Low       | High   |

**Major salary drivers absent from text:**
- Company size and prestige
- Industry sector
- Geographic cost of living
- Equity and benefits
- Market timing

LinkedIn job postings are often:
- Boilerplate-heavy
- SEO-optimized
- Salary-obfuscated
- Legally constrained rather than economically descriptive

**Conclusion:**

> Salary is not encoded in job text.  
> Text embeddings alone are insufficient for salary regression.

This is a **valid and meaningful scientific finding**.

---

## 9. Engineering Highlights

- Distributed Spark ETL at 1.3M+ row scale
- Memory-efficient parquet streaming
- GPU-accelerated training (NVIDIA T4)
- Automatic mixed precision (AMP)
- Stable training loops with NaN and overflow handling
- Clean separation between Spark and ML responsibilities
- Reproducible and modular pipelines

---

## 10. Key Takeaways

1. Representation matters more than model complexity
2. Semantic embeddings excel at classification, not numeric regression
3. PCA preserves variance, not economic signal
4. Negative results are valid scientific outcomes
5. Salary prediction requires structural and contextual features

---

## 11. Future Work

To make salary prediction viable:

- Use full 384-dimensional embeddings (no PCA)
- Incorporate company, location, and industry features
- Embed canonical skills directly
- Normalize salaries by geography
- Predict salary bands instead of point estimates
- Explore multi-task learning (seniority + salary)

---

## 12. Contributors

- **Sai Swapnesh**

---

## 13. Project Status

- ✔ Distributed Spark ETL complete  
- ✔ GPU-based NLP pipelines complete  
- ✔ ML use cases evaluated  
- ✔ Negative results rigorously analyzed  
- ✔ Demo-ready and capstone-quality  

---