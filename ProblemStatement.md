# Social Media Performance Predictor - Multi-Brand Edition

## Context
You're given a dataset of ~375 Instagram posts across 5 beverage brands competing in the Indian market. The dataset includes: captions, media (video/images hosted on S3), engagement metrics (views, likes, comments, shares, engagement_rate), collaborators, posting metadata, and AI-generated visual summaries of the media. The brands span different content strategies—sports sponsorships, pop culture, comedy, festivals, music events—and what "works" varies significantly across them.

## Objective
Given any new post creative (image/video + caption + metadata), predict how it will perform—and explain why.

**Your system should:**
1.  Analyze the dataset to understand what drives engagement.
2.  Extract features from post content (text, visual, metadata).
3.  Predict whether a new post will perform well or poorly.
4.  Explain which factors are contributing to the prediction.

How you architect this—whether you treat all brands as one model, build per-brand models, use namespaces, ensembles, or something else entirely—is up to you. Justify your choices.

---

## Core Requirements

### 1. Prediction API
* Expose your system as an API.
* Accept a new post draft (caption + image/video + any relevant metadata) as input.
* Return a performance prediction with explanation.
* Handle edge cases gracefully (missing data, unseen brand styles, expired media URLs, posts with no views because they're static images not reels, etc.).

### 2. Demo Frontend
* Build a simple frontend where a user can:
    * Upload or link a post creative (image + text, or video + text).
    * Select/specify the brand context.
    * Get a prediction with explanation of what's driving it.
* This doesn't need to be beautiful; it needs to work and clearly show the prediction + reasoning.

---

## Evaluation Strategy
This is critical. AI systems need rigorous evaluation to be trustworthy.

* Design and document your evaluation approach. How do you know your predictions are meaningful and not just noise? 
* Show concrete evaluation results. Run your evaluation and present the output—metrics, examples, failure cases.
* **Consider at minimum:**
    * How you split data (given small N per brand).
    * What baseline you're comparing against (random? average? always-predict-medium?).
    * How you measure prediction quality.
    * Where your system fails and why.
    * How you'd improve evaluation with more data.

---

## Edge Cases to Handle
* **Views at 0:** This typically means it's a static post/album, not a reel. This is expected behavior, not missing data.
* **Expired URLs:** Media URLs on S3 may be expired or inaccessible—your system should handle this gracefully.
* **Wild Engagement Rates:** Rates vary wildly (0.05% to 100%+) due to follower count differences and content type dynamics.
* **Collaborators:** Some posts may have collaborators with very different audience sizes; consider how this affects engagement attribution.
* **Noise:** The dataset may contain noise or posts that don't belong to the brand's core strategy.
* **Low Data Brands:** A brand with very few posts should still produce a reasonable prediction (degrade gracefully, don't crash).

---

## Evaluation Criteria

| Criteria | Weight |
| :--- | :--- |
| Evaluation strategy & rigor | 25% |
| Modeling approach & justification | 25% |
| Feature depth & engineering creativity | 20% |
| Explanation quality (interpretability) | 15% |
| Code quality, API design & demo | 15% |

---

## Deliverables
* **Python Backend:** A backend (framework of your choice) with the prediction API. Share a GitHub repo.
* **Demo Frontend:** A frontend that lets a user upload a creative and get a prediction.
* **Dataset:** The provided `assignment-dataset.json`.
* **README.md describing:** 
    * How to run locally.
    * Architecture and modeling decisions/justifications.
    * Evaluation strategy, results, and failure analysis.
    * Key findings from dataset analysis.
    * Future improvements with 10x more data.
* **Loom Video (3-4 minutes):** 
    * End-to-end system demo.
    * Walkthrough of evaluation and results.
    * Explanation of key architectural/modeling decisions.

---

## Hints & Non-Requirements
* You are free to augment the dataset with synthetic data or additional scrapes; the provided data is a starting point, not a constraint.
* You are not restricted to the brands in the dataset; configure your system to work with any brand.
* You do **NOT** need to build a production ML model; a well-reasoned LLM-based approach, embedding similarity, or hybrid system is valid.
* You may use classical ML, LLM-based scoring, embedding similarity, rule-based scoring, or few-shot prompting.
* The `summary` field on media items already contains vision model output—use it.
* Focus on reasoning and evaluation rigor more than raw prediction accuracy.
* A simple approach done well beats a complex approach done poorly.

---

## Data Notes
* **Brands included:** cocacola_india, redbullindia, pepsiindia, sprite_india, thumsupofficial.
* Each post has an `_id` and a `data` object with all fields.
* Not all posts have all fields; handling missing data is expected.