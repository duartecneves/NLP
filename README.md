# NLP

# Financial Tweet Sentiment Classification

Text Mining project, MSc Data Science & Advanced Analytics, NOVA IMS (2025/2026).

Classifies financial tweets as **Bearish, Bullish or Neutral**. The task is hard because the signal is subtle
("beats EPS" vs "misses EPS"), the vocabulary is domain-specific, and the data is imbalanced (65% Neutral).

## Approach
- **EDA:** class imbalance, vocabulary per class, cashtag/URL/emoji analysis, language checks
- **Preprocessing:** a full pipeline for classical models (regex cleaning, TweetTokenizer, stop-words,
  POS-aware lemmatisation, cashtag preservation) and a light pipeline for transformers
- **Features:** TF-IDF (unigrams, bigrams), Word2Vec, GloVe Twitter, FinBERT and RoBERTa embeddings
- **Models:** Logistic Regression, Random Forest, XGBoost, MLP; fine-tuned FinBERT, RoBERTa-base,
  Twitter-RoBERTa and GPT-2 with class-weighted loss
- **Evaluation:** stratified 80/20 split, macro-F1 as the main metric

## Results (validation set)
| Model | Macro-F1 |
|---|---|
| Best TF-IDF baseline (Logistic Regression) | 0.72 |
| Best frozen-embedding model (MLP) | 0.79 |
| Fine-tuned FinBERT | 0.82 |
| Fine-tuned RoBERTa-base (final model) | **0.87** |

Findings: end-to-end fine-tuning beat every classical approach, and the general-domain RoBERTa-base
outperformed the Twitter- and finance-specific models.
