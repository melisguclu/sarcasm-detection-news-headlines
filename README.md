# Sarcasm Detection in News Headlines 📰

This project uses the BERT model to classify whether a given news headline is sarcastic or not. It is based on the [News Headlines Dataset For Sarcasm Detection](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection), which contains over 28,000 headlines.

## 🔍 Dataset
The dataset contains two columns:
- `headline`: the news headline text
- `is_sarcastic`: 1 if sarcastic, 0 otherwise

The data is clean, balanced, and written in formal news language — avoiding common pitfalls of sarcasm datasets collected from Twitter (like informal text, labeling noise, and context dependency).

## 📦 Model Architecture
- **Model**: `bert-base-uncased` from HuggingFace
- **Tokenizer**: BERT tokenizer with padding and truncation
- **Fine-tuning**: 3 epochs on ~16,500 samples (80% train)
- **Evaluation metrics**: Accuracy, Precision, Recall, F1-Score

## 📊 Performance
| Epoch | Training Loss | Validation Loss | Accuracy | F1       | Precision | Recall   |
|-------|----------------|-----------------|----------|----------|-----------|----------|
| 1     | 0.0882         | 0.1725          | 0.9581   | 0.9536   | 0.9685    | 0.9391   |
| 2     | 0.0421         | 0.0979          | 0.9795   | 0.9775   | 0.9809    | 0.9742   |
| 3     | 0.0001         | 0.0993          | 0.9845   | 0.9830   | 0.9853    | 0.9807   |

Final training loss: 0.1268 after 3 epochs (~50 minutes training time on GPU).

