This project performs sentiment analysis on airline-related tweets using various machine learning models, with final deployment of a **Stacking Classifier** via **Flask**. It predicts whether a given tweet expresses **positive**, **neutral**, or **negative** sentiment.

## 🚀 Project Highlights

- **Text Preprocessing** with custom cleaning, tokenization, and TF-IDF vectorization.
- **Imbalanced Dataset Handling**: ~70% negative, 20% neutral, 10% positive.
- **Model Comparison**:
  - Logistic Regression, SVM, Naive Bayes, Decision Tree, Random Forest
  - Bagging, Gradient Boosting, XGBoost, and a final **StackingClassifier**
- **Evaluation Metrics**: Accuracy, Recall (especially for majority class), F1-score
- **Deployment**: Flask app with live prediction interface
- **Notebooks**: For EDA, model training, and evaluation

## 🧠 Final Model Performance (Stacking Classifier)

| Sentiment | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Negative (0) | 1.00 | 1.00 | 1.00 |
| Neutral (1) | 0.88 | 0.83 | 0.85 |
| Positive (2) | 0.79 | 0.84 | 0.81 |

**Overall Accuracy**: 94%  
