# Text Review Classification (NLP)

## Problem
Classify text reviews into categories (such as positive or negative sentiment)
using Natural Language Processing (NLP) and machine learning techniques.

## Dataset
Text-based dataset containing user reviews.
- Each record consists of a text review and a corresponding label
- Data is split into training and testing sets

## Approach
This project applies a complete NLP pipeline for text classification.

Workflow:
1. Load and inspect text review data
2. Clean text (lowercasing, removing punctuation and stopwords)
3. Tokenize text data
4. Convert text into numerical features using vectorization
5. Train machine learning models on vectorized text
6. Evaluate model performance on test data

## Text Preprocessing
- Lowercasing
- Removing punctuation and special characters
- Stopword removal
- Tokenization

## Feature Extraction
- CountVectorizer / TF-IDF Vectorizer
- Converts text data into numerical feature vectors

## Models Used
- Logistic Regression
- (Other classifiers used in the notebook, if any)

## Evaluation
- Metrics used:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Performance evaluated on test data

## Results
The trained model successfully learns patterns from text reviews and achieves
good classification performance on unseen data.

## Files
- `text_review.ipynb` — complete notebook including:
  - Text preprocessing
  - Feature extraction
  - Model training
  - Evaluation

## Tools & Libraries
- Python
- Pandas, NumPy
- Scikit-learn
- NLTK (for text preprocessing)
- Matplotlib / Seaborn

## How to Run
1. Open the notebook:
   ```bash
   text_review.ipynb
