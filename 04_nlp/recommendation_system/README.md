# Recommendation System (Content-Based)

## Problem
Build a recommendation system that suggests similar items based on their
textual content.
This is a content-based recommendation system where recommendations are
generated using text similarity.

## Dataset
Text-based dataset where each item contains descriptive text.
- Text features are used to represent items
- Recommendations are based on similarity between item descriptions

## Approach
This project uses Natural Language Processing (NLP) techniques to build
a content-based recommender.

Workflow:
1. Load and inspect the dataset
2. Clean and preprocess text data
3. Convert text into numerical features using CountVectorizer
4. Compute similarity between items using cosine similarity
5. Recommend top similar items based on similarity scores

## Text Processing
- Lowercasing
- Removing punctuation and special characters
- Tokenization
- Stopword removal (if applied)

## Feature Extraction
- CountVectorizer (Bag of Words)
- Converts text data into numerical vectors

## Recommendation Method
- Cosine similarity
- Items with the highest similarity scores are recommended

## Output
- List of recommended items similar to the selected item
- Similarity scores used for ranking recommendations

## Files
- `prj16Recomandation_system.ipynb` — complete notebook implementation

## Tools & Libraries
- Python
- Pandas, NumPy
- Scikit-learn
- NLTK (for text preprocessing)
- Matplotlib (if used for analysis)

## How to Run
1. Open the notebook:
   ```bash
   prj16Recomandation_system.ipynb
