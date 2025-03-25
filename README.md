# NLP Text Cleaning and Processing Pipeline

## Project Overview
This project presents a **modular and reusable Natural Language Processing (NLP) pipeline** designed for **text preprocessing and classification** tasks. The core functionality includes **text cleaning, tokenization, stopword removal, stemming, vectorization, and classification using machine learning models**.

The solution is implemented in **Python (Jupyter Notebook environment)** using **NLTK and SpaCY** libraries for linguistic preprocessing and **Scikit-learn** for model development and evaluation.

The pipeline is built to be **adaptable to any text classification problem**, making it suitable for use in **spam detection, sentiment analysis, chatbot preprocessing, and other NLP applications**.



## Tech Stack
- **Language:** Python  
- **Environment:** Jupyter Notebook  
- **Libraries:**  
  - [NLTK](https://www.nltk.org/) – Natural Language Toolkit for linguistic preprocessing 
  - [SpaCY](https://spacy.io/) – open-source library for natural language processing (NLP) in Python
  - [Scikit-learn](https://scikit-learn.org/) – Machine learning algorithms and model evaluation  
- **Data Format:** CSV (can be easily adapted for JSON, Excel, or text inputs)  



## Data Source
- Dataset: [Spam Text Message Classification](https://www.kaggle.com/team-ai/spam-text-message-classification)  
- Format: `.csv` with labeled text data  
- Target Variable: `label` (spam/ham)  
- Features: `message` (raw SMS text)



## NLP Pipeline Highlights

The pipeline includes **ready-to-use modular functions** for:

1. **Text Cleaning**
   - Lowercasing
   - Removing punctuation, numbers, and special characters
   - Stripping white spaces

2. **Tokenization**
   - Converting text into individual tokens (words)

3. **Stopword Removal**
   - Filtering out common non-informative words

4. **Stemming**
   - Reducing words to their base/root form

5. **Vectorization**
   - Using TF-IDF or CountVectorizer to transform cleaned text into numeric format

6. **Modeling**
   - Text classification using:
     - Logistic Regression
     - Naive Bayes
     - Support Vector Machines (SVM)
   - Model evaluation using:
     - Accuracy
     - Confusion Matrix
     - Precision, Recall, F1-score



## Business Applications
- Spam detection and filtering  
- Email classification and routing  
- Sentiment analysis for social media monitoring  
- Customer support automation (chatbot preprocessing)  
- Document classification in compliance systems



## Final Notes
- The pipeline is **modular**, making it **easy to extend** or plug into larger NLP systems.
- You can **integrate this notebook into web applications** or REST APIs for live classification.
- Future enhancements could include:
  - Lemmatization
  - Named Entity Recognition (NER)


