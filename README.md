# NLP Assignment — 202301100032

**Student Name:** Shreyash Kumbhar  
**Student ID:** 202301100032  
**GitHub:** [ShreyashKumbhar/202301100032_NLP](https://github.com/ShreyashKumbhar/202301100032_NLP)

---

## Overview

This project applies a wide range of Natural Language Processing (NLP) techniques to a binary text classification task using the [20 Newsgroups](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html) dataset (`talk.religion.misc` category). Messages are classified as either **religious** or **general** based on their content.

Each NLP technique is implemented in two ways:
1. **From scratch** — to demonstrate a ground-level understanding of the algorithm.
2. **Using libraries** — to show proficiency with standard NLP tools (NLTK, spaCy, scikit-learn, Gensim).

---

## Dataset

- **Source:** `sklearn.datasets.fetch_20newsgroups` (subset: `talk.religion.misc`)
- **Filter:** Document IDs specified in `list.csv`
- **Binary Labels:**
  - `religious` — messages containing religion-related keywords (e.g., *god*, *jesus*, *prayer*, *faith*)
  - `general` — all other messages
- **Features engineered:** character count, word count, digit count, uppercase count, special character count, URL/quote presence

---

## Project Structure

```
202301100032_NLP/
├── 202301100032.ipynb   # Main Colab notebook
├── list.csv             # Document IDs used to filter the dataset
└── README.md
```

---

## Dependencies

```
pandas, numpy, matplotlib, seaborn
nltk
scikit-learn
gensim
spacy (en_core_web_sm)
```

Install with:
```bash
pip install gensim spacy
python -m spacy download en_core_web_sm
```

---

## Notebook Walkthrough

### 1. Setup & Imports
Installs and imports all required libraries. Downloads NLTK corpora (`punkt`, `stopwords`, `wordnet`, `averaged_perceptron_tagger`).

---

### 2. Dataset Loading & Labelling
- Loads document IDs from `list.csv`
- Fetches the corresponding messages from the 20 Newsgroups corpus
- Assigns binary labels (`religious` / `general`) via keyword matching
- Computes basic engineered features (word count, char count, etc.)

---

### 3. Exploratory Data Analysis (EDA)
- Label distribution (class balance)
- Message length and word count statistics per class
- Top words per class
- Visualizations: count plot, histogram, box plot, correlation heatmap → saved as `advanced_eda.png`

---

### 4. NLP Preprocessing — From Scratch
Implements a full preprocessing pipeline without external NLP libraries:
- **Tokenizer:** Lowercasing, punctuation removal, digit filtering
- **Stopword removal:** Custom manually defined stopword list
- **Stemmer:** Rule-based suffix stripping (e.g., `-tion`, `-ing`, `-ed`, `-ly`)
- **Feature extractor:** Detects numbers, uppercase, URLs, questions, exclamations

---

### 5. NLP Preprocessing — Using Libraries (NLTK)
Implements an advanced pipeline using NLTK:
- Lowercasing and regex-based cleaning
- Tokenization via `word_tokenize`
- Stopword removal using `nltk.corpus.stopwords`
- POS tagging via `pos_tag`
- Lemmatization with POS context using `WordNetLemmatizer`
- Optional Porter stemming

A side-by-side comparison of manual vs. NLTK-preprocessed text is printed.

---

### 6. POS Tagging
- **From scratch:** Rule-based tagger using suffix patterns (e.g., `-ing` → VBG, `-ly` → RB, `-tion` → NN)
- **NLTK:** `pos_tag` applied to compute POS tag distributions for both classes
- **Visualization:** Bar chart comparing POS tag frequencies (NN, VB, JJ, RB) between religious and general messages

---

### 7. Named Entity Recognition (NER)
- **From scratch:** Keyword and rule-based NER tagging categories such as PERSON, PLACE, ORGANIZATION, MONEY, and PROPER_NOUN
- **spaCy:** `en_core_web_sm` model used for entity extraction
- **Visualization:** Bar chart of entity type distribution per class

---

### 8. Stemming vs. Lemmatization Comparison
- Demonstrates differences between Porter stemming and WordNet lemmatization on a set of sample words
- Applies the lemmatization pipeline to the full dataset to produce a `lemmatized_processed` column

---

### 9. Text Vectorization — Word2Vec
- Trains a Word2Vec model (skip-gram, vector size 100, window 5, 15 epochs) on lemmatized text using Gensim
- Explores semantically similar words for keywords: *god*, *church*, *faith*, *prayer*
- Computes word-pair similarity scores (e.g., `god ↔ jesus`, `faith ↔ belief`)
- Generates sentence-level vectors by averaging word vectors

---

### 10. N-Gram Analysis
Three approaches are compared:
- **From scratch:** Sliding window over token list
- **NLTK:** `FreqDist` over `ngrams`
- **scikit-learn:** `CountVectorizer` with `ngram_range=(2,2)`

Top bigrams and trigrams are reported for both classes. A bar chart shows the most frequent religious bigrams.

---

### 11. Text Similarity
- **From scratch:** Bag-of-words vectorization + manual cosine similarity formula
- **TF-IDF + scikit-learn:** `TfidfVectorizer` + `cosine_similarity` to find the top-5 most similar messages to a query

---

### 12. Classification Models

#### Manual Naive Bayes (from scratch)
Implements Multinomial Naive Bayes with Laplace smoothing using only Python built-ins.

#### scikit-learn Models
Three classifiers are trained with two vectorization strategies:

| Model               | CountVectorizer | TF-IDF |
|---------------------|:--------------:|:------:|
| Naive Bayes          | ✅             | ✅     |
| Logistic Regression  | ✅             | ✅     |
| Linear SVM           | ✅             | ✅     |

Each combination is evaluated on Accuracy and Macro F1 Score.

---

### 13. Evaluation & Visualizations
- **Confusion matrices** for all 6 model/vectorizer combinations (+ manual Naive Bayes) → saved as `confusion_matrices.png`
- **Grouped bar chart** comparing Accuracy and F1 Score across all models
- **Summary table** with the best-performing model highlighted in green

---

## Results Summary

The best model is selected by highest Macro F1 Score across all model × vectorizer combinations. A detailed classification report is printed for the winner, and a colour-highlighted summary table is displayed.
