# Sports vs Politics Classifier
**Problem 4 - Submission**

This is my project for classifying text documents as either **Sports** or **Politics**. I used a few different Machine Learning methods to see which one works best.

## Overview
The main idea was to see if we can tell these two topics apart just by looking at the words used. I used the **20 Newsgroups Dataset** and picked out the categories for:

- **Sports**: baseball and hockey
- **Politics**: guns, mideast, and misc politics

Everything was processed using a combination of techniques to find the best representation for text.

## Features & Representation
I used the following three main techniques for the analysis:
1.  **Bag of Words (BoW)**: Used as a baseline to count raw word frequencies.
2.  **TF-IDF**: Used as the primary representation to weight words by their importance and uniqueness across documents.
3.  **n-grams (Unigrams + Bigrams)**: I used `ngram_range=(1,2)` to help the models pick up on phrases like "white house" or "home run", not just single words.

## The Models I Tried
I evaluated four different models to compare them:

1. **Multinomial Naive Bayes**: A simple but very effective model for text.
2. **Logistic Regression**: A standard linear classifier.
3. **Linear SVM**: Using Stochastic Gradient Descent for speed.
4. **Random Forest**: An ensemble method using multiple decision trees.

## Results
Actually, all the models did pretty well, getting over 90% accuracy. Naive Bayes was usually the top performer in my tests.

| Model | Accuracy | F1-Score |
|---|---|---|
| **Naive Bayes** | **96.1%** | **0.97** |
| **Logistic Regression** | 95.5% | 0.95 |
| **Linear SVM** | 95.1% | 0.96 |
| **Random Forest** | 92.7% | 0.94 |

*Note: Results can change a tiny bit depending on the random split, but they're pretty consistent.*

## How to Run it
If you have the libraries installed, just run:

```bash
python M25CSE008_prob4.py
```

### What's needed:
- Python 3
- `scikit-learn`
- `numpy`

## Conclusion
It turns out that sports and politics use very different "vocabularies," so it's relatively easy for these models to tell them apart. Even simple models like Naive Bayes work great without needing anything too complex.
