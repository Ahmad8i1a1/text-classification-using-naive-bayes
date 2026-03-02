# Multinomial Naive Bayes for Text Classification

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NLP](https://img.shields.io/badge/Field-NLP-green.svg)](https://en.wikipedia.org/wiki/Natural_language_processing)

## 📌 Project Overview
This repository contains a from-scratch implementation of the **Multinomial Naive Bayes** algorithm applied to text classification. Developed as part of my academic research into Natural Language Processing (NLP), this project demonstrates the application of probabilistic generative models to categorize text based on word frequency distributions.

## 🚀 Key Features
- **Custom Preprocessing:** Implements RegEx-based tokenization, noise reduction, and text normalization.
- **Laplace Smoothing:** Integrated additive smoothing ($\alpha = 1$) to handle the "Zero-Probability" problem for unseen tokens.
- **Vectorized Logic:** Optimized for performance using NumPy for matrix operations.
- **Mathematical Integrity:** Built directly from the Maximum A Posteriori (MAP) estimation principle.

## 📊 The Mathematics
The classifier calculates the probability of a class $C$ given a document $d$ using Bayes' Theorem:

$$P(C|d) \propto P(C) \prod_{i=1}^{n} P(w_i|C)$$

To prevent arithmetic underflow with small probabilities, the implementation utilizes the **Log-Likelihood** space:

$$\log P(C|d) = \log P(C) + \sum_{i=1}^{n} \log P(w_i|C)$$



## 🛠️ Installation & Usage
### Prerequisites
- Python 3.9+
- NumPy
- Scikit-learn (for evaluation metrics)

### Setup
```bash
git clone [https://github.com/Ahmad8i1a1/text-classification-using-naive-bayes.git](https://github.com/Ahmad8i1a1/text-classification-using-naive-bayes.git)
cd text-classification-using-naive-bayes
pip install numpy scikit-learn
