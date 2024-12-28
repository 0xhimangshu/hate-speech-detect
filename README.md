# Hate Speech Detection Model

This repository contains a machine learning model for detecting hate speech in text, using an ensemble approach combining multiple classifiers for robust performance.

## Overview

The system implements a sophisticated hate speech detection pipeline using natural language processing and machine learning techniques. It can identify hateful content while handling common evasion tactics like leetspeak.

## Key Features

- Robust text preprocessing and cleaning
- Advanced TF-IDF vectorization with n-grams
- Ensemble model combining multiple classifiers:
  - Logistic Regression with balanced class weights
  - Random Forest with 200 trees
  - Gradient Boosting with 200 estimators
- Cross-validation and detailed performance metrics
- Confidence scores for predictions
- Handling of leetspeak and text obfuscation
- Analysis of high-confidence misclassifications

## Technical Details

### Text Preprocessing
- Case normalization
- Leetspeak conversion (e.g. '4' → 'a', '3' → 'e')
- Whitespace normalization
- Unicode accent stripping

### Feature Engineering
The TF-IDF vectorizer uses:
- N-grams from 1-4 words
- Maximum 20,000 features
- English stop words removal
- Document frequency filtering (min_df=2, max_df=0.95)
- Sublinear TF scaling
- IDF smoothing

### Model Architecture

#### Ensemble Components
1. **Logistic Regression**
   - Balanced class weights
   - L2 regularization (C=1.0)
   - Maximum 1000 iterations

2. **Random Forest**
   - 200 trees
   - Maximum depth of 20
   - Balanced class weights
   - Parallel processing

3. **Gradient Boosting**
   - 200 estimators
   - Learning rate of 0.1
   - Maximum depth of 5

The ensemble uses soft voting to combine predictions.




