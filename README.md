# Topic Modeling of Hebrew Tweets

This repository documents the process of topic modeling on Hebrew tweets, including data cleaning, topic extraction, classification, and validation. Below is a detailed overview of each step and the associated files.

## Overview

0. **EDA**: The first thing I did was to gather insights from the data with different kinds of simple processing and graphs.

1. **Data Preparation**: Cleaned the data by removing frequently occurring words and performed lemmatization to normalize words to their base forms.

2. **Topic Modeling**: Applied language models to generate a list of topics from the cleaned data.

3. **Classification**: Classified several hundred tweets into the generated topics, creating a dataset of tweets and their associated topics.

4. **Validation**: Used various validation methods to assess the accuracy of topic representation.

5. **Improvement Suggestions**
   - I plan to use a language model specifically trained for Hebrew to improve topic representation.
   - I would apply embeddings directly to tweets to identify similar ones without relying solely on classification.
   - I was considering fine-tuning BERT or similar models for more effective topic extraction and event identification.
   - And much much more :-)

## Files and Notebooks

### 1. Data Analysis and Modeling Notebook
   - **Filename**: `twitter-topic-modeling.ipynb`
   - **Description**: This notebook contains the analysis of the data. Also, in this notebook I generated the topics.

### 2. Topic Modeling Script
   - **Filename**: `main.py`
   - **Description**: This Python script creates topics.

### 3. Topic Validation Notebook
   - **Filename**: `Topic_Validation.ipynb`
   - **Description**: This notebook performs validation of the topic modeling process and evaluates the accuracy of topic representation.

Note: I ran the notebooks on with Google Colab with GPU A100.
