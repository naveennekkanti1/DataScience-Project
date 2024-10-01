# 5C-Network
# Word Pair Similarity and Impression Generation with GPT-2

This repository implements a comprehensive pipeline for generating medical impressions using a fine-tuned GPT-2 model and analyzing word similarities through cosine similarity. It utilizes various libraries such as Transformers, NLTK, Sentence Transformers, and Plotly for data processing, model training, evaluation, and visualization.

## Table of Contents
1. [Installation](#installation)
2. [Data Source](#data-source)
3. [Usage](#usage)
4. [Code Structure](#code-structure)
   - [1. Model Fine-tuning](#1-model-fine-tuning)
   - [2. Model Evaluation](#2-model-evaluation)
   - [3. Text Analysis](#3-text-analysis)
   - [4. Visualization](#4-visualization)
5. [Contributing](#contributing)
6. [License](#license)

## Installation

To run this project, make sure you have Python 3.6 or later installed. You can install the required dependencies using pip:

```bash
pip install pandas numpy scikit-learn transformers torch nltk sentence-transformers matplotlib seaborn plotly evaluate rouge_score
```

# 1-model-fine-tuning
Importing necessary libraries and downloading required NLTK data files.
Loading the dataset and combining relevant columns to create a comprehensive text feature called "Impression."
Splitting the data into training and evaluation sets.
Loading a pre-trained GPT-2 model and its tokenizer.
Tokenizing the data and preparing it for training.
Defining a custom dataset class to facilitate the training process.
Setting up training arguments and using the Trainer API to fine-tune the model on the dataset.
