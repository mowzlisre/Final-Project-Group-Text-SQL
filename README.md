# Project Overview

This project is designed for building and evaluating models for SQL and context-based predictions using Advance Natural Language Processing technics, BERT base-uncased model and T5ForConditionalGeneration models. This readme file explores the project structure to provide sufficient information on maintaining and scaling

## Directory Structure

### **data/**
This folder contains the dataset(s) used in the project.

- `synthetic_data.csv`: A synthetic dataset used for training and evaluation purposes.

### **lib/**
This directory contains core library files that define utility functions and feature extraction processes.

- `label_extract.py`: A script for extracting labels from the dataset before training of BERT model
- `similarity.py`: Contains functions to compute similarity scores and mapping relevant synonyms

### **models/**
This folder houses the pre-trained model files and tokenizer configurations.

- `BERT_SQL_TOKENIZER`: The tokenizer used for preprocessing SQL-related inputs.
- `BERT_SQL_MODEL.pth`: The pre-trained model specifically for SQL-based tasks.
- `context_model.pth`: A pre-trained model for context-based predictions.

If you wish to train the model. Please ignore this.

### **support/**
This directory contains the synthetic data generation script using Ollama pipeline and Open Llama3 model from Meta

- `generate_data.py`: A utility script to generate synthetic datasets
- `req.txt`: Please use this text file to install all the dependencies required for this project in a safe virtual environment. This default install Ollama, PyTorch, Huggingface, nltk and related support libraries.

### **Root Folder ./**

- `Label_Trainer.ipynb`: This notebook guides you through the process of training the BERT model used to perform Multilabel Classification.
- `Context_Trainer.ipynb`: This notebook guides you through the process of training the T5 model to perform generative predictions
- `Label_Predictor.py`: Python script to predict labels from a give Natural Language Text
- `Context_Predictor.py`: Python script to predict SQL syntax from a give Natural Language Text and its relevant labels
- `schema.txt`: A sample ext file defining the schema of the dataset. This can be a template how the pipeline idetifies the Schema structure

---

## Setup Instructions

1. Clone the repository to your local machine.
2. Install the required dependencies using the following command:
   ```bash
   pip install -r support/req.txt
   ```
3. Ensure the `data/synthetic_data.csv` file is present or download from the [Google Drive](https://drive.google.com/drive/folders/1ulMSMRsLgu1n5v-YyM-FYb7eSUUEP4iZ?usp=sharing). Alternatively you can generate a synthetic dataset using `generate_data.py`.

---

## Usage

### Training Models
- Use `Context_Trainer.ipynb` to train the context prediction model.
- Use `Label_Trainer.ipynb` to train the label extraction model.

### Making Predictions
If you don't intend to train and only predict, please download the pth model files from the [Google Drive](https://drive.google.com/drive/folders/1ulMSMRsLgu1n5v-YyM-FYb7eSUUEP4iZ?usp=sharing).
- Run `Context_Predictor.py` for context-based predictions.
- Run `Label_Predictor.py` for label-based predictions.

### Dataset
Ensure the dataset follows the structure defined in `schema.txt`.

---

Feel free to reach out for further clarification or contributions!
If you face any errors, please raise an issue.
This project is a collection of knowledge from multiple internet sources. If you feel any part of your code is included, please [contact us](mailto:mowzlisre.mohandass@gwu.edu)

<span style="color:red;font-weight:bold">The Google Drive links are restricted to the George Washington Univeristy 
Organization. Please raise a request to access the resources.</span>
