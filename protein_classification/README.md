# Protein Classification Model

## Introduction
This repository contains data for developing a model to classify proteins into different categories. The goal is to build and evaluate a model that can accurately classify proteins based on their amino-acid sequences. Google Colab should be used for development and training. The relevant `pip` dependencies for one possible solution (cf. below) are listed in the notebook [here](protein_classification/notebooks/protein_classification.ipynb). 
You do not need to look at the this notebook. Just if you don't know hot to start, feel free to look at the notebook.

## Getting Started
To begin, follow the steps below:

1. Clone the GitHub repository: `git clone https://github.com/chrisk2b/summer_school_2023` and navigate to the folder with the name `protein_classification`.

2. Open a Google Colab Notebook on your computer.

3. In the repository, you will find a file under `protein_classification/data`. This file contains protein data with associated features of a protein (sequence of amino-acids) and categories reflecting where the protein is located.
4. Upload the file into your Colab environment and transform it e.g. into a pandas dataframe.

## Goal
The main objectives of this project are as follows:

- Develop a model that can accurately classify proteins into their respective categories. To simplify the task, consider only 2 categories, namely if a protein will be located in the cell membrane or not.
- Evaluate the performance of the model on a test dataset.

## Requirements
To successfully complete this project, please adhere to the following requirements:

- Try to keep things as simple as possible.
- Illustrate that the model is able to learn.
-  Reflect about the analogies to the NLP use case for the adverse event detection. 

## Compute Power
If your Colab environment is not supporting GPUs, training can take a lot of time. We suggests to use  GPUs for this use case and to reduce the number of samples in the provided dataset.