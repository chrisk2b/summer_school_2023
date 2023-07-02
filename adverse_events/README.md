# Adverse Event Detection Model

## Introduction
This repository contains data for developing a model to detect the presence of adverse events in sentences.
The goal is to build and evaluate a model that can accurately identify whether a given sentence contains an adverse event.
Google Colab should be used. The relevant `pip` dependencies for the sample solution (cf. below) are in the `requirements.txt` file.
Please debug  potential issues if necessary.
An example of a solution for the adverse event detection engine can be found in this [notebook](adverse_events/notebooks). Please use it only if you don't know how to start.
Rather, focus on the provided data, cf. below and create your own solution in terms of a notebook.

## Getting Started
To begin, follow the steps below:

1. Clone the GitHub repository: `git clone https://github.com/chrisk2b/summer_school_2023.git`

2. Open a Google Colab Notebook on your computer.

3. In the repository, you will find a file under `adverse_events/data/ae_summer_school.json`. This file contains sentences which potentially contain adverse events, along with a label indicating if the sentence contains an adverse event or not.

4. Upload the file into your Colab environment.

## Goal
The main objectives of this project are as follows:

- Develop a model which can detect if an adverse event is present or not.
- Evaluate the performance of the model.

## Requirements
To successfully complete this project, please adhere to the following requirements:

- Try to keep things as simple as possible.
- Do not use `adverse_events/data/ae_summer_school_new.json`. This file will be relevant for part 2 of the coding exercise.

## Inference on new data
Once you have developed a model, you can proceed with the following steps:

1. Upload the data `adverse_events/data/ae_summer_school_new.json` into Colab.

2. Use the model developed in part 1 to perform inference on this dataset.

3. The new data contains the label as well. Ignore the label during inference, but use it to evaluate the performance of the model developed in part 1 on the new dataset.

4. Compare the model's performance on the new dataset with the performance on the previous dataset.

5. Discuss your results and observations based on the comparison and potential next steps
