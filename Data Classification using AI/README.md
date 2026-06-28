# Data Classification Using AI - Project 2

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Classification-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Handling-lightgrey.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blueviolet.svg)](https://matplotlib.org/)

## Project Overview

This project demonstrates supervised learning on the Iris dataset using K-Nearest Neighbors. It covers data loading, preprocessing, train-test splitting, feature scaling, model training, validation, and metric analysis.

## What It Does

- loads the Iris benchmark dataset
- scales features with StandardScaler
- splits the data into 80% training and 20% testing sets
- trains multiple KNN models with different K values
- selects the best K using test accuracy
- prints a confusion matrix, classification report, accuracy, and weighted F1 score
- saves visualizations for the confusion matrix and K-value comparison

## Tech Stack

- Python
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## Dataset

The Iris dataset contains 150 samples across 3 balanced classes:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

Each sample has 4 measurements:

- Sepal length
- Sepal width
- Petal length
- Petal width

## Run the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the script:

```bash
python app.py
```

The script writes plots to an `artifacts/` folder next to the app file.

## Key Concepts Demonstrated

- data handling and preprocessing
- supervised classification
- KNN model training
- train-test validation
- confusion matrix analysis
- F1 score evaluation

## Expected Outcome

You should see a report similar to:

- dataset shape: `(150, 4)`
- training samples: `120`
- testing samples: `30`
- weighted F1 score close to `0.97`
- test accuracy close to `0.97`

## Repository Layout

```text
Data Classification using AI/
├── app.py
├── README.md
├── requirements.txt
└── hi.py
```

## Project Status

Status: Complete
Powered by: DecodeLabs
