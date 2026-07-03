# AI Recommendation Logic - Project 3

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Recommendation-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Handling-lightgrey.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blueviolet.svg)](https://matplotlib.org/)

## Project Overview

Project 3 introduces recommendation logic through a content-based filtering workflow. The system maps user skills to job-role attributes, applies TF-IDF weighting, computes cosine similarity, and returns the top career matches.

## Objectives

- collect three user skills or interests
- map user input to the same vocabulary as the item dataset
- score job roles using TF-IDF and cosine similarity
- rank the results and return the top three recommendations
- generate a visual summary of the recommendations

## Core Concepts

- content-based filtering
- vector mapping
- TF-IDF weighting
- cosine similarity
- top-N ranking
- cold start mitigation through profile-driven input

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- matplotlib

## Dataset

The recommender uses a curated dataset of 12 job roles spanning data science, cloud, development, DevOps, security, and mobile engineering.

## Run the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

When prompted, enter three skills separated by commas. If no input is provided, the script falls back to the sample profile used in the project documentation.

## Expected Output

The script prints:

- dataset shape
- available job roles
- similarity scores for each role
- top three recommended roles
- a detailed recommendation report
- the saved chart path inside `artifacts/`

## Repository Layout

```text
AI Recommendation Logic/
├── app.py
├── README.md
└── requirements.txt
```

## Project Status

Status: Complete
Powered by: DecodeLabs
