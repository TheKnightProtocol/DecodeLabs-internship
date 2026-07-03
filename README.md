# DecodeLabs Internship Repository

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Handling-lightgrey.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blueviolet.svg)](https://matplotlib.org/)

## Repository Summary

This repository contains three completed internship projects developed for the DecodeLabs AI training track. The first project demonstrates a deterministic rule-based chatbot implemented with Streamlit. The second project demonstrates supervised learning on the Iris dataset using a K-Nearest Neighbors classifier. The third project demonstrates a content-based recommendation engine built with TF-IDF and cosine similarity.

Each project is maintained in its own directory so the implementation, documentation, and dependencies remain clearly separated.

## Projects Included

### Project 1: Rule-Based AI Chatbot

Location: [Rule-Based AI Chatbot/](Rule-Based%20AI%20Chatbot)

This project implements a white-box chatbot using explicit if-elif-else logic, session state, and audit logging. It is intended to demonstrate deterministic control flow, traceability, and IPO-style processing.

Run instructions:

```bash
pip install -r "Rule-Based AI Chatbot/requirements.txt"
streamlit run "Rule-Based AI Chatbot/app.py"
```

### Project 2: Data Classification Using AI

Location: [Data Classification using AI/](Data%20Classification%20using%20AI)

This project implements a supervised learning workflow for the Iris dataset. It includes data loading, feature scaling, train-test splitting, KNN model training, evaluation, confusion matrix analysis, and metric reporting.

Run instructions:

```bash
pip install -r "Data Classification using AI/requirements.txt"
python "Data Classification using AI/app.py"
```

### Project 3: AI Recommendation Logic

Location: [AI Recommendation Logic/](AI%20Recommendation%20Logic)

This project implements a content-based filtering pipeline for job-role recommendations. It maps user skills to a shared vocabulary, scores roles using TF-IDF and cosine similarity, and returns the top career matches.

Run instructions:

```bash
pip install -r "AI Recommendation Logic/requirements.txt"
python "AI Recommendation Logic/app.py"
```

## Technical Stack

- Python
- Streamlit for the chatbot interface
- scikit-learn for machine learning
- pandas and numpy for data handling
- matplotlib and seaborn for visualisation and ranking charts

## Repository Structure

```text
DecodeLabs/
├── README.md
├── LICENSE
├── app.py
├── requirements.txt
├── Data Classification using AI/
│   ├── app.py
│   ├── README.md
│   ├── requirements.txt
│   └── hi.py
├── AI Recommendation Logic/
│   ├── app.py
│   ├── README.md
│   └── requirements.txt
└── Rule-Based AI Chatbot/
    ├── app.py
    ├── README.md
    ├── requirements.txt
    └── rule_based_chatbot.py
```

## Notes

- The root-level files are retained for repository convenience, while the project-specific implementations are maintained inside their respective folders.
- Generated files such as `artifacts/`, `__pycache__/`, and log files are excluded from version control through `.gitignore`.

## Status

All projects are complete and available in the repository.
