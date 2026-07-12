# DecodeLabs Internship Repository

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Handling-lightgrey.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blueviolet.svg)](https://matplotlib.org/)

## Repository Summary

This repository contains four completed internship projects developed for the DecodeLabs AI training track. The first project demonstrates a deterministic rule-based chatbot implemented with Streamlit. The second project demonstrates supervised learning on the Iris dataset using a K-Nearest Neighbors classifier. The third project demonstrates a content-based recommendation engine built with TF-IDF and cosine similarity. The fourth project demonstrates image and text recognition using OCR and MobileNet SSD object detection.

Each project is maintained in its own directory so the implementation, documentation, and dependencies remain clearly separated.

## Projects Included

### Project 1: Rule-Based AI Chatbot

Location: [Rule-Based AI Chatbot/](Rule-Based%20AI%20Chatbot)

This project implements a white-box chatbot using explicit if-elif-else logic, session state, and audit logging. It is intended to demonstrate deterministic control flow, traceability, and IPO-style processing.

Run instructions:

```bash
pip install -r "Rule-Based AI Chatbot/requirements.txt"
streamlit run "Rule-Based AI Chatbot/app.py"
