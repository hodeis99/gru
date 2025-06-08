# GRU for Backorder Prediction - Supply Chain Forecasting

A high-performance GRU model for backorder prediction in supply chain management, featuring advanced time-series processing of tabular data.

## Key Features

- **GRU Architecture**: 2-layer Gated Recurrent Unit network optimized for sequential data
- **Time-Series Transformation**: Converts tabular data to 10-step sequences
- **Class Imbalance Handling**: Custom class weighting (2.5:1) and AUC-focused training
- **Production-Grade Training**: Gradient clipping, batch normalization, and dropout
- **Comprehensive Evaluation**: Precision, Recall, AUC, and Confusion Matrix

## Dataset Requirements

Preprocessed CSV files from the data cleaning pipeline:
- `Train_Preprocess.csv` (balanced dataset)
- `Test_Preprocess.csv` (original distribution)

## Installation

```bash
git clone https://github.com/hodeis99/gru.git
cd gru
pip install -r requirements.txt
