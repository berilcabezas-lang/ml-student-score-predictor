# Student Score Predictor (CSV → Pandas → ML)

Small beginner ML project: load data from a CSV file, train a model, and predict exam score from study hours.

## What it does
- Reads `students.csv`
- Splits data into:
  - X = hours studied
  - y = exam score
- Trains a Linear Regression model
- Predicts score for a given number of hours

## Tech used
- Python
- Pandas
- scikit-learn

## How to run
```bash
pip3 install pandas scikit-learn
python3 train.py