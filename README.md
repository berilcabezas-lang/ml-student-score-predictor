# Student Score Predictor

This project demonstrates a simple machine learning workflow using Python.
It predicts exam scores based on the number of hours studied.

## Project Overview

The program:

1. Loads data from a CSV file
2. Prepares input and output variables
3. Splits data into training and testing sets
4. Trains a Linear Regression model
5. Evaluates the model
6. Predicts a score for new input
7. Visualizes the data with a plot

## Technologies Used

* Python
* Pandas
* scikit-learn
* Matplotlib

## Dataset

The dataset (`students.csv`) contains:

* Hours studied
* Exam score

Example:

hours,score
1,40
2,50
3,60

## How to Run

1. Install dependencies:

```
pip3 install pandas scikit-learn matplotlib
```

2. Run the program:

```
python3 train.py
```

3. Enter study hours when prompted to see a prediction.

## Example Output

Predicted score for 7 hours: 88.5

(Model results may vary slightly.)

## What I Learned

* Loading and processing CSV data with Pandas
* Training and evaluating a machine learning model
* Using train/test split and metrics
* Visualizing results with Matplotlib
* Version control with Git and GitHub

## Future Improvements

* Use larger datasets
* Add model saving
* Add logging and experiment tracking
