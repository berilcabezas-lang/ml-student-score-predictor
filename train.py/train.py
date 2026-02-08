import pandas as pd
from sklearn.linear_model import LinearRegression

def main():
    data = pd.read_csv("students.csv")

    X = data[["hours"]]
    y = data["score"]

    model = LinearRegression()
    model.fit(X, y)

    hours = 7
    pred = model.predict([[hours]])[0]
    print(f"Predicted score for {hours} hours: {pred:.2f}")

if __name__ == "__main__":
    main()