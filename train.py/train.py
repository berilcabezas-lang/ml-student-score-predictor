import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def main():
    # Load data
    df = pd.read_csv("students.csv")

    # Basic checks
    required_cols = {"hours", "score"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}. Found: {set(df.columns)}")

    # Prepare features/labels
    X = df[["hours"]]
    y = df["score"]

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"MAE (lower is better): {mae:.2f}")
    print(f"R² (closer to 1 is better): {r2:.2f}")

    # User prediction
    hours = float(input("Enter study hours to predict score: "))
    pred_score = model.predict([[hours]])[0]
    print(f"Predicted score for {hours:.1f} hours: {pred_score:.2f}")

    # Plot (data + fitted line)
    plt.scatter(df["hours"], df["score"])
    x_line = pd.DataFrame({"hours": sorted(df["hours"])})
    y_line = model.predict(x_line)
    plt.plot(x_line["hours"], y_line)
    plt.xlabel("Hours studied")
    plt.ylabel("Score")
    plt.title("Student Score Predictor")
    plt.show()

if __name__ == "__main__":
    main()