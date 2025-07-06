import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from regression_model import train_and_evaluate_regression


def load_data(file_path):
    """Load the stock dataset from a CSV file."""
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def clean_data(df):
    """Clean the data: drop duplicates, handle missing values."""
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def engineer_features(df):
    """Create rolling and return-based features for prediction."""
    df = df.copy()
    # Daily return
    df['Daily_Return'] = (df['Close'] - df['Open']) / df['Open']
    # Moving averages
    df['MA_5']  = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    # Volatility (rolling std dev)
    df['Volatility_5'] = df['Close'].rolling(window=5).std()
    # Volume change
    df['Volume_Change'] = df['Volume'].pct_change()
    # Shift target so features at day t predict Close at day t+1
    df['Target_Close'] = df['Close'].shift(-1)
    # Drop last row (NaN target) and any NaNs from rolling windows
    df.dropna(inplace=True)
    return df


def plot_correlation(df, features):
    corr = df[features + ['Target_Close']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation with Next-Day Close")
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             'r--', lw=2)
    plt.xlabel("Actual Next-Day Close")
    plt.ylabel("Predicted Next-Day Close")
    plt.title("Actual vs Predicted Next-Day Stock Prices")
    plt.tight_layout()
    plt.show()


def main():
    file_path = 'GS_stock_data.csv'  # your Goldman Sachs CSV

    # Load, clean, and engineer
    df = load_data(file_path)
    df = clean_data(df)
    df = engineer_features(df)

    # Define predictors and target
    features = ['Daily_Return', 'MA_5', 'MA_10', 'Volatility_5', 'Volume_Change']
    target   = 'Target_Close'

    # Correlation
    plot_correlation(df, features)

    # Train model & get test predictions
    y_test, y_pred, model = train_and_evaluate_regression(df, features, target)

    # Plot performance
    plot_actual_vs_predicted(y_test, y_pred)

    # Print coefficients
    print("\nFeature Coefficients:")
    for feat, coef in zip(features, model.coef_):
        print(f"  {feat}: {coef:.6f}")


if __name__ == "__main__":
    main()
