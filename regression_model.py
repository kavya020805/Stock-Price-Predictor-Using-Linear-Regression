from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def train_and_evaluate_regression(df, features, target):
    """Train/test split, fit LinearRegression, and return test set results."""
    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n--- Linear Regression Model Performance ---")
    print(f"  R² Score:           {r2_score(y_test, y_pred):.4f}")
    print(f"  Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")

    return y_test, y_pred, model
