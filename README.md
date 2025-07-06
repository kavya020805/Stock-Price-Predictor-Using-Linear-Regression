# 📈 Stock Price Predictor Using Linear Regression

A simple and interpretable stock price forecasting model for Goldman Sachs stock using linear regression. This project applies feature engineering and predictive modeling on historical stock data to estimate the next-day closing price.

---

## 🧪 Project Workflow

1. Load and clean historical stock data  
2. Engineer meaningful features (returns, moving averages, volatility)  
3. Visualize correlations  
4. Train a linear regression model  
5. Evaluate prediction accuracy using R² and MSE  
6. Visualize predicted vs actual prices

---

## 📊 Visualizations

### 🔥 Feature Correlation Heatmap

This heatmap visualizes the correlation between each feature and the target (next-day close). Strong correlations help identify which indicators are most predictive of future prices.

> **Interpretation:** Features with higher absolute correlation values may contribute more significantly to the regression model.

📸 *Example Output:*  
![Figure_1](https://github.com/user-attachments/assets/50f2a521-86c4-474f-aa5e-d8096c4c985b)


---

### 📈 Actual vs Predicted Prices

This scatter plot compares the predicted stock prices (from the model) to the actual next-day prices. A perfect model would align all points on the red dashed line.

> **Interpretation:** The closer the points are to the red line, the better the model performance.

📸 *Example Output:*  
![Figure_2](https://github.com/user-attachments/assets/2aae29d1-a299-40b4-bed0-5962f30f9ea3)


---

## 🚀 How to Run

1. Place your Goldman Sachs stock CSV as `GS_stock_data.csv` in the project folder  
2. Run the script:
   ```bash
   python stock_price_predictor.py
