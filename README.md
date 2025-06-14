# 🟡 Gold Price Prediction using Random Forest Regressor

This project predicts the future price of gold using historical financial data and the **Random Forest Regression** algorithm. It demonstrates the use of **machine learning** for **financial forecasting**, focusing on accuracy, interpretability, and model evaluation.

---

## 📌 Project Overview

* **Objective**: Predict gold prices using past trends and related financial indicators.
* **Model Used**: Random Forest Regressor
* **Tools & Libraries**: Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn
* **Dataset**: Historical gold prices along with indicators like USD index, oil prices, and stock indices.

---

## 📁 Files in this Repository

| File Name                     | Description                                                       |
| ----------------------------- | ----------------------------------------------------------------- |
| `gold_price_prediction.ipynb` | Main Jupyter Notebook containing EDA, model training, and results |
| `gold_data.csv`               | Dataset containing historical prices and indicators               |
| `requirements.txt`            | List of Python packages used in the project                       |
| `README.md`                   | Project documentation                                             |

---

## 🔍 Features & Workflow

1. **Data Preprocessing**

   * Handling missing values
   * Feature scaling (if needed)
   * Train-test split

2. **Exploratory Data Analysis (EDA)**

   * Correlation heatmaps
   * Time series trend plots
   * Distribution checks

3. **Model Building**

   * Training a Random Forest Regressor
   * Hyperparameter tuning (if implemented)
   * Feature importance extraction

4. **Model Evaluation**

   * MAE, MSE, RMSE
   * Visualization of actual vs predicted prices

---

## 🚀 How to Run the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/gold-price-prediction.git
   cd gold-price-prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:

   ```bash
   jupyter notebook gold_price_prediction.ipynb
   ```

---

## 📊 Sample Output

* Feature Importance Bar Chart
* Predicted vs Actual Price Line Plot
* Evaluation Metrics Displayed Clearly

---

## ✅ Requirements

* Python 3.x
* scikit-learn
* pandas
* numpy
* matplotlib
* seaborn
  *(Install all with `pip install -r requirements.txt`)*

---

## 💡 Future Improvements

* Incorporate deep learning models (e.g., LSTM)
* Use real-time API data for live predictions
* Integrate with Streamlit for web deployment


