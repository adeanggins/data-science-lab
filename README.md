# Data Science Lab: Applied Analytics & Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)](https://github.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

## 📋 Overview

Welcome to my **Data Science Lab**. This repository serves as a centralized portfolio of my technical exercises, experiments, and proof-of-concept models derived from public datasets. 

The objective of this collection is to demonstrate rigorous data analysis workflows, ranging from raw data ingestion and advanced preprocessing to statistical modeling and machine learning implementation. Each project within this repository is designed to address specific analytical challenges using the Python ecosystem.

## 🛠️ Technical Arsenal & Libraries

This repository encompasses my expertise across the full data science lifecycle, utilizing the following libraries and frameworks:

### **Core Computing & Statistics**
* **NumPy:** High-performance multidimensional array manipulation and linear algebra.
* **SciPy:** Advanced scientific computing (optimization, integration, interpolation, eigenvalue problems).
* **Statsmodels:** Statistical data exploration, estimation of statistical models, and performing statistical tests.

### **Data Manipulation & ETL**
* **Pandas:** Efficient data structures for data analysis and manipulation (Timeseries, DataFrames).
* **SQLAlchemy / PyODBC:** Database connectivity and ORM operations.

### **Machine Learning & AI**
* **Scikit-Learn:** Supervised and unsupervised learning algorithms (Regression, Classification, Clustering, Dimensionality Reduction).
* **XGBoost / LightGBM:** Gradient boosting frameworks for high-efficiency modeling.
* **TensorFlow / Keras:** Deep learning architectures for complex pattern recognition.

### **Visualization & Reporting**
* **Matplotlib & Seaborn:** Static, publication-quality statistical graphics.
* **Plotly:** Interactive dashboards and visualization.

---

## 📂 Featured Projects

| Project Name | Domain | Key Techniques | Libraries Used |
| :--- | :--- | :--- | :--- |
| **Predictive Maintenance Engine** | Industrial | Time-series forecasting, Anomaly detection | `pandas`, `scikit-learn`, `prophet` |
| **Customer Segmentation** | Retail | K-Means Clustering, PCA, Cohort Analysis | `seaborn`, `scipy`, `sklearn` |
| **Loan Default Classifier** | Finance | Binary Classification, SMOTE (imbalance handling) | `xgboost`, `imblearn` |
| **Housing Price Optimization** | Real Estate | Multivariate Regression, GridSearch CV | `statsmodels`, `sklearn` |

*(Note: The projects above are placeholders. Update the table with your specific exercises.)*

---

## 🔬 Methodological Approach

For every analysis in this repository, I adhere to a structured data science methodology:

1.  **Problem Definition:** Clearly defining the target variable $y$ and the feature vector $X$.
2.  **Exploratory Data Analysis (EDA):** utilizing statistical moments (mean $\mu$, variance $\sigma^2$) and distribution analysis to understand data topology.
3.  **Feature Engineering:** Transforming raw data into informative features using techniques such as:
    * Normalization: $$x' = \frac{x - \min(x)}{\max(x) - \min(x)}$$
    * Standardization: $$z = \frac{x - \mu}{\sigma}$$
4.  **Model Selection & Tuning:** Leveraging cross-validation and hyperparameter optimization to minimize loss functions (e.g., $MSE$, $LogLoss$).
5.  **Evaluation:** Assessing performance via precision, recall, F1-score, and ROC-AUC curves.

## 🚀 Getting Started

To replicate the analyses found in this repository, follow these steps to set up the environment.

### Prerequisites
* Python 3.8+
* Git

### Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/adeanggins/data-science-lab.git](https://github.com/adeanggins/data-science-lab.git)
    cd data-science-lab
    ```

2.  **Create a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 📬 Contact

If you have questions regarding the methodology or code implementation, feel free to reach out.

* **LinkedIn:** [Ade Anggi Naluriawan Santoso](https://www.linkedin.com/in/ade-anggi-naluriawan-santoso-83493a81/)
* **Email:** [adeanggins@gmail.com]

---
*Disclaimer: All datasets used in this repository are public and open-source. References to specific sources are provided within each project folder.*
