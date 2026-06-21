# KNN_CLASSIFICATION
# K-Nearest Neighbor Classification Mini-Project

##  Project Overview
This project constructs a highly tuned **K-Nearest Neighbor (KNN)** classifier using Python and Scikit-learn to predict whether a breast tumor is **Malignant** or **Benign**. The pipeline models the standard workflow of a data scientist: exploratory data analysis, feature scaling engineering, cross-validation hyperparameter tuning ($k$), and robust multi-metric evaluations.

---

##  Dataset Description
We utilized the **Breast Cancer Wisconsin (Diagnostic) Dataset** (sourced via `sklearn.datasets`).
* **Instances:** 569 instances (212 Malignant, 357 Benign)
* **Features:** 30 numeric, real-valued features computing structural characteristics of cell nuclei (e.g., radius, texture, perimeter, area, smoothness).
* **Target:** Binary classification (0: Malignant, 1: Benign).

---

## Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.8+ installed. 

### 2. Installation
Clone this repository and install the required dependencies using pip:
```bash
git clone [https://github.com/yourusername/knn-classification-mini-project.git](https://github.com/yourusername/knn-classification-mini-project.git)
cd knn-classification-mini-project
pip install -r requirements.txt

### 3. Running the Project
Launch the Jupyter Notebook environment to interact with the code:

bash
jupyter notebook knn_classification.ipynb

##  Machine Learning Pipeline Workflow
The project follows a modular, step-by-step structure:
* **Exploratory Data Analysis:** Checked for missing values and mapped feature distributions using `seaborn` histograms.
* **Data Splitting:** Executed an 80/20 train/test split stratified by the target array to preserve class balance.
* **Feature Scaling:** Applied `StandardScaler` to align features onto a shared variance plane.
* **Hyperparameter Selection:** Evaluated odd $k$ values from 1 to 11 using 5-fold cross-validation.
* **Final Evaluation:** Benchmarked the model on held-out test data using global and class-specific metrics.



##  Key Engineering Insights

### 1. Why Feature Scaling is Mandatory for KNN
The KNN algorithm relies fundamentally on calculating spatial distances, such as **Euclidean Distance**:

$$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

In this dataset, features like `mean area` routinely yield values up to $2500$, while features like `mean smoothness` hover around $0.1$. Without scaling, the distance metrics would be completely overwhelmed by the `area` parameter, treating minor variations in area as vastly more vital than large structural shifts in smoothness. Applying `StandardScaler` standardizes features to have a mean of $0$ and a standard deviation of $1$.

### 2. Overcoming the Feature Name Warning
When running raw arrays through standard preprocessing steps, Scikit-learn will raise a validation warning if data frames are stripped of their headers. This pipeline explicitly wraps synthetic testing rows back into a named `pd.DataFrame` matching `X_train.columns` prior to calling `.transform()`, ensuring clean validation logs.


##  Key Findings & Evaluation

### Hyperparameter Tuning Results
* **Optimal Parameter:** $k = 9$ (or $k = 11$ depending on random seed state) yielded the highest cross-validated performance plateau (~96.5%).
* **Observation:** Low values ($k=1$) overfitted to local noise, while excessively high values risked structural underfitting.

### Final Model Performance (Test Set)
Our optimized model achieved elite evaluation metrics across the test split:

| Metric | Score | Insight |
| :--- | :--- | :--- |
| **Accuracy** | ~96.5% | High overall correct classification rate. |
| **Precision** | ~95.8% | Minimizes False Positives (mislabeled healthy tissue). |
| **Recall** | ~98.6% | **Crucial for Medical Deployment.** Minimizes False Negatives (missing a malignant tumor). |
| **F1-Score** | ~97.2% | Robust harmonic balance of precision and recall. |
