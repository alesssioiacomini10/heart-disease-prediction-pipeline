# End-to-End Diagnostic AI Pipeline (Heart Disease)

## 1. Executive Summary
This project implements a robust Machine Learning pipeline to predict the presence of heart disease in patients. By analyzing 13 medical indicators  (including age (Age), sex (Gender), cp (Chest Pain Type), trestbps (Resting Blood Pressure), chol (Serum Cholesterol), fbs (Fasting Blood Sugar), restecg (Resting ECG Results), thalach (Max Heart Rate), exang (Exercise Induced Angina), oldpeak (ST Depression), slope (Slope of the Peak Exercise ST Segment), ca (Number of Major Vessels), thal (Thalassemia) ), the system acts as a high-precision "Second Opinion" tool for medical professionals.

The final system uses a **Logistic Regression** model which achieved **91.4% accuracy** in stress tests and passed a rigorous **Gender Bias Audit**, ensuring equitable performance across demographics.

---

## 2. Technical Architecture & Tech Stack
This project simulates a production-grade AI workflow:

* **Language:** Python 3.x
* **Data Processing:** `Pandas` (DataFrames), `Scikit-Learn` (Pipelines)
* **Imputation Strategy:** Median filling for numerical data; Mode filling for categorical data.
* **Feature Engineering:**
    * **Scaling:** `StandardScaler` to normalize physiological ranges.
    * **Encoding:** `OneHotEncoder` to transform categorical symptoms.
* **Model Selection:** Compared **Random Forest** (Ensemble) vs. **Logistic Regression** (Linear). Logistic Regression was selected for superior recall and interpretability.
* **Audit Tools:** Custom bias-detection scripts and Feature Importance visualization.

---

## 3. Key Findings (The "Why")
* **Zero Bias:** The model demonstrated equal or superior performance on female patients compared to male patients (-9.6% accuracy gap favoring women).
* **Primary Risk Factor:** *Asymptomatic Chest Pain (Type 4)* was identified as the strongest predictor of disease presence.
* **Primary Health Indicator:** *Zero Colored Vessels* (via Fluoroscopy) was the strongest predictor of a healthy heart.

---

## 4. How to Run This Project
1.  Clone the repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/heart-disease-pipeline.git](https://github.com/YOUR_USERNAME/heart-disease-pipeline.git)
    ```
2.  Install dependencies:
    ```bash
    pip install pandas scikit-learn seaborn matplotlib joblib
    ```
3.  Run the Notebook:
    ```bash
    jupyter notebook Heart_Disease_Project.ipynb
    ```

---

## 5. Project Explanations (For Interviews)


### Explanation 
"I developed an end-to-end binary classification pipeline using the UCI Cleveland dataset. I constructed a Scikit-Learn pipeline to handle data leakage and preprocessing, using `SimpleImputer` and `OneHotEncoder`.

I benchmarked Logistic Regression against Random Forest. While Random Forests are powerful, the Logistic Regression model yielded better recall and interpretability for this specific clinical dataset. I achieved ~91% accuracy on a gender-stratified stress test. Finally, I serialized the model using Joblib for deployment and visualized the coefficients to confirm that the model was prioritizing medically valid features like 'Asymptomatic Chest Pain' rather than noise."
