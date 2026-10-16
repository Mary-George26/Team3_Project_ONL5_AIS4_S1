# 🩺 Cardiovascular Disease Prediction & Analysis

A comprehensive data science project focused on understanding, cleaning, exploring, and predicting cardiovascular disease status based on demographic, biological, and lifestyle indicators.

---

## 📋 Table of Contents
1. [Executive Overview & Dataset Summary](#1-executive-overview--dataset-summary)
2. [Initial Data Inspection & Quality Assessment](#2-initial-data-inspection--quality-assessment)
3. [Data Anomalies & Cleaning Decisions Matrix](#3-data-anomalies--cleaning-decisions-matrix)
4. [Pipeline-Safe Imputation Strategy](#4-pipeline-safe-imputation-strategy)
5. [Feature Engineering: Body Mass Index (BMI)](#5-feature-engineering-body-mass-index-bmi)
6. [Post-Cleaning Exploratory Data Analysis (EDA) & Key Insights](#6-post-cleaning-exploratory-data-analysis-eda--key-insights)

---

## 1. Executive Overview & Dataset Summary
* **Dataset Source:** Publicly available on Kaggle — [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
* **Dataset Size:** 70,000 records and 13 original features describing patient demographic characteristics, biological measurements, lifestyle factors, and cardiovascular disease status[cite: 1].
* **Target Variable:** `cardio` (a binary classification indicator: `0 = Healthy / No Disease`, `1 = Heart Disease Present`)[cite: 1].
* **Class Balance:** The target variable is **perfectly balanced** (~50.03% healthy vs. ~49.97% diseased), eliminating the need for complex resamplers (e.g., SMOTE) and allowing models to learn both classes fairly[cite: 8].

---

## 2. Initial Data Inspection & Quality Assessment
During the preliminary assessment, dataset structural integrity, column data types, missing values, and potential anomalies were evaluated:

* **Missing Values:** `0` null values across all original columns.
* **Duplicate Records:** `0` duplicate rows detected.
* **ID Uniqueness:** All `id` values are 100% unique[cite: 1].
* **Categorical Integrity:** All categorical variables (`gender`, `cholesterol`, `gluc`, `smoke`, `alco`, `active`) strictly conform to expected categorical ranges[cite: 1, 2].
* **Feature Engineering:** `age` was converted from days to years into `age_years` ($\text{age} / 365.25$) for improved interpretability[cite: 1]. Age ranges between **29.6 and 64.9 years** (Mean: **53.3 years**).

---

### 3. Data Anomalies & Cleaning Decisions Matrix

To prepare the dataset for machine learning without losing valuable patient records or introducing data leakage, strict domain-specific cleaning rules were adopted:

| Feature | Identified Anomaly Description | Action Taken & Rationale |
| :--- | :--- | :--- |
| **`id` Column** | Identifier column with no predictive medical power. | **Dropped / Reset** to prevent overfitting on index numbers. |
| **Height (`height`)** | 30 records below 100 cm or above 220 cm. | **Replaced with `NaN`**. Preserves rest of patient row; imputed via Median in pipeline. |
| **Weight (`weight`)** | 7 records below 30 kg. | **Replaced with `NaN`**. Preserves rest of patient row; imputed via Median in pipeline. |
| **Blood Pressure (`ap_hi` / `ap_lo`)** | Extreme noise (e.g., values > 10,000, negatives, or `ap_hi <= ap_lo`). | **Filtered & Dropped Rows**. Kept valid human range:<br>• `70 <= ap_hi <= 250`<br>• `40 <= ap_lo <= 150`<br>• `ap_hi > ap_lo` |

---

## 4. Pipeline-Safe Imputation Strategy

```text
[ Extreme Height/Weight Outliers ] ───► Converted to NaN ───► Preserved Patient Rows ───► Imputed via Median in Scikit-Learn Pipeline

## Engineering Note: Data Preprocessing & Feature Engineering

Instead of hardcoding replacement values or dropping entire rows, anomalous height and weight values were set to `NaN`. Median imputation will be executed within a `SimpleImputer` inside a Machine Learning Pipeline. The median will be computed solely on training splits (`X_train`) during model training to completely avoid Data Leakage.

### 5. Feature Engineering: Body Mass Index (BMI)

To enrich model predictions, the Body Mass Index (BMI) was engineered from clean height and weight features:

$$\text{BMI} = \frac{\text{weight (kg)}}{\left(\frac{\text{height (cm)}}{100}\right)^2}$$

* **Clinical Significance:** Provides a holistic measure of body composition, which strongly correlates with hypertension and cardiovascular risk.

---

### 6. Post-Cleaning Exploratory Data Analysis (EDA) & Key Insights

Following data cleaning and engineering, exploratory visualizations were conducted to uncover underlying patterns:

1. **Numerical Feature Distributions (Histograms):**
   * **Finding:** Height, weight, and age_years demonstrate clean, near-normal distributions bounded within realistic human limits post-cleaning.
   * **Conclusion:** Blood pressure features (`ap_hi`, `ap_lo`) now reside entirely within medically plausible clinical bounds.

2. **Target Balance Verification (Bar Plot):**
   * **Finding:** Both classes (0: Healthy, 1: Disease) each maintain approximately 35,000 clean records.
   * **Conclusion:** Eliminates classification bias toward a dominant class during model execution.

3. **Demographic & Physiological Comparisons (Bar & Box Plots):**
   * **Finding:** Patients with cardiovascular disease (`cardio = 1`) display noticeably higher mean Age, Systolic BP (`ap_hi`), Diastolic BP (`ap_lo`), and BMI compared to non-diseased individuals (`cardio = 0`). Average height shows minimal variation between groups.
   * **Conclusion:** Age, blood pressure, and BMI are primary risk indicators for heart disease.

4. **Categorical Risk Factors:**
   * **Finding:** The probability of heart disease increases dramatically at higher levels of Cholesterol and Glucose (particularly at Level 3: Well above normal).
   * **Conclusion:** High cholesterol and glucose are critical categorical predictors.

5. **Feature Importance & Correlation Analysis:**
   * **Finding:** Systolic Blood Pressure (`ap_hi`) exhibits the strongest positive correlation with heart disease ($\sim 0.43$), followed by Diastolic BP BP (`ap_lo`) ($\sim 0.34$), Age ($\sim 0.24$), Cholesterol ($\sim 0.22$), and BMI ($\sim 0.19$). Height shows almost zero correlation ($\sim -0.01$).
   * **Conclusion:** Blood pressure metrics, age, cholesterol, and body mass are the most impactful features for predictive model training.


