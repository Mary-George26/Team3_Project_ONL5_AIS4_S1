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

- **Identifier Column Removal:** The `id` column was removed because it is an identifier with no medical predictive value. Removing it prevents overfitting on index numbers.
- **Duplicate Records:** After removing the `id` column, **24 duplicate rows** were identified and completely **dropped/removed** from the dataset to ensure data purity and prevent model bias caused by duplicated instances.
- **Categorical Integrity:** All categorical variables (`gender`, `cholesterol`, `gluc`, `smoke`, `alco`, `active`) strictly conform to expected categorical ranges.
- **Age Standardization:** `age` was converted from days to years into `age_years` ($\text{age} / 365.25$) for improved interpretability. Age ranges between **29.6 and 64.9 years** (Mean: **53.3 years**).

## 3. Missing Values Handling & Imputation Strategy

Following initial inspection, missing values were identified across three specific numerical features:
- **`height`:** 27 missing values
- **`weight`:** 3 missing values
- **`bmi`:** 33 missing values

```text
[ Missing Height / Weight ] ───► Imputed via Gender-Based Median ───► Recalculated BMI ───► Zero Null Values
---

### 3. Data Anomalies & Cleaning Decisions Matrix

To prepare the dataset for machine learning without losing valuable patient records or introducing data leakage, strict domain-specific cleaning rules and statistical IQR boundaries were adopted:

| Feature | Identified Anomaly Description | Action Taken & Rationale |
| :--- | :--- | :--- |
| **`id` Column** | Identifier column with no medical predictive value. | **Dropped / Removed** to prevent overfitting on index numbers and unmask duplicate patient records. |
| **Height (`height`)** | 478 statistical outliers detected outside the IQR bounds ($142.5 \le \text{height} \le 186.5\text{ cm}$). | **Categorized & Retained**. Missing values imputed via gender-grouped median; valid original extreme measurements preserved. |
| **Weight (`weight`)** | 1,739 statistical outliers detected outside the IQR bounds ($39.5 \le \text{weight} \le 107.5\text{ kg}$). | **Categorized & Retained**. Missing values imputed via gender-grouped median; valid original extreme measurements preserved. |
| **Blood Pressure (`ap_hi` / `ap_lo`)** | Extreme noise, non-physiological values, and high diastolic readings ($> 120\text{ mmHg}$). | **Clipping & Preservation**. Copied to `ap_lo_original`; values of `ap_lo` above $120\text{ mmHg}$ capped at $120$. Extreme noise categorized into clinical risk groups. |

---

## 4. Outlier Detection, Categorization & Treatment (IQR Method)

The **Interquartile Range (IQR)** method was applied to identify potential statistical outliers across numerical features. Bounds were calculated using the standard formulas:

$$\text{IQR} = Q_3 - Q_1$$
$$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$

To preserve realistic medical measurements without unnecessary data loss, statistical outlier categorization was systematically separated from numerical modification:

| Feature | IQR Bounds | Outliers Detected | Action Taken & Rationale |
| :--- | :--- | :--- | :--- |
| **`height`** | 142.5 – 186.5 cm | 478 records | Categorized into `height_category`; original values retained. |
| **`weight`** | 39.5 – 107.5 kg | 1,739 records | Categorized into `weight_category`; original values retained. |
| **`ap_hi`** | 90 – 170 mmHg | 1,005 records | Categorized into `ap_hi_category`; potential outliers preserved without automatic clipping. |
| **`ap_lo`** | 65 – 105 mmHg | 3,499 records | Categorized into `ap_lo_category`. Original values copied to `ap_lo_original`; values above 120 capped at 120 (clipping). |
| **`age_years`** | 33.15 – 73.55 yrs | 4 records | Retained without modification as ages represent plausible human ranges. |
| **`bmi`** | ~14.52 – 39.48 | 1,992 records | Recalculated after height/weight imputation; potential outliers retained. |

## 5. Feature Engineering

To enrich exploratory data analysis and provide optimal inputs for predictive modeling, several domain-specific features were engineered:

* **`age_years`:** Converted age from days into years ($\text{age} / 365.25$) for intuitive clinical interpretability.
* **`bmi`:** Body Mass Index calculated precisely from the cleaned weight and height measurements.
* **Categorical Risk Feature Groupings:**
  * **`height_category` & `weight_category`:** Categorized records relative to their statistical Interquartile Range (IQR) bounds.
  * **`ap_hi_category` & `ap_lo_category`:** Binned systolic and diastolic blood pressure levels into structured clinical risk categories.

---


