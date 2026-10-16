### 📁 Dataset Source
The dataset used in this project is publicly available on Kaggle — [COVID-19 Patient Dataset](https://www.kaggle.com/datasets/meirnizri/covid19-dataset) (Sourced from the Mexican Ministry of Health).

---
### 📋 Table of Contents
1. [Dataset Overview](#1-dataset-overview)
2. [Data Types and Missing Values](#2-data-types-and-missing-values)
3. [Special Categorical Codes](#3-special-categorical-codes)
4. [Categorical Value Validation](#4-categorical-value-validation)
5. [Duplicate Records](#5-duplicate-records)
6. [Numerical Analysis and Potential Outliers](#6-numerical-analysis-and-potential-outliers)
7. [Logical Consistency Checks](#7-logical-consistency-checks)
8. [Cleaning Decisions Summary](#8-cleaning-decisions-summary)
9. [Post-Cleaning Exploratory Data Analysis (EDA) & Key Insights](#9-post-cleaning-exploratory-data-analysis-eda--key-insights)
10. [Technologies & Tools Used](#11-technologies--tools-used)


## Data Quality Assessment and Cleaning

### 1. Dataset Overview

The COVID-19 dataset contains **1,048,575 records and 21 original columns**. It includes demographic information, patient admission status, pre-existing medical conditions, COVID-19 classification, and death-date information.

The dataset was inspected to identify missing values, special category codes, potential outliers, duplicate records, data type issues, and logical inconsistencies before Exploratory Data Analysis (EDA).

### 2. Data Types and Missing Values

- The `DATE_DIED` column was converted from a string to a datetime data type.
- The original value `9999-99-99` was interpreted as an indicator that no death date was recorded and converted to a missing datetime value.
- A `NO_DEATH_DATE` indicator was created to preserve this distinction.
- **971,633 records** have no recorded death date, while **76,942 records** have a valid death date.
- No parsing failures or inconsistencies between `DATE_DIED` and `NO_DEATH_DATE` were detected.

### 3. Special Categorical Codes

Special codes were identified in several categorical variables:

- `97`: interpreted as Not Applicable in `INTUBED`, `ICU`, and `PREGNANT`.
- `98` and `99`: treated as unknown or unavailable responses in the relevant variables, based on the dataset's coding conventions.

The special codes were replaced with missing values in the main columns, while indicator columns and backup copies were retained to preserve information about the original codes.

After this transformation, the reviewed categorical columns contained only the expected binary values (`1` and `2`) or missing values.

### 4. Categorical Value Validation

The following variables were checked for unexpected values:

- `SEX`
- `PATIENT_TYPE`
- `USMER`
- `MEDICAL_UNIT`
- `CLASIFFICATION_FINAL`
- The reviewed medical-condition and treatment-related variables

No unexpected category codes were found in the reviewed variables.

### 5. Duplicate Records

An initial check identified **812,049 exact duplicate rows (77.44%)**.

However, the dataset does not contain a unique patient identifier, so identical rows cannot automatically be assumed to represent erroneous duplicate patients. Different patients may share the same recorded characteristics.

Therefore, duplicate records were retained pending further investigation and verification of the dataset's structure and provenance.

### 6. Numerical Analysis and Potential Outliers

The `AGE` variable was examined using descriptive statistics and the Interquartile Range (IQR) method.

| Statistic | Value |
|---|---:|
| Mean | 41.79 |
| Median | 40 |
| Q1 | 30 |
| Q3 | 53 |
| IQR | 23 |
| Lower bound | -4.5 |
| Upper bound | 87.5 |
| Potential outliers | 5,640 |

The IQR method flagged **5,640 records with ages above 87.5 years**. A further logical check identified **138 records with ages above 100 years**.

These records were retained because statistical outliers are not necessarily invalid observations. Their validity should be assessed using domain knowledge and, where possible, the original data source.

No negative ages were detected.

### 7. Logical Consistency Checks

Cross-variable checks were performed on:

- `PATIENT_TYPE` versus `INTUBED`
- `PATIENT_TYPE` versus `ICU`
- `SEX` versus `PREGNANT`
- `DATE_DIED` versus `NO_DEATH_DATE`

The observed patterns were consistent with the dataset's special-code conventions. No obvious inconsistencies were detected in these checks.

The relationship between `DATE_DIED` and `CLASIFFICATION_FINAL` was also explored. The association was retained as an observation for further investigation rather than treated as proof of a data error.

### 8. Cleaning Decisions

- Special category codes were handled explicitly rather than treated as valid Yes/No responses.
- Original category codes and indicator variables were preserved for traceability.
- Potential age outliers were not removed automatically.
- Exact duplicate rows were retained because the absence of a unique patient identifier prevents reliable confirmation that they are erroneous duplicates.
- No rows were removed during the checks described above.

---

### 9. Post-Cleaning Exploratory Data Analysis (EDA) & Key Insights

Following data cleaning and engineering, exploratory visualizations and statistical analyses were conducted to uncover underlying demographic and clinical patterns:

1. **Patient Age Distribution:**
   - **Finding:** The age distribution shows a prominent peak among adults aged **25 to 50 years**, representing the most frequently recorded demographic group.

2. **COVID-19 Classifications Distribution:**
   - **Finding:** The distribution of COVID-19 classifications highlights that **codes 3 and 7** contain the highest volume of patient records.

3. **Patient Type Distribution:**
   - **Finding:** Patient types show that the vast majority of cases (code 1) were managed as outpatients, while a smaller portion required hospital admission (code 2).

4. **Death Records Comparison:**
   - **Finding:** Comparison of death records indicates that the majority of patients survived (`False`), with a significantly lower count of recorded deaths (`True`).

5. **Patient Age by COVID-19 Classification:**
   - **Finding:** Box plot analysis of patient age across different classification codes reveals that **classification code 2** generally has an older patient median age.

6. **Patient Age by Death Status:**
   - **Finding:** Age comparison by death status clearly demonstrates that deceased patients (`True`) have a significantly higher median age compared to survivors (`False`).

7. **COVID-19 Classifications by Patient Type:**
   - **Finding:** Stacked bar chart analysis illustrating COVID-19 classifications broken down by patient type, highlighting compositional shifts between outpatients and inpatients.

8. **Death Rates by Chronic Conditions:**
   - **Finding:** Death rate percentages categorized by chronic conditions show that patients with severe conditions, such as **renal disease, diabetes, and COPD**, face substantially higher mortality rates.

9. **Chronic Conditions Prevalence:**
   - **Finding:** The prevalence analysis of pre-existing health issues indicates that **Hypertension and Obesity** are the most frequently recorded chronic conditions in the dataset, underscoring their widespread presence among patients.
  
### 10. Technologies & Tools Used

* **Programming Language:** Python 3.x
* **Core Libraries & Data Processing:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **File & Path Management:** Pathlib, CSV
* **Environment & Interactive Display:** IPython.display (display), Jupyter Notebook / Visual Studio Code
