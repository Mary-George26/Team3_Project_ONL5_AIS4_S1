# Chronic Kidney Disease (CKD) Dataset — Data Cleaning & Preparation

## 1. Project Overview

This project focuses on exploring, cleaning, validating, and preparing a Chronic Kidney Disease (CKD) dataset for future data analysis and machine learning applications.

The main objective is to improve data quality while preserving potentially meaningful medical information. The project follows a structured data preparation workflow, including missing-value handling, duplicate detection, categorical encoding, outlier investigation, feature selection, feature engineering, and final validation.

An important principle throughout this project is that **statistical outliers are not automatically data errors**. Values are investigated before any decision is made to remove, replace, or transform them.

## 2. Dataset Information

- **Dataset name:** Chronic Kidney Disease Dataset
- **Source:** [Kaggle — Chronic Kidney Failure Dataset Reboot](https://www.kaggle.com/datasets/marcklugerhart/chronic-kidney-failure-dataset-reboot)
- **Original shape:** 15,051 rows × 21 columns
- **Target variable:** `Insuficiencia renal crónica`
- **Problem type:** Binary classification
- **Target labels:** `no` and `si`
- **Working dataset after initial cleaning:** 15,049 rows × 26 columns, before temporary exploratory columns or optional engineered features

The original dataset contains patient identifiers, demographic information, laboratory measurements, urine test results, and a binary CKD outcome.

> **Data integrity:** The original dataset is preserved in `df_raw`. All cleaning and transformations are performed on a separate working copy, `df_work`.

## 3. Business and Analytical Objective

The objective is to produce a reliable, well-documented dataset that can support exploratory analysis and subsequent machine learning experiments related to chronic kidney disease.

The preparation process aims to:

- Identify missing, duplicated, inconsistent, and potentially invalid data.
- Standardize categorical values and data types.
- Preserve clinically meaningful observations whenever their validity cannot be disproved.
- Investigate suspicious distributions and potential source-data quality issues.
- Identify redundant or uninformative features.
- Evaluate potential engineered features.
- Validate the resulting dataset before saving it for downstream analysis.

This dataset is intended for educational and analytical purposes. It does not establish a clinical diagnosis or replace professional medical assessment.

## 4. Dataset Features

The original dataset contains the following columns:

| Original column | Description |
|---|---|
| `Identificación del paciente` | Patient identifier |
| `Edad` | Age |
| `Presión arterial` | Blood pressure |
| `Gravedad específica de la orina` | Urine specific gravity |
| `Albúmina en orina` | Urine albumin |
| `Azúcar en orina` | Urine sugar |
| `Glóbulos rojos` | Red blood cell test result |
| `Células de pus` | Pus cell test result |
| `Grupos de células de pus` | Pus cell clumps |
| `Bacterias` | Bacteria |
| `Glucosa en sangre aleatoria` | Random blood glucose |
| `Urea en sangre` | Blood urea |
| `Creatinina sérica` | Serum creatinine |
| `Sodio en sangre` | Blood sodium |
| `Potasio en sangre` | Blood potassium |
| `Hemoglobina` | Hemoglobin |
| `Volumen corpuscular empaquetado` | Packed cell volume |
| `Recuento de glóbulos blancos` | White blood cell count |
| `Recuento de glóbulos rojos` | Red blood cell count |
| `Apetito` | Appetite |
| `Insuficiencia renal crónica` | CKD target label |

The exact interpretation and units of unusual numeric columns should be verified against reliable dataset documentation before applying clinical thresholds or transformations.

## 5. Data Cleaning Workflow

### 5.1. Loading and Preserving the Original Dataset

The dataset was loaded into pandas, and a separate working copy was created.

```python
df_raw = pd.read_csv("kidney_disease.csv")
df_work = df_raw.copy(deep=True)
```

The raw dataset remains unchanged, allowing the cleaned values to be compared with the original source values.

### 5.2. Initial Inspection

The initial inspection covered:

- Dataset dimensions.
- Column names and data types.
- Non-null counts.
- Descriptive statistics.
- Unique values in categorical columns.
- Target distribution.
- Potential inconsistencies in numerical and categorical features.

The original dataset contained 15,051 records and 21 columns.

### 5.3. Missing Values Analysis

The original dataset contained missing values in the following columns:

| Column | Missing values |
|---|---:|
| `Grupos de células de pus` | 9 |
| `Bacterias` | 6 |
| `Creatinina sérica` | 9 |
| `Apetito` | 1 |
| `Insuficiencia renal crónica` | 2 |

All other original columns were initially reported as having no missing values.

#### Treatment decisions

- **Missing target values:** Two records with missing CKD labels were removed because the target could not be used reliably for supervised learning.
- **Missing categorical values:** Missing values in `Grupos de células de pus`, `Bacterias`, and `Apetito` were filled with the category `desconocido` (unknown).
- **Missing serum creatinine:** Missing values were filled using the median, which was 1.4 in the working dataset.

The categorical missing-value strategy preserves the distinction between an unknown observation and a known positive or negative result.

Median imputation is less sensitive to extreme values than mean imputation, although it does not eliminate uncertainty introduced by missing measurements.

### 5.4. Duplicate and Identifier Validation

The patient identifier column, `Identificación del paciente`, was reviewed and then removed from the working dataset.

The ID was excluded because it serves as a record identifier rather than a direct clinical measurement. Removing it also avoids accidentally treating an identifier as a meaningful numerical predictor.

After removing the identifier, one exact duplicate row was detected in the feature space. The two records had different original identifiers, reported as 1531 and 15040.

The duplicate was retained because identical feature values do not prove that two records represent the same patient or that one record is erroneous.

**Decision:** Retain the observation and document it for further source verification rather than deleting it automatically.

### 5.5. Data Types and Categorical Standardization

Categorical values were inspected and standardized before encoding.

The following binary categories were encoded:

| Column | Original values | Encoded values |
|---|---|---|
| `Glóbulos rojos` | `normal`, `anormal` | 0, 1 |
| `Células de pus` | `normal`, `anormal` | 0, 1 |
| `Insuficiencia renal crónica` | `no`, `si` | 0, 1 |

One-hot encoding was applied to:

- `Grupos de células de pus`
- `Bacterias`
- `Apetito`

The unknown category was retained during encoding where applicable, producing indicator columns such as `Bacterias_desconocido` and `Apetito_desconocido`.

One-hot encoding represents categorical values numerically without imposing an artificial ordering between categories.

The target encoding is:

- `0` = `no`
- `1` = `si`

The target is kept separate conceptually from the predictor features and must not be used to construct new predictor variables.

### 5.6. Categorical and Logical Validation

Categorical values were checked for unexpected labels, and encoded categories were reviewed to ensure that the transformations were consistent with the original values.

The numerical columns were also inspected for unusual values and ranges.

A key distinction was maintained throughout the process:

- A value outside an expected statistical range is not automatically invalid.
- A value that appears clinically unusual should not be modified without sufficient evidence.
- Data-quality concerns should be documented even when the correct transformation is unknown.

## 6. Outlier Detection and Treatment

The Interquartile Range (IQR) method was used strictly as a **detection technique, not as an automatic deletion rule**. Statistical boundaries were calculated as:

$$\text{IQR} = Q_3 - Q_1 \quad \implies \quad [\text{Lower Bound}, \text{Upper Bound}] = [Q_1 - 1.5 \times \text{IQR}, \, Q_3 + 1.5 \times \text{IQR}]$$

Observations outside these boundaries were investigated prior to making any removal, clipping, or transformation decisions.

### 6.1. Summary of IQR Findings and Treatment Decisions

| Feature | Statistical Bounds | Flagged Count | Flagged Range | Decision & Clinical Context |
|---|---|---:|---|---|
| `Edad` (Age) | — | 850 | 2 – 14 | **Retain.** Younger ages are clinically valid observations. |
| `Presión arterial` (Blood Pressure) | [39.5, 147.5] | 200 | 148 – 180 | **Retain.** High blood pressure carries meaningful clinical signal. |
| `Azúcar en orina` (Urine Sugar) | Upper: 500 | 1,808 | 501 – 1,000 | **Retain temporarily.** Values >300 map exclusively to the negative class in the raw source data; requires further source investigation. |
| `Urea en sangre` (Blood Urea) | Upper: 196.25 | 4 | 198 – 200 | **Retain.** No definitive evidence of data-entry error. |
| `Creatinina sérica` (Serum Creatinine) | Upper: ~6.715 | 1,536 | Up to 76 | **Retain temporarily.** Extreme values were present in raw data; avoid arbitrary clipping without official documentation. |
| `Sodio en sangre` (Blood Sodium) | Lower: 123 | 52 | 120 – 122 | **Retain.** 48 of 52 cases belong to the positive target class ($CKD = 1$), indicating clinical relevance. |

> **Note:** Other numerical features—including random blood glucose, potassium, hemoglobin, packed cell volume (PCV), white blood cell count, and red blood cell count—yielded zero IQR-flagged outliers.

## 7. Data Quality Issues Requiring Further Investigation

The analysis identified unusual distributions in several laboratory-related columns.

### 7.1. Urine Specific Gravity

The feature `Gravedad específica de la orina` contained values ranging from 10 to 1,025.

Its quartiles included values around 1,015 and 1,020, while other observations included values such as 10.18, 11.31, and 14.72.

These values raise questions about the representation, scale, or meaning of the feature.

The same pattern was present in the raw dataset. No rescaling or conversion was applied because the correct interpretation could not be established confidently from the available information.

### 7.2. Urine Albumin

The feature `Albúmina en orina` contained values ranging from 0 to 989.9, with a median of 1 and a third quartile of 452.2.

This distribution is unusual enough to require verification against the dataset documentation.

The values were retained without arbitrary transformations because the correct scale and interpretation have not yet been confirmed.

### 7.3. Relationship Between Urine Features

The analysis also found strong alignment between certain urine-related values, including observations with specific-gravity values below 1,000 and higher urine-sugar values.

These relationships may reflect the structure or encoding of the source dataset. Their underlying cause remains unresolved.

### 7.4. Important Limitation

The unusual distributions were found in the raw source data, so they were not caused by the cleaning operations.

These issues should remain explicitly documented. The current cleaning process does not establish that every laboratory measurement is clinically valid.

Before using the dataset for clinical conclusions or making strong claims about model reliability, the suspicious features should be investigated further.

## 8. Preliminary Feature Selection

The initial feature-selection review checked for constant columns, exactly duplicated columns, and the number of unique values in each feature.

Results:

- No constant columns were identified.
- No exactly duplicated columns were identified.
- The patient identifier was removed.
- Binary and one-hot encoded columns were retained because they represent meaningful categories.

A feature having only two unique values is not necessarily uninformative. Binary indicators and encoded categorical features naturally have two possible values.

Potentially related features, such as `Células de pus` and `Grupos de células de pus`, were not removed merely because their names are similar. They may represent different measurements.

No additional feature was removed solely on the basis of its name or number of unique values.

Further correlation analysis can help identify highly related numerical features, but high correlation alone does not prove that a feature should be removed.

## 9. Feature Engineering

### 9.1. Age Group Feature

A new feature, `Age_Group`, was created from the original `Edad` column using the following age intervals:

- `Under_18`: Ages 0–17
- `18_39`: Ages 18–39
- `40_59`: Ages 40–59
- `60_plus`: Ages 60 and above

The original `Edad` feature was retained to preserve the continuous numerical age information.

These age groups are exploratory categories rather than validated clinical thresholds. The usefulness of `Age_Group` should be evaluated in subsequent analysis rather than assumed.

The temporary exploratory feature `Sugar_Range` was removed because it was not selected as part of the final feature set.

### 9.2. Candidate Feature: Urine Test Abnormality Count

A potential feature could count abnormal results across selected urine-related tests.

This feature should only be created after confirming the meaning and encoding of each source column, including how unknown results are handled.

It has not been treated as a validated feature in the current cleaning workflow.

### 9.3. Features Not Automatically Created

A composite kidney-marker score was considered but not implemented because the scales and distributions of some laboratory features require further investigation.

No feature should be created using the target variable, `Insuficiencia renal crónica`, because this would introduce target leakage.

**Feature-engineering status:** Candidate features have been identified, but only features that are actually implemented, validated, and documented should be included in the final dataset.

## 10. Final Working Dataset Status

After the main cleaning, encoding, and feature-engineering operations, the working dataset contains **15,049 rows and 27 columns**.

The final working dataset includes the original clinical features, encoded categorical variables, the binary target, and the engineered `Age_Group` feature. The patient identifier and temporary `Sugar_Range` column are excluded.

The target distribution is:

| Target value | Meaning | Records |
|---|---|---:|
| 0 | No CKD (`no`) | 7,696 |
| 1 | CKD (`si`) | 7,353 |
| **Total** | | **15,049** |

No missing values were detected during final validation.

The dataset is ready to be saved as a separate cleaned file for subsequent analysis, subject to the documented limitations concerning unusual source-data distributions.

## 11. Final Data Validation

The cleaned working dataset was checked after the main cleaning and feature-engineering operations.

| Validation check | Final result |
|---|---|
| Number of rows | 15,049 |
| Number of columns | 27 |
| Total missing values | 0 |
| Target value `0` (`no`) | 7,696 |
| Target value `1` (`si`) | 7,353 |
| Patient identifier | Removed |
| Temporary `Sugar_Range` feature | Removed |
| `Age_Group` feature | Present |
| Original dataset dimensions | Unchanged at 15,051 × 21 |

The final checks confirmed that the working dataset contains no missing values, the target has the expected two encoded classes, the identifier and temporary exploratory column are absent, and the engineered `Age_Group` feature is present.

The original dataset's dimensions remained unchanged. This check confirms that the original DataFrame was not resized, but dimensions alone do not prove that every original value remained unchanged.

The suspicious laboratory-value distributions and the retained feature-space duplicate remain documented data-quality considerations.

## 12. Tools and Technologies

- Python
- pandas
- NumPy
- Jupyter Notebook
- Statistical descriptive analysis
- IQR-based outlier detection
- Categorical encoding
- Data validation

## 13. Limitations

- Some laboratory columns contain unusual distributions whose underlying causes have not been established.
- The source documentation has not yet confirmed the correct scale or interpretation of all suspicious measurements.
- IQR boundaries identify statistical extremes, not necessarily incorrect medical observations.
- Median imputation replaces missing measurements with a statistical estimate and may affect downstream analyses.
- The retained duplicate feature row requires further source verification.
- Engineered features require evaluation before being considered beneficial.
- No model performance or clinical validity claims are made as part of this data-cleaning stage.

## 14. Conclusion

The main data-cleaning and preparation workflow has been completed, including missing-value treatment, identifier removal, duplicate investigation, categorical encoding, descriptive analysis, IQR-based outlier investigation, preliminary feature-selection checks, and the creation of the `Age_Group` feature.

Final validation confirmed a working dataset of 15,049 rows and 27 columns, with no detected missing values and the expected binary target distribution.

The original dataset dimensions remained unchanged, and the temporary `Sugar_Range` column was removed from the working dataset.

Some unusual laboratory-value distributions remain unresolved and are documented as source-data quality concerns. The dataset should therefore be used with these limitations in mind.

The next stage can focus on further exploratory analysis or preparing the data for machine learning, including appropriate feature evaluation, leakage prevention, and model validation.
