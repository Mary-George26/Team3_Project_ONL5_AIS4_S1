
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

### 9. Status Before EDA

The initial data quality checks and preliminary cleaning steps have been completed. Before finalizing the modeling dataset, further review is needed to confirm the dataset's coding conventions, assess duplicate records, and determine how missing and not-applicable values should be handled for each modeling task.

The dataset is now ready to proceed to structured Exploratory Data Analysis (EDA), while unresolved data quality questions remain documented for subsequent review.