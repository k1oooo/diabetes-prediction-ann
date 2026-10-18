# Diabetes Prediction with an Artificial Neural Network

A healthcare machine learning project for **SAIA 2113 (Machine Learning)**. It predicts whether a patient has diabetes from demographic, medical-history and clinical measurements, using a feed-forward neural network (Keras/TensorFlow) trained on an imbalanced dataset of 100,000 records.

## Highlights

- Full pipeline: cleaning, EDA, preprocessing, stratified train/validation/test split, model training and evaluation
- ANN with batch normalisation, dropout, class weighting and early stopping
- Test **ROC-AUC of 0.974** and **recall of 0.998** on the diabetic class at a recall-oriented decision threshold

## Dataset

`diabetes_prediction_dataset.csv`: 100,000 rows, 9 columns, no missing values.

| Feature | Type | Description |
|---|---|---|
| `gender` | Categorical | Female / Male / Other |
| `age` | Numerical | Age in years |
| `hypertension` | Binary | 1 = has hypertension |
| `heart_disease` | Binary | 1 = has heart disease |
| `smoking_history` | Categorical | never, No Info, former, current, not current, ever |
| `bmi` | Numerical | Body mass index |
| `HbA1c_level` | Numerical | Average blood sugar level over the past 2-3 months |
| `blood_glucose_level` | Numerical | Current blood glucose level |
| `diabetes` | Binary (target) | 1 = diabetic |

The classes are heavily imbalanced: roughly **91.5% non-diabetic vs 8.5% diabetic**.

## Pipeline

### 1. Preprocessing and EDA (`diabetes_preprocessing_eda.ipynb`)

| Step | Action |
|---|---|
| Duplicates | Removed 3,854 duplicate rows (100,000 → 96,146) |
| Gender | Dropped 18 rows labelled "Other" (→ 96,128) |
| Smoking history | Consolidated 6 categories into 4: `never`, `No Info`, `former` (merging former / ever / not current) and `current` |
| BMI outliers | Winsorised to the IQR bounds [13.71, 39.55], affecting 5.57% of rows |
| Encoding | Label encoding for `gender` and `smoking_history` |
| Scaling | `StandardScaler` on age, BMI, HbA1c and blood glucose |
| Split | 70% train / 15% validation / 15% test, stratified (67,289 / 14,419 / 14,420) |

**EDA findings**

- The features most correlated with diabetes are `blood_glucose_level` (r = 0.42), `HbA1c_level` (r = 0.41) and `age` (r = 0.26).
- Diabetes rates rise with age and are higher among patients with hypertension or heart disease.
- Diabetic patients tend to have a higher BMI, though the distributions overlap considerably.

### 2. Model training (`model_training_2.ipynb`)

**Architecture:** 8 input features → Dense(64) + BatchNorm + ReLU + Dropout(0.3) → Dense(32) + BatchNorm + ReLU + Dropout(0.3) → Dense(1, sigmoid), with 3,073 parameters in total.

| Setting | Value |
|---|---|
| Optimiser | Adam (learning rate 1e-3) |
| Loss | Binary cross-entropy |
| Batch size / max epochs | 512 / 100 |
| Class weights | `balanced` → {0: 0.548, 1: 5.667} |
| Callbacks | `EarlyStopping` on `val_auc` (patience 10, restore best weights), `ReduceLROnPlateau` on `val_loss` |

## Results

Test set (14,420 samples, 1,273 diabetic), decision threshold **0.05**:

| Metric | Score |
|---|---|
| ROC-AUC | 0.9738 |
| Recall (diabetic) | 0.9976 |
| Precision (diabetic) | 0.2241 |
| F1-score (diabetic) | 0.3660 |
| Accuracy | 0.6949 |

The threshold was deliberately set low to favour **recall**, because missing a diabetic patient is clinically costlier than a false alarm. The trade-off is low precision: many non-diabetic patients are flagged for follow-up. ROC-AUC is threshold-independent and shows the model separates the two classes well.

## Repository Structure

```
.
├── diabetes_prediction_dataset.csv   # Raw dataset
├── diabetes_preprocessing_eda.ipynb  # Cleaning, EDA, splitting
├── model_training_2.ipynb            # ANN training and evaluation
├── X_train.csv / y_train.csv         # Training split
├── X_val.csv   / y_val.csv           # Validation split
├── X_test.csv  / y_test.csv          # Test split
├── requirements.txt
└── .gitattributes
```

Running the training notebook also saves `training_curves.png` and `confusion_matrix.png`.

## Getting Started

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
jupyter lab
```

1. *(Optional)* Run `diabetes_preprocessing_eda.ipynb` to regenerate the train/validation/test CSVs.
2. Run `model_training_2.ipynb` to train and evaluate the model.

Keep all CSV files in the same folder as the notebooks.

## Limitations and Future Work

- **Precision is low** at the 0.05 threshold. Tuning the threshold on the validation set and comparing precision-recall trade-offs would give a more balanced operating point.
- **Baselines are missing.** Comparing against logistic regression, random forest or gradient boosting would show how much the ANN actually adds.
- **Resampling** (for example SMOTE) could be compared against class weighting.
- `HbA1c_level` and `blood_glucose_level` are near-diagnostic measurements, so the model may rely heavily on them. Performance without these features would be worth testing.
- This is an academic project and **not a medical diagnostic tool**.

## Tech Stack

Python · TensorFlow/Keras · scikit-learn · pandas · NumPy · matplotlib · seaborn · JupyterLab

## Author

**Faiz**, AI student. Course project for SAIA 2113 Machine Learning.
