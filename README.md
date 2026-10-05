# NPS-Driven Strategy for Aviation

Analysis of airline post-flight survey data to find which service touchpoints drive Net Promoter Score (NPS). A single Jupyter notebook builds regression models (Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost) that predict a passenger's NPS outcome from their ratings of individual touchpoints (booking, check-in, boarding, on-board, arrival, crew, announcements and so on), then uses feature importance and SHAP values to rank those touchpoints. On a held-out 20% test split of 73,398 survey responses, the tuned Random Forest reaches R² = 0.718; gradient-boosted models reach R² = 0.725.

## Business problem

NPS sorts customers into Promoters, Passives and Detractors, but the headline score does not say what to fix. An airline collects ratings for many separate touchpoints. The question here is which of those touchpoints actually move a passenger's NPS outcome, so that improvement spend can target the stages of the journey that matter most.

## Dataset

The data is **not included in this repository** and is not publicly available. The notebook was run in Google Colab against two CSV files that were removed from the repo (see the "Delete Dataset" commit):

| File (path in notebook) | Used in | Size reported in notebook |
|---|---|---|
| `/content/NPS.csv` | Cells 1-4 (baseline models) | Row count not printed |
| `/content/drive/MyDrive/Feb 71LakhRecordsNPS4CorelationV1.csv` | Cell 5 (main pipeline) | 73,398 rows x 85 columns |

Both are exports of an airline's passenger survey records, with Salesforce-style column names (`*__c`). The main file contains:

- About 35 touchpoint ratings on a 1-5 scale, for example `On_board_experience__c`, `Boarding_experience__c`, `Arrival_experience__c`, `Check_in_experience__c` and `Crew_helpfulness__c`. Many have missing values (for example `Snacks_and_beverage_if_experienced__c` has 24,725 non-null values).
- The NPS category columns `Promotors__c`, `Passive__c` and `Detractors__c`, plus `NPS_Type__c`.
- Booking and flight metadata (route, delay, equipment, seat, booking source) and passenger attributes (gender, date of birth, age).

Because the data includes personal passenger information, it should not be committed to a public repository. To rerun the notebook you need your own export with the same column names.

**NPS definition used.** In the main pipeline, NPS is computed for each row as:

```
NPS_Score = (Promotors__c - Detractors__c) / (Promotors__c + Passive__c + Detractors__c) * 100
```

In `NPS.csv`, `Promotors__c` is a 0/1 flag for each respondent (see the "Actual_NPS" column in the printed prediction table). With per-respondent flags, this formula gives +100 for a Promoter, 0 for a Passive and -100 for a Detractor. It is not an aggregate NPS over a group of passengers.

## Approach

**Baseline (notebook cells 1-4, `NPS.csv`)**
- Uses 16 touchpoint ratings as features. Rows with any missing value are dropped.
- The target is the binary `Promotors__c` flag, modelled as a regression problem.
- Uses an 80/20 train/test split (`random_state=42`) and fits a Linear Regression and an XGBoost regressor (100 trees, depth 5, learning rate 0.1).
- Also includes a correlation heatmap and an XGBoost feature-importance plot. Cell 4 writes test-set predictions to `NPS_Predictions.csv`, which is not in the repo.

**Main pipeline (notebook cell 5, 73,398-row export)**
1. Drops columns that are more than 50% missing and fills the remaining numeric gaps with the median. Categorical columns are label-encoded.
2. Computes `NPS_Score` with the formula above.
3. Selects features in three steps: keep features with |correlation with `NPS_Score`| > 0.2, then drop features with VIF >= 5, then use Recursive Feature Elimination with a 50-tree Random Forest to keep 10 features.
4. Splits the data 80/20 (`random_state=42`) and applies `StandardScaler` (fit on the training split).
5. Tunes a Random Forest with `RandomizedSearchCV` (10 candidates, 5-fold CV).
6. Compares five models on the test split and with 5-fold CV R² on the training split.
7. Computes impurity-based feature importances and a SHAP summary (`TreeExplainer`) for the Random Forest.
8. Saves the tuned Random Forest and the scaler with `joblib`.

## Results

All numbers are copied from the executed outputs in the notebook.

**Main pipeline: target `NPS_Score`, test split = 20% of 73,398 rows**

| Model | Test R² | 5-fold CV mean R² (train split) |
|---|---|---|
| XGBoost | 0.7251 | 0.7271 |
| Gradient Boosting | 0.7250 | 0.7271 |
| Random Forest (tuned, saved) | 0.7179 | 0.7225 |
| Ridge Regression | 0.6337 | 0.6399 |
| Lasso Regression | 0.6337 | 0.6399 |

The notebook reports R² only for these models. MAE and RMSE are imported but not printed. The saved model is the Random Forest, even though the boosted models score slightly higher.

**Baseline: target binary `Promotors__c`, test split = 20% of `NPS.csv`**

| Model | Test R² | Test RMSE |
|---|---|---|
| XGBoost | 0.6276 | 0.2832 |
| Linear Regression | 0.5297 | 0.3183 |

**Which touchpoints matter.** RFE kept these 10 features. They are listed in the order of the Random Forest's impurity-based importance plot:

1. `On_board_experience__c`
2. `Arrival_experience__c`
3. `Boarding_experience__c`
4. `Pre_travel_information_experience__c`
5. `Check_in_experience__c`
6. `Crew_helpfulness__c`
7. `Staff_efficiency_at_the_counter__c`
8. `Query_handling_Contact_center_Dottie__c`
9. `Did_you_receive_your_bag_within_25_min_o__c`
10. `Clarity_of_announcements__c`

**Business takeaways, as supported by the outputs:**
- **The core flight stages drive NPS.** On-board, arrival and boarding experience together account for most of the Random Forest's importance. On-board experience is also the clear top feature in the separate XGBoost baseline on `NPS.csv`.
- **Bad experiences hurt more than good ones help.** In the SHAP summary plot, low ratings on on-board, arrival and boarding experience push predictions down sharply (close to -100 at the extreme for on-board experience). High ratings add a much smaller positive amount. Fixing poor experiences at these stages is therefore likely to do more for NPS than further polishing stages that already score well.
- **Some touchpoints add little.** Announcement clarity, baggage delivery time and contact-centre query handling contribute little once the main journey stages are known.

The baseline correlation cell reports that `Detractors__c` is the factor most negatively correlated with `Promotors__c` (r = -0.7248). This is expected by construction, because the two are mutually exclusive NPS categories. It is not a service insight.

## Limitations

- **The data is not available.** The notebook cannot be rerun without a private export that uses the same schema.
- **Hyperparameter search was mostly broken.** The search grid includes `max_features='auto'`, which scikit-learn 1.3+ rejects. 35 of the 50 CV fits failed, so only 3 of the 10 sampled candidates were actually evaluated.
- **Preprocessing ran before the split.** Median imputation, label encoding and all three feature-selection steps (correlation filter, VIF, RFE) were run on the full dataset before the train/test split. This leaks a little information from the test set, so the test scores may be slightly optimistic. The scaler was correctly fit on the training split only.
- **The target is treated as continuous.** For each respondent the target only takes three values (or two in the baseline), but it is modelled with regression. A classification model (Promoter / Passive / Detractor) with accuracy, F1 and a confusion matrix would be a more natural framing. Class balance is not reported.
- **Importances show association, not cause.** Impurity-based importances and SHAP values describe what the model relies on. Touchpoint ratings are also correlated with one another.
- **Use the saved model with care.** The 10 input features must be passed in the column order used during training, and the notebook does not print that order (the importance plot is sorted by importance, not by column order). The model also predicts a per-respondent score, not an aggregate NPS.
- **Environment-specific code.** The notebook uses Colab paths (`/content/...`, Google Drive) and the Jupyter `display()` function. The baseline code appears in three near-identical cells.

## How to run

The executed outputs come from Google Colab running Python 3.11. Python 3.11 is recommended.

```bash
git clone https://github.com/Akshatb848/NPS-Driven-Strategy-for-Aviation.git
cd NPS-Driven-Strategy-for-Aviation

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook "NPS (1).ipynb"
```

Before you run the notebook, change `file_path` in cells 1-5 to point to your local copies of the CSV files. To make the Random Forest search work on current scikit-learn, change `'auto'` to `None` (or `1.0`) in `param_grid['max_features']`.

### Loading the saved model

`best_rf_model.zip` holds two `joblib` pickles:

| File | Contents |
|---|---|
| `best_rf_model.pkl` (about 49 MB uncompressed) | The tuned `RandomForestRegressor`: 100 trees, `max_depth=30`, `min_samples_split=10`, `min_samples_leaf=4`, `max_features='sqrt'`, `bootstrap=False`, 10 input features, trained on 58,718 rows |
| `scaler.pkl` | The `StandardScaler` fit on the training split (10 features) |

Both were pickled with scikit-learn 1.6.1. Install that version to load them without version-mismatch warnings. Only unpickle files from sources you trust.

```bash
pip install "scikit-learn==1.6.1"
unzip best_rf_model.zip
```

```python
import joblib
import numpy as np

model = joblib.load("best_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# X_raw: shape (n_samples, 10). Columns must be the 10 RFE-selected touchpoint
# ratings, in the same order as `selected_feature_names` in the notebook.
X_raw = np.array([[...]])
nps_score_pred = model.predict(scaler.transform(X_raw))   # roughly -100 to 100
```

## Project structure

```
.
├── NPS (1).ipynb        # Full analysis: baselines, main pipeline, model comparison, SHAP
├── best_rf_model.zip    # best_rf_model.pkl (tuned Random Forest) + scaler.pkl (StandardScaler)
├── requirements.txt     # Python dependencies for the notebook
└── README.md
```
