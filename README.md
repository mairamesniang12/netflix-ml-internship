# Netflix Machine Learning Analytics Suite
**Auspify Technologies - Machine Learning Internship Program (4-Week)**

This repository contains my submission for the Auspify ML internship task
list, based on a Netflix-style content catalog (`Dataset.csv`, 8,790
titles). All **6 of 6** tasks are completed (minimum required: 4). Each
task is delivered as a self-contained, already-executed Jupyter notebook
with outputs and plots included, so it can be reviewed without re-running
anything.

## Dataset
`Dataset.csv` — Netflix-style catalog with columns: `show_id`, `type`,
`title`, `director`, `country`, `date_added`, `release_year`, `rating`,
`duration`, `listed_in`. No missing values (unlisted directors/countries
are stored as the literal string `"Not Given"` rather than `NaN`).

## Tasks & Results

| # | Task | Notebook | Key Result |
|---|------|----------|------------|
| 1 | Content Recommendation System (Easy) | `task1_recommendation.ipynb` | TF-IDF + cosine similarity on genre/type/rating/director/country; avg genre overlap **0.90** vs **0.15** for a random baseline |
| 2 | Content Type Prediction (Easy) | `task2_classification.ipynb` | Logistic Regression / Decision Tree / Random Forest; **94.8%** accuracy with leakage removed (vs 69.7% majority baseline) |
| 3 | Audience Rating Classification (Medium) | `task3_rating.ipynb` | 12-class classification after grouping rare ratings; GridSearchCV-tuned Random Forest reaches **53.1%** accuracy (vs 36.5% baseline), F1-macro and confusion matrix included |
| 4 | Content Segmentation (Medium) | `task4_clustering.ipynb` | Genre-aware K-Means (multi-hot genres via SVD); k=6 chosen for interpretable business segments (e.g. US Comedies, International TV Dramas, Bollywood-style Dramas) |
| 5 | Trend Forecasting (Advanced) | `task5_forecasting.ipynb` | Monthly title-addition time series, partial trailing month removed; Linear Trend (MAE ≈ 43) outperformed a hand-implemented Holt-Winters (MAE ≈ 69) and a naive baseline (MAE ≈ 48); 6-month forecast generated |
| 6 | Content Success Analytics Engine (Advanced) | `task6_analytics_engine.ipynb` | End-to-end pipeline: feature engineering → 3 leakage-free classifiers + genre-aware clustering → automated insights (with an annualized, not-misleading YoY growth figure) → visual dashboard + text report |

### Bugs found and fixed during development
- **Task 1:** the original ranking logic assumed the top-ranked result was
  always the query title itself; when another title tied at similarity
  1.0, the query title could leak into its own recommendation list. Fixed
  by excluding the query's row index explicitly.
- **Tasks 2 & 6:** `has_director` was computed against `""`, but missing
  directors are encoded as the string `"Not Given"`, so the column was a
  silent constant. Fixed to compare against `"Not Given"`.
- **Tasks 2 & 6:** the raw numeric part of `duration` leaks the Movie/TV
  label almost perfectly (Movies: 3–312 minutes: TV Shows: 1–17 seasons)
  and was inflating classifier accuracy to ~99.8–99.9%. Removed from the
  classifier features; the honest, leakage-free accuracy is ~94–95%.
- **Tasks 4 & 6:** genre and country were `LabelEncoder`-ed into arbitrary
  integers before K-Means, imposing a fake ordering on unordered
  categories. Replaced with a multi-hot genre matrix reduced via SVD, and
  k was chosen for business interpretability (k=6) rather than blindly
  taking the silhouette-optimal k=2 (which just reproduces the Movie/TV
  split).
- **Task 5:** the trailing month (September 2021) is only ~25 days of
  data (extraction date), understating a full month; it was silently kept
  as a real data point. Fixed by dropping it. Also implemented Holt-Winters
  by hand in NumPy, since `statsmodels` could not be installed in the
  development sandbox — the update equations are the standard
  triple-exponential-smoothing formulas.
- **Task 6:** the automated "2021 vs 2020" growth insight compared a
  partial year (through Sep 25) against a full year, showing a false
  ~20% decline. Fixed by annualizing the partial year before comparing,
  which reveals real growth (~+8.6%).

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook
```
Open any `taskN_*.ipynb` and run all cells — `Dataset.csv` sits in the
same folder as every notebook, so no path changes are needed. Each
notebook is already executed with all outputs and plots saved inline;
re-running is only needed to verify or modify the analysis.

## Outputs
Pre-generated copies of every plot and the written report also live in
`outputs/` for quick viewing without opening Jupyter:
- `cluster_visualization.png`, `cluster_elbow.png` — Task 4
- `forecast_trend.png` — Task 5
- `analytics_dashboard.png`, `analytics_report.txt` — Task 6

## Screenshots
See `screenshots/README.txt` — screenshots of each notebook's executed
cells should be added there before final submission, as required by the
internship's project evaluation criteria.

## Skills Demonstrated
Recommendation systems, NLP fundamentals (TF-IDF), feature engineering,
data-leakage detection and mitigation, classification (Logistic
Regression, Decision Trees, Random Forest), hyperparameter tuning
(GridSearchCV), unsupervised learning (K-Means, PCA, SVD), time series
forecasting (Linear Trend, hand-implemented Holt-Winters), and end-to-end
ML pipeline design.

## Author
Mairame Samba NIANG — Machine Learning Intern

#Auspify #AuspifyTechnologies #AuspifyInternship #AuspifyProjects
