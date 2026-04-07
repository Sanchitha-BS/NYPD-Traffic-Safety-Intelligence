# NYPD-Traffic-Safety-Intelligence

> **An end-to-end data science project** analyzing 10 years of NYPD motor vehicle collision data (2014–2024) to uncover when, where, and why accidents happen in New York City — combining large-scale data engineering, statistical analysis, interactive visualizations, and a machine learning severity prediction model.

<br>

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-ML%20Model-339933?style=flat-square)](https://lightgbm.readthedocs.io)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com)
[![Folium](https://img.shields.io/badge/Folium-Geospatial%20Maps-77B829?style=flat-square)](https://python-visualization.github.io/folium)
[![Data](https://img.shields.io/badge/Data-NYPD%20Open%20Data-003087?style=flat-square)](https://data.cityofnewyork.us)
[![Status](https://img.shields.io/badge/Status-Complete-1D9E75?style=flat-square)]()



## 1. What This Project Does

Every year, tens of thousands of vehicle collisions occur across New York City's five boroughs. Each crash generates a record — when it happened, where, who was involved, what caused it, and how severe the outcome was. But raw crash records alone don't tell a story.

This project takes **~2 million raw NYPD collision records** across three separate datasets, merges and engineers them into a single clean analytical layer, and then answers four specific public safety questions through statistical analysis, interactive visualizations, and a machine learning model that can **predict accident severity** based on driver and crash attributes.

The goal is actionable intelligence — findings that a city planner, traffic safety officer, or policy maker can actually use.

---

## 2. Why It Matters

| Challenge | What This Project Addresses |
|---|---|
| Three massive datasets with no shared structure | Built a unified master table linking crash, vehicle, and person records via collision ID |
| Raw data has millions of rows, missing values, inconsistent labels | 13-step cleaning pipeline produces a reliable 101,000-record analytical dataset |
| Hard to know *when* and *where* risk is highest | Borough × time-of-day analysis with Chi-Square statistical validation |
| Human error causes are inconsistently labeled across years | Normalized and standardized 50+ factor variations into clean categories |
| No way to anticipate severity before a crash outcome is known | LightGBM classifier predicts 5-level severity with 99.3% F1-macro score |
| Safety equipment impact is anecdotal | Statistical odds ratios and Chi-Square tests quantify protective effects by age group and borough |

---

## 3. The Dataset

All data comes from the **NYPD Motor Vehicle Collisions** database, maintained and updated daily on NYC Open Data.

### Three source datasets — one for each dimension of a crash

**Crashes** — the event itself

| Key Fields | Description |
|---|---|
| `collision_id` | Unique crash identifier — the key that links all three datasets |
| `crash_date`, `crash_time` | When the collision occurred |
| `borough`, `on_street_name` | Where — NYC borough and exact street |
| `latitude`, `longitude` | GPS coordinates for spatial mapping |
| `contributing_factor_vehicle_1/2` | What caused it — human error, weather, mechanical |
| `number_of_motorist_injured/killed` | Outcome count |

**Vehicles** — the cars and drivers involved

| Key Fields | Description |
|---|---|
| `vehicle_type`, `vehicle_make`, `vehicle_year` | What kind of vehicle was involved |
| `driver_sex` | Driver gender |
| `driver_license_status` | Licensed / Permit / Unlicensed / Suspended |
| `vehicle_occupants` | Number of people in the vehicle |

**Persons** — the individuals involved *(JSON format)*

| Key Fields | Description |
|---|---|
| `person_type` | Driver / Pedestrian / Cyclist / Passenger |
| `bodily_injury` | Body region injured (Head, Neck, Chest, etc.) |
| `emotional_status` | Conscious / Shock / Unconscious / Apparent Death |
| `safety_equipment` | Seatbelt type used or "None" |
| `person_age` | Age of the individual |
| `position_in_vehicle` | Where in the vehicle they were seated |

### Data scale

| Dataset | Format | Raw Size | Records |
|---|---|---|---|
| Crashes | CSV | 447 MB | ~2 million events |
| Vehicles | CSV | 778 MB | ~4 million vehicle records |
| Persons | JSON | 191 MB | ~5 million person records |
| **Final analytical dataset** | **Parquet + CSV** | **22 MB** | **~101,000 driver records** |

---

## 4. Data Pipeline — How Three Datasets Became One

Getting from three raw files totaling 1.4GB to a clean, analysis-ready 101,000-record dataset required a 13-step pipeline. Here is exactly what was done and why.

```
RAW FILES: Crashes.csv · Vehicles.csv · Person_data.json
        │
        ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STEP 1 — Memory-efficient loading
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Columns cast to int32, float32, category, string
  → Reduces memory footprint for million-row datasets

  STEP 2 — Year filtering (2014–2024)
  → Removes out-of-scope records before any joins

  STEP 3 — Column name normalization
  → All columns: lowercase, spaces → underscores

  STEP 4 — Column subsetting
  → Kept only the fields relevant to research questions

  STEP 5 — Three-way merge on collision_id + vehicle_id
  → Creates one master row: crash + vehicle + person context

  STEP 6 — DateTime engineering
  → crash_date + crash_time → crash_datetime
  → crash_hour extracted for time-of-day analysis

  STEP 7 — Geospatial cleaning
  → Latitude/longitude → numeric, out-of-range → NaN

  STEP 8 — Missing value strategy
  → borough, street, driver_sex, license_status:
    missing → "Not Reported" (preserve the record)
  → bodily_injury, emotional_status:
    missing → dropped (required for severity scoring)

  STEP 9 — Safety equipment simplification
  → Ambiguous entries removed
  → Remaining → binary: "Yes" / "No"

  STEP 10 — Drivers-only filter
  → Pedestrians, cyclists, passengers excluded
  → Focus: driver behavior and human error

  STEP 11 — Severity score engineering
  → emotional_status → numeric score (0–12)
  → bodily_injury → numeric score (0–12)
  → Combined → total severity_score per driver
  → Binned into: No Injury / Minor / Moderate / Severe / Death

  STEP 12 — Time-of-day categorization
  → crash_hour → Morning / Afternoon / Evening / Night

  STEP 13 — Export
  → final_processed_data.parquet (fast, compressed)
  → final_processed_data.csv (portable)
        │
        ▼
  FINAL DATASET: 101,000 driver records · 24 columns
```

### Severity scoring — how accident outcomes were quantified

Instead of treating injury as a simple yes/no, a composite severity score was engineered:

```python
# Emotional status mapped to numeric scale
emotional_status_score = {
    'Apparent Death': 12, 'Unconscious': 11, 'Incoherent': 10,
    'Shock': 9, 'Semiconscious': 8, 'Conscious': 7, 'Does Not Apply': 0
}

# Bodily injury location mapped to numeric scale
bodily_injury_score = {
    'Entire Body': 12, 'Neck': 11, 'Back': 10, 'Head': 9,
    'Abdomen - Pelvis': 8, 'Chest': 7, 'Hip-Upper Leg': 6 ...
}

# Combined severity score → binned into 5 ordinal levels
severity_score = emotional_status_score + bodily_injury_score
bins = [0, 4, 9, 14, 19, 24]
labels = ['No Injury', 'Minor Accident', 'Moderate Accident', 'Severe Accident', 'Death']
```

---

## 5. Research Questions and Methodology

### Question 1 — When and where do accidents cluster?
*How do traffic accident patterns vary across NYC boroughs with respect to time of day and contributing factors?*

**Approach:**
- Crash times bucketed into Morning / Afternoon / Evening / Night
- Unique collision counts grouped by year and borough — trend analysis 2014–2024
- Top boroughs and streets identified for each time window
- Severity scores aggregated across borough × time combinations
- **Chi-Square test** confirmed borough and time of day are statistically dependent (p ≈ 2.6×10⁻⁵⁸)

**Key outputs:** time-of-day bar plots, borough × time heatmap, severity pivot tables, top contributing factors per time window

---

### Question 2 — What human errors drive the worst outcomes?
*How do human-error contributing factors impact accident severity, frequency, and demographics across NYC boroughs?*

**Approach:**
- Standardized 50+ spelling/capitalization variations of error labels into clean categories
- Assigned primary human error per collision from Vehicle 1 or Vehicle 2 factor fields
- Frequency and severity cross-tabulated — which errors cause the most accidents vs. the deadliest accidents
- Driver demographics (sex, license status) analyzed per error type
- **Folium interactive maps** plotted exact collision coordinates, colored by error type

**Key outputs:** ranked frequency tables, severity heatmap by error type, demographic bar plots, geospatial Folium maps

---

### Question 3 — Can machine learning predict how severe a crash will be?
*Can we predict accident severity level based on available features — and understand which factors most influence fatal outcomes?*

See [Section 6](#6-machine-learning--severity-prediction-model) for full model details.

---

### Question 4 — Does safety equipment actually save lives?
*How does safety equipment use influence injury severity across different age groups and NYC boroughs?*

**Approach:**
- Focused on drivers only for consistent exposure comparison
- High-severity defined as: Severe Accident + Death combined
- High-severity rates computed per equipment type
- **Chi-Square tests** confirmed differences are statistically significant
- **Odds ratios** calculated relative to "no equipment" baseline — quantified protective effect
- Interactive heatmap built: borough × age group × equipment type explorer

**Key outputs:** high-severity rate bar chart, odds ratio table, interactive Plotly heatmap

---

## 6. Machine Learning — Severity Prediction Model

### The problem
Given what we know at or before the time of a crash — the driver's age, license status, vehicle type, time of day, borough, and safety equipment — can we accurately predict how severe the outcome will be?

### Target variable
5-class ordinal prediction: `No Injury` → `Minor` → `Moderate` → `Severe` → `Death`

### Features used
```
vehicle_type       vehicle_make        driver_sex
driver_license_status  person_age      vehicle_year
safety_equipment   crash_hour          borough
position_in_vehicle  contributing_factor_vehicle_1/2
```

### Preprocessing
- Numerical features → StandardScaler
- Categorical features → OneHotEncoder
- Combined via ColumnTransformer pipeline

### Model — LightGBM Classifier
LightGBM was selected for its speed and accuracy on large tabular datasets with mixed feature types.

### Hyperparameter tuning — Optuna
Automated search across 20 optimization trials using **5-fold Stratified Cross-Validation**, optimizing F1-macro score to ensure balanced performance across all five severity classes.

```
Best F1-macro score (cross-validation): 0.9931
```

### Results on held-out test set

| Metric | Score |
|---|---|
| Accuracy | 1.00 |
| Precision (macro) | 1.00 |
| Recall (macro) | 1.00 |
| F1-Score (macro) | 1.00 |

The confusion matrix showed zero misclassifications across all five severity levels.

### Model explainability — SHAP analysis
SHAP (SHapley Additive exPlanations) values identified the features most responsible for predicting fatal outcomes:

| Feature | Role in Prediction |
|---|---|
| `crash_hour` | Night crashes → strongest predictor of fatal outcomes |
| `person_age` | Drivers 61+ → dramatically elevated fatal risk |
| `driver_license_status` | Unlicensed drivers → strong indicator of severe outcomes |
| `safety_equipment` | Lap belt + harness → strongest protective predictor |
| `borough` | Manhattan and Brooklyn → higher baseline severity |

> The alignment between SHAP-identified predictors and findings from the exploratory analysis (Questions 1–4) validates both the model and the statistical results — they tell the same story from two directions.

---

## 7. Key Findings

### Behavior is the #1 risk factor
Driver Inattention and Distraction accounted for over **70% of all human-error accidents** — consistently the leading cause across every time of day, every borough, and every year from 2014 to 2024.

### Afternoon volume, nighttime severity
The **Afternoon** period had the highest raw accident count. But **Night** crashes disproportionately resulted in severe injuries and fatalities — fewer crashes, but far deadlier when they occurred.

### Borough patterns are not random
A **Chi-Square test (p ≈ 2.6×10⁻⁵⁸)** confirmed that accident timing is not independent of borough. **Brooklyn and Queens** consistently recorded the highest accident frequencies. The accident distribution across boroughs and times of day follows a statistically significant pattern — not random chance.

### Age is a strong severity predictor
Drivers aged **61 and above** had markedly higher rates of severe or fatal outcomes compared to all other age groups — particularly in Queens and Brooklyn. This held true across multiple safety equipment types and was confirmed by both statistical analysis and SHAP model explainability.

### Unlicensed drivers appear in the worst outcomes
While most drivers involved in accidents were licensed, **unlicensed drivers were significantly overrepresented** among Alcohol Involvement and Aggressive Driving incidents — the two error types with the highest severe/fatal outcome rates.

### Safety equipment saves lives — with numbers to prove it
- Drivers with **no safety equipment** had high-severity accident rates approaching **14%**
- Drivers with **Lap Belt + Harness** had high-severity rates of **~5–6%**
- Chi-Square tests confirmed these differences are **statistically significant**
- Odds ratios quantified the protective effect compared to the no-equipment baseline

### COVID-19 is visible in the data
A sharp and visible **decline in accident counts in 2020** appears in the temporal trend analysis — consistent with NYC's lockdown period. Accident density heatmaps show clear reduction across all boroughs during pandemic years.

---

## 8. Visualizations

All visualizations are interactive and built with Plotly and Folium. They are rendered within the Jupyter notebook.

| Visualization | Type | What It Shows |
|---|---|---|
| Accident Density Heatmap | Animated Plotly map with year slider | How accident hotspots shift across NYC year-by-year (2014–2024) |
| Top Contributing Factors | Interactive donut chart with year dropdown | Top 10 accident causes for each year — how causes shifted over time |
| Monthly & Weekly Trends + ARIMA Forecast | Line charts | Seasonal patterns, COVID dip, and a 12-month accident count forecast |
| Accidents by Time of Day | Bar plot | Volume comparison — Morning / Afternoon / Evening / Night |
| Borough × Time Heatmap | Seaborn heatmap | Accident density by borough and time of day combined |
| Severity by Borough (Pivot Tables) | Styled pandas tables | Severity breakdown (No Injury → Death) per borough per time window |
| Top Human Errors (Ranked) | Horizontal bar chart | Top 10 human error causes by accident count |
| Severity × Human Error | Seaborn heatmap | Which errors cause the most severe outcomes |
| Driver Demographics by Error | Grouped bar charts | Gender and license status distribution across error types |
| Human Error Accident Locations | Folium interactive map | Geospatial plot of accident locations colored by error type |
| Safety Equipment vs. Severity | Bar chart | High-severity rates by equipment type |
| Borough × Age × Equipment | Interactive Plotly heatmap | User-selectable view of severity rates by equipment, age, and borough |
| SHAP Feature Importance | SHAP summary plot | Which features drive the LightGBM severity predictions most |

---

## 9. Repository Structure

```
nyc-traffic-accident-risk-analysis/
│
├── README.md                               ← You are here
│
├── notebook/
│   └── nypd-traffic-accident-analysis.ipynb  ← Full analysis: cleaning, EDA, ML, visualizations
│
├── data/
│   ├── final_processed_data.parquet        ← Clean analytical dataset (fast, compressed)
│   ├── final_processed_data.csv            ← Same dataset in portable CSV format
│   └── README_data.md                      ← Data source info and download instructions
│
└── docs/
    ├── nypd-traffic-accident-report.pdf    ← Full written project report
    └── nypd-traffic-accident-proposal.pdf  ← Original project proposal
```

> **Note on raw data:** The three source files (Crashes.csv · Vehicles.csv · Person_data.json) total ~1.4GB and are not included in this repository. Download them from [NYC Open Data — NYPD Motor Vehicle Collisions](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95). Place them in the `data/` folder before running the notebook. The processed output files are included so you can explore the analysis without re-running the full pipeline.

---

## 10. Tech Stack

| Tool | What It Was Used For |
|---|---|
| **Python** | All data processing, analysis, and modeling |
| **Pandas + NumPy** | Data loading, cleaning, merging, feature engineering |
| **Plotly Express + Graph Objects** | Interactive charts, animated heatmaps, donut charts, trend lines |
| **Seaborn + Matplotlib** | Static heatmaps, bar plots, confusion matrices |
| **Folium** | Interactive geospatial maps with collision location markers |
| **LightGBM** | Gradient boosting classifier for severity prediction |
| **Optuna** | Automated hyperparameter tuning with 5-fold cross-validation |
| **Scikit-learn** | Preprocessing pipelines, train/test split, evaluation metrics |
| **SHAP** | Model explainability — feature importance and local case analysis |
| **SciPy** | Chi-Square statistical hypothesis testing |
| **Statsmodels (ARIMA)** | Time-series forecasting of monthly accident counts |
| **Tabulate** | Clean markdown table rendering in notebook outputs |

---

## 11. How to Run This Project

### Prerequisites
- Python 3.8 or newer
- Jupyter Notebook or JupyterLab

### Step 1 — Clone the repository
```bash
git clone https://github.com/Sanchitha-BS/nyc-traffic-accident-risk-analysis.git
cd nyc-traffic-accident-risk-analysis
```

### Step 2 — Install all required packages
```bash
pip install pandas numpy plotly matplotlib seaborn folium lightgbm optuna shap scikit-learn statsmodels scipy tabulate jupyter
```

### Step 3 — Download the raw data
Go to [NYC Open Data — NYPD Motor Vehicle Collisions](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95) and download:
- `Crashes.csv`
- `Vehicles.csv`
- `Person_data.json`

Place all three files in the `data/` folder.

> If you want to skip the data pipeline and go straight to analysis, the pre-processed `final_processed_data.parquet` is already included. Jump to the **Research Questions** section of the notebook.

### Step 4 — Launch the notebook
```bash
jupyter notebook notebook/nypd-traffic-accident-analysis.ipynb
```

### Step 5 — Run all cells in order
`Kernel` → `Restart & Run All`

The notebook runs in sequence:
1. Data loading and pipeline (cells 1–15) — takes ~5–10 minutes for full raw data
2. Exploratory visualizations (cells 16–30)
3. Research Questions 1–4 (cells 31–110)
4. Machine learning model (cells 111–126)

