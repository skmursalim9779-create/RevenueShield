# Revenue Shield
### AI-Powered Booking Revenue Intelligence & Cancellation Risk Platform

**IBM SkillsBuild / BharatCares Internship — Final Capstone Project**
Domain: Data Analytics · Predictive Modelling · Generative AI
Author: **Mursalim**

---

## 1. Problem statement

A hotel group books a large volume of reservations, but a significant share of that
contracted value never turns into cash because guests cancel. Traditional dashboards
report the cancellation percentage *after* the money is gone. They answer *what
happened* and stop there.

**Revenue Shield** closes the loop taught in the masterclass —

> **Data → Information → Insight → Decision → Action**

It ingests raw booking data, cleans it, reports the KPIs that matter, explains *why*
the numbers move, predicts *which individual bookings* will cancel before they do,
attaches a **euro value** to that risk, and finally writes an executive decision brief
with a prioritised, downloadable action register.

The guiding principle of the build: a project is not better because it has more charts.
It is better because it produces a decision someone can act on tomorrow morning.

---

## 2. Dataset

| | |
|---|---|
| **Name** | Hotel Booking Demand |
| **Source (Kaggle)** | https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand |
| **File used** | `hotel_bookings.csv` |
| **Rows × columns** | ~119,390 × 32 |
| **Grain** | One row = one hotel reservation |
| **Target variable** | `is_canceled` (1 = cancelled, 0 = honoured) |
| **Licence** | Open data, derived from *Antonio, Almeida & Nunes (2019), Data in Brief 22* |

> This dataset was **not** used in any of the masterclass sessions. The masterclass
> worked with e-commerce sales data; this project deliberately uses an independent
> public dataset from a different industry.

**Getting the data:** download `hotel_bookings.csv` from the Kaggle link above and
either upload it in the app sidebar or place it next to `app.py`.
If the file is absent the app automatically runs a built-in statistical simulator with
an identical schema, so the platform is **never broken for a reviewer** — the data
source in use is always displayed in the sidebar and on the overview page.

---

## 3. What the platform does

| Page | Question it answers | Contents |
|---|---|---|
| **Executive overview** | *What is happening?* | 8 headline KPIs, realised-vs-destroyed revenue by month, cancellation-rate trend, property split, lead-time risk curve |
| **Revenue & segment lab** | *Why is it happening?* | Value-vs-risk bubble map for any dimension, revenue contribution and revenue-destroyed rankings, segment scorecard, month × segment cancellation heatmap |
| **Risk engine (ML)** | *What will go wrong?* | Random Forest cancellation classifier, ROC curve, confusion matrix, business-readable driver importance, risk banding, and a filterable **save-desk queue** of the highest-value bookings most likely to cancel |
| **Decision brief** | *What should we do?* | Auto-generated Fact → Insight → Action cards with a euro value on each, a downloadable action register (CSV) and brief (Markdown), plus an optional Gemini-narrated CEO memo |
| **Data & method** | *Can we trust it?* | Cleaning audit trail, missing-value profile, retention statistics, methodology and leakage controls |

### Key performance indicators tracked
Realised revenue · contracted booking value · revenue lost to cancellations ·
revenue leakage % · cancellation rate · ADR · average booking value ·
average lead time · average length of stay · repeat-guest mix ·
probability-weighted revenue at risk.

---

## 4. Technical architecture

```
                       hotel_bookings.csv  /  built-in simulator
                                    |
                    [ 1 ] INGEST  ->  pandas
                                    |
                    [ 2 ] CLEAN   ->  de-duplicate, impute, remove impossible ADR
                                      and ghost bookings, drop leakage columns
                                    |
                    [ 3 ] ENGINEER->  booking_value, realised/lost revenue,
                                      lead-time buckets, party type,
                                      engagement flags, arrival period
                                    |
        +---------------------------+---------------------------+
        |                           |                           |
   [ 4 ] KPI LAYER           [ 5 ] SEGMENT LAYER        [ 6 ] ML LAYER
   descriptive metrics       value-vs-risk mapping      RandomForest + ColumnTransformer
        |                           |                           |
        +---------------------------+---------------------------+
                                    |
                    [ 7 ] INSIGHT ENGINE  ->  fact / insight / action
                                              + euro value at stake
                                    |
                    [ 8 ] PRESENTATION -> Streamlit UI (5 pages)
                          + optional Gemini narrative memo
```

- **Backend, ML and front-end all live in a single file, `app.py`**, as required by the
  submission guidelines.
- **Model:** `RandomForestClassifier` (250 trees, `class_weight="balanced_subsample"`)
  inside a `Pipeline` with a `ColumnTransformer` — median imputation for 13 numeric
  features, most-frequent imputation plus one-hot encoding for 8 categorical features.
- **Validation:** stratified 80/20 hold-out split; scored on ROC-AUC, precision, recall,
  F1, accuracy and a confusion matrix, all shown live in the UI.
- **Leakage control:** `reservation_status` and `reservation_status_date` encode the
  outcome directly and are dropped before modelling. The model only ever sees
  information that exists at the moment of booking.
- **Explainability:** one-hot importances are folded back onto the original business
  variable and relabelled in plain English, so "deposit policy" appears as a driver
  rather than `cat__deposit_type_Non Refund`.
- **Caching:** `st.cache_data` for the data pipeline and `st.cache_resource` for the
  fitted model, so filter changes are instant after the first run.
- **Generative AI is optional by design.** The brief is generated deterministically from
  the data; a Gemini key merely restyles it as a CEO memo. No API key, no internet — the
  platform still works end to end.

---

## 5. Quick start

```bash
# 1. clone / download this repository
git clone <your-repository-url>
cd <repository-folder>

# 2. (optional but recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt

# 4. (optional) place hotel_bookings.csv from Kaggle next to app.py

# 5. launch
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**.
Streamlit serves both the backend and the front-end, so no separate server is needed.

**First-run notes**
- Without the CSV the sidebar shows *"Built-in simulator (demo mode)"* — everything
  still works; upload the Kaggle file at any time to switch to real data.
- The model trains once per filter combination (a few seconds) and is then cached.
- To enable the generative memo, paste a Google Gemini API key on the
  *Decision brief → Generative-AI memo* tab.

---

## 6. Repository contents

| File | Purpose |
|---|---|
| `app.py` | Complete project — data pipeline, ML model, insight engine and full Streamlit UI |
| `requirements.txt` | Pinned Python dependencies |
| `README.md` | This file |
| `project_report.docx` | Project report with methodology, results and interface walkthrough |

---

## 7. Representative findings

Run against the Kaggle dataset, the platform surfaces findings of this shape (exact
figures are recomputed live from whatever data is loaded — nothing is hard-coded):

- **Revenue leakage** — roughly a third of all bookings cancel, destroying a comparable
  share of contracted value. Conversion integrity, not demand generation, is the binding
  constraint.
- **Lead time is the dominant driver** — cancellation risk climbs steeply with how far
  ahead a guest books. Long-horizon bookings behave like free options.
- **The deposit policy inverts** — `Non Refund` bookings cancel at a *higher* rate than
  `No Deposit` ones, so the policy is tracking risk rather than deterring it.
- **Channel concentration** — the dominant online travel agent carries both the largest
  revenue share and the largest absolute loss.
- **Repeat guests and engaged guests cancel far less** — loyalty and a single pre-arrival
  interaction both function as cheap risk controls.

Each finding is delivered as *Fact → Insight → Action* with the value at stake attached,
and the model converts the lot into a daily save-desk queue ranked by
`cancellation probability × booking value`.

---

## 8. Limitations & next steps

- The dataset covers 2015–2017 for two Portuguese properties; coefficients should be
  refit before transfer to another market.
- Costs (commission, variable cost per occupied room) are not in the source data, so the
  platform quantifies **revenue** at risk rather than **margin** at risk.
- The 30 % save rate used to size recoverable revenue is a planning assumption and should
  be replaced by measured results after the first intervention cycle.
- Natural extensions: an overbooking optimiser driven by the risk scores, SHAP values for
  per-booking explanations, and a scheduled job that emails the save-desk queue each morning.

---

## 9. Acknowledgements

Built for the IBM SkillsBuild × BharatCares Data Analytics & Generative AI internship.
Dataset courtesy of Antonio, Almeida & Nunes via Kaggle (uploader: *jessemostipak*).


## Safe Enhancements Added
The original Revenue Shield implementation has been preserved. The final package additionally includes:
- non-destructive data-quality validation
- duplicate/missing-cell visibility
- evidence-mode clarification for simulator vs real CSV results
- CSV export for the currently loaded dataset
- optional risk-queue CSV export
- defensive error handling around enhancement-only UI elements

These additions are isolated and do not remove the original BI, ML, dashboard, or AI workflow.

## Python 3.14 Compatibility Update
This build is updated for Python 3.14 using current compatible releases of Streamlit,
pandas, NumPy, scikit-learn, Plotly, and Requests. Python 3.14 can therefore be used
for the local development environment; no Python 3.11 installation is required for
this updated dependency set.
