"""
=======================================================================================
 REVENUE SHIELD  -  AI-Powered Booking Revenue Intelligence & Cancellation Risk Platform
=======================================================================================
 IBM SkillsBuild / BharatCares Internship - Final Capstone Project
 Domain      : Data Analytics + Generative AI + Predictive Modelling
 Dataset     : Hotel Booking Demand (Kaggle - jessemostipak/hotel-booking-demand)
 Author      : Mursalim
---------------------------------------------------------------------------------------
 WHAT THIS PROJECT DOES
---------------------------------------------------------------------------------------
 This is NOT a "chart dump". It is a Business Intelligence decision platform that walks
 the full BI chain taught in the masterclass:

        DATA  ->  INFORMATION  ->  INSIGHT  ->  DECISION  ->  ACTION

 It ingests raw, dirty booking data, cleans it, surfaces the KPIs that matter, explains
 WHY the numbers move (drivers), predicts WHICH bookings will cancel (ML), quantifies
 HOW MUCH money is at risk (Euro value), and finally writes an executive Decision Brief
 in plain English with a prioritised, downloadable action list.

 Pages
   1. Executive Overview      - what is happening (KPIs + trends)
   2. Revenue & Segment Lab   - where the money comes from (drivers, mix, pricing)
   3. Risk Engine (ML)        - what will go wrong (cancellation prediction + drivers)
   4. Decision Brief (AI)     - what management must do on Monday morning

 Everything - backend, ML, analytics and the entire front-end - lives in THIS single
 Python file, as required by the submission guidelines.
---------------------------------------------------------------------------------------
 HOW TO RUN
---------------------------------------------------------------------------------------
     pip install -r requirements.txt
     streamlit run app.py

 Then open http://localhost:8501 in a browser.

 Data options (sidebar):
   A. Upload `hotel_bookings.csv` downloaded from Kaggle  (recommended)
   B. Drop `hotel_bookings.csv` next to this file - it is auto-detected
   C. Click "Use built-in demo data" - a statistically faithful simulator runs so the
      app is never broken for a reviewer who has not downloaded the CSV yet.
=======================================================================================
"""

from __future__ import annotations

import io
import json
import os
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Python 3.14 compatibility guard.
# The project can run on modern Python releases; dependencies are pinned in
# requirements.txt to versions with Python 3.14 wheels/support.
import sys
if sys.version_info < (3, 12):
    raise RuntimeError(
        "Revenue Shield requires Python 3.12+ with the current dependency set. "
        "Python 3.14 is recommended for this build."
    )

# ======================================================================================
# 0. GLOBAL CONFIG
# ======================================================================================

APP_TITLE = "Revenue Shield"
APP_SUB = "AI-Powered Booking Revenue Intelligence & Cancellation Risk Platform"
CURRENCY = "EUR"
CUR = "\u20ac"  # euro sign - the source dataset is priced in EUR
DATA_FILENAME = "hotel_bookings.csv"
RANDOM_STATE = 42
MODEL_ROW_CAP = 45_000  # keep training snappy on a laptop
DEFAULT_INTERVENTION_SUCCESS = 0.30  # % of at-risk revenue realistically saved

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

PALETTE = ["#1f6feb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#06b6d4", "#e11d48"]


# ======================================================================================
# 1. DATA LAYER  -  loading, simulating, cleaning, feature engineering
# ======================================================================================

def make_demo_data(n_rows: int = 24_000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate a schema-identical simulation of the Kaggle hotel booking dataset.

    This exists purely so the platform is demonstrable without the CSV present.
    Relationships (lead time, deposit type, repeat guests, special requests,
    previous cancellations) are injected deliberately so that the analytics and
    the ML model behave the way they do on the genuine data.
    """
    rng = np.random.default_rng(seed)

    hotel = rng.choice(["City Hotel", "Resort Hotel"], n_rows, p=[0.66, 0.34])
    lead_time = np.clip(rng.gamma(2.0, 55, n_rows), 0, 640).astype(int)
    year = rng.choice([2015, 2016, 2017], n_rows, p=[0.15, 0.47, 0.38])
    month = rng.choice(MONTH_ORDER, n_rows,
                       p=np.array([6, 6, 8, 9, 10, 10, 11, 12, 9, 8, 6, 5]) / 100)
    seg = rng.choice(
        ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups", "Complementary"],
        n_rows, p=[0.47, 0.20, 0.13, 0.08, 0.11, 0.01],
    )
    deposit = rng.choice(["No Deposit", "Non Refund", "Refundable"], n_rows,
                         p=[0.875, 0.12, 0.005])
    customer = rng.choice(["Transient", "Transient-Party", "Contract", "Group"], n_rows,
                          p=[0.75, 0.21, 0.035, 0.005])
    country = rng.choice(
        ["PRT", "GBR", "FRA", "ESP", "DEU", "ITA", "IRL", "BEL", "BRA", "NLD", "USA"],
        n_rows, p=[0.40, 0.12, 0.11, 0.09, 0.07, 0.04, 0.035, 0.03, 0.03, 0.03, 0.045],
    )
    room = rng.choice(list("ADEFGBC"), n_rows,
                      p=[0.62, 0.08, 0.06, 0.05, 0.04, 0.09, 0.06])
    meal = rng.choice(["BB", "HB", "SC", "FB"], n_rows, p=[0.77, 0.12, 0.10, 0.01])
    repeated = rng.choice([0, 1], n_rows, p=[0.968, 0.032])
    prev_cancel = rng.poisson(0.09, n_rows)
    prev_ok = rng.poisson(0.14, n_rows)
    changes = rng.poisson(0.22, n_rows)
    waiting = np.where(rng.random(n_rows) < 0.03, rng.integers(1, 120, n_rows), 0)
    special = rng.poisson(0.57, n_rows).clip(0, 5)
    parking = (rng.random(n_rows) < 0.062).astype(int)
    adults = rng.choice([1, 2, 3], n_rows, p=[0.22, 0.72, 0.06])
    children = rng.choice([0, 1, 2], n_rows, p=[0.92, 0.05, 0.03])
    babies = (rng.random(n_rows) < 0.008).astype(int)
    wkend = rng.poisson(0.93, n_rows).clip(0, 8)
    week = rng.poisson(2.5, n_rows).clip(0, 20)

    season = pd.Series(month).map(
        {m: s for m, s in zip(MONTH_ORDER, [.80, .82, .95, 1.0, 1.05, 1.15,
                                            1.30, 1.35, 1.10, .95, .80, .88])}
    ).to_numpy()
    base_adr = np.where(hotel == "Resort Hotel", 96, 108)
    adr = np.clip(
        base_adr * season
        + (adults - 2) * 21 + children * 17
        + pd.Series(room).map({"A": 0, "B": 9, "C": 26, "D": 14,
                               "E": 24, "F": 36, "G": 52}).to_numpy()
        + rng.normal(0, 22, n_rows),
        18, 460,
    ).round(2)

    # ---- cancellation generating process (logistic) --------------------------------
    z = (
        -1.62
        + 0.0041 * lead_time
        + 1.95 * (deposit == "Non Refund")
        + 0.58 * (seg == "Groups")
        + 0.41 * (seg == "Online TA")
        - 0.62 * (seg == "Direct")
        - 0.55 * (seg == "Corporate")
        + 0.66 * (country == "PRT")
        - 1.35 * repeated
        + 0.92 * np.minimum(prev_cancel, 3)
        - 0.30 * prev_ok
        - 0.42 * special
        - 0.55 * parking
        - 0.26 * changes
        + 0.004 * waiting
        + 0.0021 * (adr - 100)
    )
    p = 1 / (1 + np.exp(-z))
    is_canceled = (rng.random(n_rows) < p).astype(int)

    df = pd.DataFrame({
        "hotel": hotel,
        "is_canceled": is_canceled,
        "lead_time": lead_time,
        "arrival_date_year": year,
        "arrival_date_month": month,
        "arrival_date_week_number": rng.integers(1, 54, n_rows),
        "arrival_date_day_of_month": rng.integers(1, 29, n_rows),
        "stays_in_weekend_nights": wkend,
        "stays_in_week_nights": week,
        "adults": adults,
        "children": children.astype(float),
        "babies": babies,
        "meal": meal,
        "country": country,
        "market_segment": seg,
        "distribution_channel": np.where(seg == "Direct", "Direct",
                                         np.where(seg == "Corporate", "Corporate", "TA/TO")),
        "is_repeated_guest": repeated,
        "previous_cancellations": prev_cancel,
        "previous_bookings_not_canceled": prev_ok,
        "reserved_room_type": room,
        "assigned_room_type": room,
        "booking_changes": changes,
        "deposit_type": deposit,
        "days_in_waiting_list": waiting,
        "customer_type": customer,
        "adr": adr,
        "required_car_parking_spaces": parking,
        "total_of_special_requests": special,
    })

    # inject realistic dirt so the cleaning stage is not cosmetic
    dirty = rng.choice(df.index, size=int(0.012 * n_rows), replace=False)
    df.loc[dirty[: len(dirty) // 2], "children"] = np.nan
    df.loc[dirty[len(dirty) // 2:], "country"] = np.nan
    dupes = df.sample(int(0.004 * n_rows), random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)
    return df


def load_raw(uploaded_file) -> tuple[pd.DataFrame, str]:
    """Return (raw dataframe, source label). Priority: upload > local file > demo."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), "Uploaded CSV"
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_FILENAME)
    if os.path.exists(local):
        return pd.read_csv(local), f"Local file ({DATA_FILENAME})"
    return make_demo_data(), "Built-in simulator (demo mode)"


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply an auditable cleaning pipeline and return the log of what was done."""
    log: list[str] = []
    df = raw.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    start = len(df)

    # columns that are >90% null / pure identifiers add no decision value
    for col in ("company", "agent", "reservation_status", "reservation_status_date"):
        if col in df.columns:
            df = df.drop(columns=col)
    log.append("Dropped identifier / leakage columns (company, agent, reservation_status).")

    before = len(df)
    df = df.drop_duplicates()
    if before - len(df):
        log.append(f"Removed {before - len(df):,} exact duplicate booking rows.")

    if "children" in df.columns:
        n_null = int(df["children"].isna().sum())
        df["children"] = pd.to_numeric(df["children"], errors="coerce").fillna(0)
        if n_null:
            log.append(f"Imputed {n_null:,} missing 'children' values with 0 (no child recorded).")
    if "country" in df.columns:
        n_null = int(df["country"].isna().sum())
        df["country"] = df["country"].fillna("UNKNOWN")
        if n_null:
            log.append(f"Labelled {n_null:,} missing 'country' values as UNKNOWN.")

    for c in ("adr", "lead_time", "adults", "children", "babies",
              "stays_in_week_nights", "stays_in_weekend_nights"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "adr" in df.columns:
        bad = int(((df["adr"] < 0) | (df["adr"] > 1000)).sum())
        df = df[(df["adr"] >= 0) & (df["adr"] <= 1000)]
        if bad:
            log.append(f"Removed {bad:,} rows with impossible ADR (<0 or >1000 {CURRENCY}).")

    guests = df.get("adults", 0) + df.get("children", 0) + df.get("babies", 0)
    nights = df.get("stays_in_week_nights", 0) + df.get("stays_in_weekend_nights", 0)
    ghost = int(((guests == 0) | (nights == 0)).sum())
    df = df[(guests > 0) & (nights > 0)]
    if ghost:
        log.append(f"Removed {ghost:,} ghost bookings (zero guests or zero nights).")

    df = df.dropna(subset=["is_canceled"])
    df["is_canceled"] = df["is_canceled"].astype(int)
    log.append(f"Final clean dataset: {len(df):,} rows retained out of {start:,} "
               f"({100 * len(df) / max(start, 1):.1f}%).")
    return df.reset_index(drop=True), log


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the derived business variables the whole platform reasons with."""
    d = df.copy()
    d["total_nights"] = d["stays_in_week_nights"] + d["stays_in_weekend_nights"]
    d["total_guests"] = d["adults"] + d["children"] + d["babies"]
    d["booking_value"] = (d["adr"] * d["total_nights"]).round(2)
    d["realised_revenue"] = np.where(d["is_canceled"] == 1, 0.0, d["booking_value"])
    d["lost_revenue"] = np.where(d["is_canceled"] == 1, d["booking_value"], 0.0)

    d["lead_time_bucket"] = pd.cut(
        d["lead_time"],
        bins=[-1, 7, 30, 90, 180, 10_000],
        labels=["0-7 days", "8-30 days", "31-90 days", "91-180 days", "180+ days"],
    )
    d["party_type"] = np.where(
        d["children"] + d["babies"] > 0, "Family",
        np.where(d["adults"] == 1, "Solo", "Couple / Adults"),
    )
    d["has_special_request"] = (d["total_of_special_requests"] > 0).astype(int)
    d["room_downgraded"] = (d.get("reserved_room_type") != d.get("assigned_room_type")).astype(int)

    month_num = {m: i + 1 for i, m in enumerate(MONTH_ORDER)}
    d["arrival_month_num"] = d["arrival_date_month"].map(month_num)
    d["arrival_period"] = pd.to_datetime(
        dict(year=d["arrival_date_year"], month=d["arrival_month_num"], day=1),
        errors="coerce",
    )
    d["arrival_label"] = d["arrival_period"].dt.strftime("%Y-%m")
    return d


# ======================================================================================
# 2. KPI LAYER  -  "what is happening"
# ======================================================================================

def compute_kpis(d: pd.DataFrame) -> dict:
    total = len(d)
    canc = int(d["is_canceled"].sum())
    kept = d[d["is_canceled"] == 0]
    booked_value = float(d["booking_value"].sum())
    realised = float(d["realised_revenue"].sum())
    lost = float(d["lost_revenue"].sum())
    return {
        "total_bookings": total,
        "confirmed_bookings": total - canc,
        "cancellations": canc,
        "cancellation_rate": canc / total if total else 0.0,
        "booked_value": booked_value,
        "realised_revenue": realised,
        "lost_revenue": lost,
        "revenue_leakage_pct": lost / booked_value if booked_value else 0.0,
        "adr": float(kept["adr"].mean()) if len(kept) else 0.0,
        "avg_booking_value": float(kept["booking_value"].mean()) if len(kept) else 0.0,
        "avg_lead_time": float(d["lead_time"].mean()),
        "avg_nights": float(kept["total_nights"].mean()) if len(kept) else 0.0,
        "repeat_guest_rate": float(d["is_repeated_guest"].mean()),
    }


def monthly_frame(d: pd.DataFrame) -> pd.DataFrame:
    g = (d.dropna(subset=["arrival_period"])
           .groupby("arrival_period", as_index=False)
           .agg(bookings=("is_canceled", "size"),
                cancellations=("is_canceled", "sum"),
                realised_revenue=("realised_revenue", "sum"),
                lost_revenue=("lost_revenue", "sum"),
                adr=("adr", "mean")))
    g["cancellation_rate"] = g["cancellations"] / g["bookings"]
    g["label"] = g["arrival_period"].dt.strftime("%b %Y")
    return g.sort_values("arrival_period")


def segment_frame(d: pd.DataFrame, dim: str, min_share: float = 0.01) -> pd.DataFrame:
    """Aggregate any dimension into the standard risk/revenue view."""
    g = (d.groupby(dim, as_index=False)
           .agg(bookings=("is_canceled", "size"),
                cancellations=("is_canceled", "sum"),
                realised_revenue=("realised_revenue", "sum"),
                lost_revenue=("lost_revenue", "sum"),
                adr=("adr", "mean"),
                avg_lead_time=("lead_time", "mean")))
    g["cancellation_rate"] = g["cancellations"] / g["bookings"]
    g["share_of_bookings"] = g["bookings"] / g["bookings"].sum()
    g = g[g["share_of_bookings"] >= min_share]
    return g.sort_values("realised_revenue", ascending=False)


# ======================================================================================
# 3. MACHINE LEARNING LAYER  -  "what will go wrong"
# ======================================================================================

MODEL_NUMERIC = [
    "lead_time", "total_nights", "total_guests", "adr", "booking_changes",
    "previous_cancellations", "previous_bookings_not_canceled",
    "days_in_waiting_list", "total_of_special_requests",
    "required_car_parking_spaces", "is_repeated_guest",
    "arrival_month_num", "room_downgraded",
]
MODEL_CATEGORICAL = [
    "hotel", "market_segment", "distribution_channel", "deposit_type",
    "customer_type", "reserved_room_type", "meal", "party_type",
]

FRIENDLY_NAMES = {
    "lead_time": "Lead time (days booked in advance)",
    "deposit_type": "Deposit policy",
    "adr": "Average daily rate (price)",
    "market_segment": "Market segment / channel",
    "total_of_special_requests": "Special requests made",
    "previous_cancellations": "Guest's past cancellations",
    "required_car_parking_spaces": "Car parking requested",
    "booking_changes": "Booking modifications",
    "customer_type": "Customer type",
    "total_nights": "Length of stay",
    "is_repeated_guest": "Repeat guest",
    "country": "Source market",
    "total_guests": "Party size",
    "arrival_month_num": "Arrival month (seasonality)",
    "distribution_channel": "Distribution channel",
    "reserved_room_type": "Room type reserved",
    "party_type": "Party type",
    "meal": "Meal plan",
    "days_in_waiting_list": "Days on waiting list",
    "previous_bookings_not_canceled": "Guest's past completed stays",
    "room_downgraded": "Room reassigned vs reserved",
    "hotel": "Property",
}


def build_model_pipeline() -> Pipeline:
    num = Pipeline([("impute", SimpleImputer(strategy="median"))])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=25)),
    ])
    pre = ColumnTransformer(
        [("num", num, MODEL_NUMERIC), ("cat", cat, MODEL_CATEGORICAL)],
        remainder="drop",
    )
    clf = RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("prep", pre), ("model", clf)])


def group_importances(pipe: Pipeline) -> pd.DataFrame:
    """Fold one-hot importances back onto the original business variable."""
    pre: ColumnTransformer = pipe.named_steps["prep"]
    names = list(pre.get_feature_names_out())
    imps = pipe.named_steps["model"].feature_importances_
    rows = []
    for name, imp in zip(names, imps):
        stripped = name.split("__", 1)[-1]
        source = next((c for c in MODEL_CATEGORICAL if stripped.startswith(c + "_")), None)
        rows.append((source or stripped, imp))
    out = (pd.DataFrame(rows, columns=["feature", "importance"])
             .groupby("feature", as_index=False)["importance"].sum()
             .sort_values("importance", ascending=False))
    out["driver"] = out["feature"].map(lambda f: FRIENDLY_NAMES.get(f, f))
    out["importance_pct"] = out["importance"] / out["importance"].sum()
    return out.reset_index(drop=True)


def train_risk_model(d: pd.DataFrame) -> dict:
    """Train, evaluate and score. Returns every artefact the UI and brief need."""
    work = d if len(d) <= MODEL_ROW_CAP else d.sample(MODEL_ROW_CAP, random_state=RANDOM_STATE)
    feats = [c for c in MODEL_NUMERIC + MODEL_CATEGORICAL if c in work.columns]
    X, y = work[feats], work["is_canceled"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    pipe = build_model_pipeline()
    pipe.fit(X_tr, y_tr)

    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_te, proba)

    metrics = {
        "rows_trained": int(len(X_tr)),
        "rows_tested": int(len(X_te)),
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(precision_score(y_te, pred, zero_division=0)),
        "recall": float(recall_score(y_te, pred, zero_division=0)),
        "f1": float(f1_score(y_te, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "confusion": confusion_matrix(y_te, pred).tolist(),
        "baseline_rate": float(y.mean()),
    }

    scored = d.copy()
    scored["risk_score"] = pipe.predict_proba(d[feats])[:, 1]
    scored["risk_band"] = pd.cut(
        scored["risk_score"], [-0.01, 0.35, 0.65, 1.01],
        labels=["Low", "Medium", "High"],
    )
    scored["revenue_at_risk"] = (scored["risk_score"] * scored["booking_value"]).round(2)

    return {
        "pipeline": pipe,
        "features": feats,
        "metrics": metrics,
        "roc": pd.DataFrame({"fpr": fpr, "tpr": tpr}),
        "importances": group_importances(pipe),
        "scored": scored,
    }


# ======================================================================================
# 4. INSIGHT ENGINE  -  fact -> insight -> risk/opportunity -> action -> value
# ======================================================================================

def _fmt_money(x: float) -> str:
    if abs(x) >= 1_000_000:
        return f"{CUR}{x / 1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"{CUR}{x / 1_000:.0f}K"
    return f"{CUR}{x:,.0f}"


def generate_insights(d: pd.DataFrame, art: dict) -> list[dict]:
    """Derive board-ready findings from the data. Nothing here is hard-coded -
    every number is recomputed from whatever dataset is loaded."""
    k = compute_kpis(d)
    scored = art["scored"]
    imp = art["importances"]
    out: list[dict] = []

    # --- 1. headline revenue leakage -------------------------------------------------
    out.append({
        "type": "Risk",
        "priority": 1,
        "title": "Cancellations are the single largest source of revenue leakage",
        "fact": (f"{k['cancellations']:,} of {k['total_bookings']:,} bookings "
                 f"({k['cancellation_rate']:.1%}) were cancelled, destroying "
                 f"{_fmt_money(k['lost_revenue'])} of contracted value."),
        "insight": (f"{k['revenue_leakage_pct']:.1%} of everything the sales engine books "
                    f"never converts to cash. Demand generation is not the constraint - "
                    f"conversion integrity is."),
        "action": ("Make cancellation rate a board-level KPI alongside revenue, and hold "
                   "each channel owner accountable for its own conversion rate."),
        "value": k["lost_revenue"],
    })

    # --- 2. lead-time decay ----------------------------------------------------------
    lt = (d.groupby("lead_time_bucket", observed=True)
            .agg(bookings=("is_canceled", "size"),
                 rate=("is_canceled", "mean"),
                 lost=("lost_revenue", "sum"))
            .reset_index())
    lt = lt[lt["bookings"] >= max(50, 0.02 * len(d))]
    if len(lt) >= 2:
        worst, best = lt.loc[lt["rate"].idxmax()], lt.loc[lt["rate"].idxmin()]
        out.append({
            "type": "Driver",
            "priority": 2,
            "title": "Cancellation risk compounds with booking lead time",
            "fact": (f"Bookings made {worst['lead_time_bucket']} ahead cancel at "
                     f"{worst['rate']:.1%} versus {best['rate']:.1%} for "
                     f"{best['lead_time_bucket']} bookings - a "
                     f"{(worst['rate'] - best['rate']) * 100:.0f} percentage point gap."),
            "insight": ("Long-horizon bookings are options, not commitments. The further out "
                        "a guest books, the more time competitors and life have to intervene."),
            "action": (f"Introduce a re-confirmation touchpoint 30 and 7 days before arrival "
                       f"for the {worst['lead_time_bucket']} cohort, plus a small "
                       f"non-refundable deposit or a free-change incentive to convert the "
                       f"option into a commitment."),
            "value": float(worst["lost"]),
        })

    # --- 3. deposit policy paradox ---------------------------------------------------
    if "deposit_type" in d.columns:
        dep = segment_frame(d, "deposit_type", min_share=0.002)
        if len(dep) >= 2:
            hi = dep.loc[dep["cancellation_rate"].idxmax()]
            lo = dep.loc[dep["cancellation_rate"].idxmin()]
            out.append({
                "type": "Risk",
                "priority": 3,
                "title": "The deposit policy is not doing the job it was designed to do",
                "fact": (f"'{hi['deposit_type']}' bookings cancel at "
                         f"{hi['cancellation_rate']:.1%} while '{lo['deposit_type']}' "
                         f"bookings cancel at {lo['cancellation_rate']:.1%}."),
                "insight": ("The policy correlates with the riskiest traffic rather than "
                            "suppressing it - it is being applied after risk is detected by "
                            "partners, or it is concentrated in a low-trust channel. Either "
                            "way the deterrent effect is absent."),
                "action": ("Audit where the high-cancellation deposit policy is applied, and "
                           "A/B test a partial-deposit + flexible-change product against it "
                           "for one quarter."),
                "value": float(hi["lost_revenue"]),
            })

    # --- 4. channel concentration risk ------------------------------------------------
    seg = segment_frame(d, "market_segment", min_share=0.02)
    if len(seg):
        top = seg.iloc[0]
        risky = seg.sort_values("lost_revenue", ascending=False).iloc[0]
        out.append({
            "type": "Risk",
            "priority": 4,
            "title": f"Channel concentration: '{top['market_segment']}' carries the business",
            "fact": (f"'{top['market_segment']}' delivers {top['share_of_bookings']:.0%} of "
                     f"bookings and {_fmt_money(top['realised_revenue'])} of realised revenue, "
                     f"at a {top['cancellation_rate']:.1%} cancellation rate. "
                     f"'{risky['market_segment']}' alone loses "
                     f"{_fmt_money(risky['lost_revenue'])}."),
            "insight": ("Dependence on a single intermediated channel means commission "
                        "pressure and its cancellation behaviour are both imported wholesale "
                        "into the P&L."),
            "action": ("Fund a direct-booking push (rate parity + loyalty perk) with the "
                       "objective of moving 10 percentage points of volume from the dominant "
                       "intermediary to Direct over 12 months."),
            "value": float(risky["lost_revenue"]),
        })

    # --- 5. loyalty opportunity -------------------------------------------------------
    if d["is_repeated_guest"].nunique() > 1:
        rep = d.groupby("is_repeated_guest")["is_canceled"].mean()
        new_r, rep_r = float(rep.get(0, 0)), float(rep.get(1, 0))
        share = float(d["is_repeated_guest"].mean())
        upside = (new_r - rep_r) * float(d["booking_value"].mean()) * len(d) * 0.05
        out.append({
            "type": "Opportunity",
            "priority": 5,
            "title": "Repeat guests are the cheapest cancellation insurance available",
            "fact": (f"Repeat guests cancel at {rep_r:.1%} against {new_r:.1%} for first-time "
                     f"guests, yet they are only {share:.1%} of the book."),
            "insight": ("Loyalty is not a marketing nicety here - it is a risk control. Every "
                        "point of repeat-guest mix converts directly into forecastable revenue."),
            "action": ("Launch a post-stay re-booking offer at checkout and target a 5 point "
                       "lift in repeat mix; even a modest shift is worth "
                       f"{_fmt_money(max(upside, 0))} in protected revenue."),
            "value": float(max(upside, 0)),
        })

    # --- 6. engagement signal ---------------------------------------------------------
    if "has_special_request" in d.columns and d["has_special_request"].nunique() > 1:
        sr = d.groupby("has_special_request")["is_canceled"].mean()
        out.append({
            "type": "Opportunity",
            "priority": 6,
            "title": "Engagement before arrival predicts arrival",
            "fact": (f"Guests who make at least one special request cancel at "
                     f"{float(sr.get(1, 0)):.1%} versus {float(sr.get(0, 0)):.1%} for guests "
                     f"who never interact after booking."),
            "insight": ("Interaction is a commitment signal. It is also manufacturable - the "
                        "hotel can prompt the interaction instead of waiting for it."),
            "action": ("Send a one-tap pre-arrival personalisation message (pillow, floor, "
                       "arrival time, parking) to every booking over 30 days lead time and "
                       "monitor the cohort's cancellation delta."),
            "value": float(d.loc[d["has_special_request"] == 0, "lost_revenue"].sum() * 0.1),
        })

    # --- 7. seasonality ---------------------------------------------------------------
    mon = (d.groupby("arrival_date_month", observed=True)
             .agg(rev=("realised_revenue", "sum"), rate=("is_canceled", "mean"),
                  adr=("adr", "mean"), n=("is_canceled", "size")).reset_index())
    if len(mon) >= 3:
        peak = mon.loc[mon["rev"].idxmax()]
        trough = mon.loc[mon["rev"].idxmin()]
        out.append({
            "type": "Opportunity",
            "priority": 7,
            "title": "Seasonal spread is wide enough to arbitrage",
            "fact": (f"{peak['arrival_date_month']} realises {_fmt_money(peak['rev'])} at an "
                     f"ADR of {CUR}{peak['adr']:.0f}, while {trough['arrival_date_month']} "
                     f"realises {_fmt_money(trough['rev'])} at {CUR}{trough['adr']:.0f}."),
            "insight": ("Fixed cost is flat across the year; revenue is not. The trough months "
                        "are where marginal contribution is won or lost."),
            "action": (f"Build a {trough['arrival_date_month']} demand package (corporate "
                       f"mid-week, long-stay discount, local events) and protect "
                       f"{peak['arrival_date_month']} inventory with tighter overbooking "
                       "limits driven by the risk model."),
            "value": float(peak["rev"] - trough["rev"]),
        })

    # --- 8. the ML-derived forward view -----------------------------------------------
    high = scored[scored["risk_band"] == "High"]
    at_risk_value = float(scored["revenue_at_risk"].sum())
    recover = at_risk_value * DEFAULT_INTERVENTION_SUCCESS
    top_driver = imp.iloc[0]["driver"] if len(imp) else "lead time"
    out.append({
        "type": "Action",
        "priority": 9,
        "title": "A targeted save-desk is worth real money, and the model can aim it",
        "fact": (f"The model (ROC-AUC {art['metrics']['roc_auc']:.3f}) flags "
                 f"{len(high):,} bookings as high risk, carrying "
                 f"{_fmt_money(float(high['booking_value'].sum()))} of contracted value. "
                 f"Probability-weighted revenue at risk across the whole book is "
                 f"{_fmt_money(at_risk_value)}."),
        "insight": (f"Risk is concentrated, not diffuse - '{top_driver}' dominates the "
                    "explanation, so interventions can be aimed at a small population "
                    "instead of blanketing every guest with discounts."),
        "action": (f"Route the daily High-risk queue to a save-desk with a tiered offer "
                   f"(free upgrade > flexible date > partial refund waiver). At a "
                   f"{DEFAULT_INTERVENTION_SUCCESS:.0%} save rate this recovers roughly "
                   f"{_fmt_money(recover)}."),
        "value": recover,
    })

    return sorted(out, key=lambda r: r["priority"])


def build_action_table(insights: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Priority": i + 1,
        "Category": r["type"],
        "Finding": r["title"],
        "Evidence": r["fact"],
        "Recommended action": r["action"],
        f"Value at stake ({CURRENCY})": round(r["value"], 2),
    } for i, r in enumerate(insights)])


def build_brief_markdown(d: pd.DataFrame, art: dict, insights: list[dict], source: str) -> str:
    k = compute_kpis(d)
    lines = [
        f"# {APP_TITLE} - Executive Decision Brief",
        f"_Generated {datetime.now():%d %B %Y, %H:%M} | Data source: {source} | "
        f"{k['total_bookings']:,} bookings analysed_",
        "",
        "## 1. Where the business stands",
        f"- Contracted booking value: **{_fmt_money(k['booked_value'])}**",
        f"- Realised revenue: **{_fmt_money(k['realised_revenue'])}**",
        f"- Revenue lost to cancellations: **{_fmt_money(k['lost_revenue'])}** "
        f"({k['revenue_leakage_pct']:.1%} leakage)",
        f"- Cancellation rate: **{k['cancellation_rate']:.1%}** | ADR: "
        f"**{CUR}{k['adr']:.2f}** | Average lead time: **{k['avg_lead_time']:.0f} days**",
        "",
        "## 2. Predictive risk position",
        f"- Model: Random Forest classifier | ROC-AUC **{art['metrics']['roc_auc']:.3f}** | "
        f"Recall **{art['metrics']['recall']:.1%}**",
        f"- Probability-weighted revenue at risk: "
        f"**{_fmt_money(float(art['scored']['revenue_at_risk'].sum()))}**",
        f"- Top three risk drivers: " + ", ".join(art["importances"].head(3)["driver"]),
        "",
        "## 3. Findings and required actions",
    ]
    for i, r in enumerate(insights, 1):
        lines += [
            f"### {i}. [{r['type']}] {r['title']}",
            f"- **Fact:** {r['fact']}",
            f"- **Insight:** {r['insight']}",
            f"- **Action:** {r['action']}",
            f"- **Value at stake:** {_fmt_money(r['value'])}",
            "",
        ]
    return "\n".join(lines)


# ======================================================================================
# 5. OPTIONAL GENERATIVE-AI NARRATOR
# ======================================================================================

def llm_narrative(brief_md: str, api_key: str, model: str = "gemini-1.5-flash") -> str:
    """Optionally rewrite the deterministic brief as a CEO-ready narrative.

    Entirely optional. Without a key (or without internet) the platform falls back to
    the rule-based brief above, so the project is never dependent on an external call.
    """
    import requests  # imported lazily so the app runs offline

    prompt = textwrap.dedent(f"""
        You are the revenue strategy analyst for a hotel group. Rewrite the analytics
        brief below as a crisp executive memo of at most 300 words for the CEO.
        Rules: keep every number exactly as given, invent nothing, lead with the money,
        end with the three actions that matter this quarter. Plain prose, no bullet spam.

        ---
        {brief_md}
        ---
    """).strip()

    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": api_key},
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=45,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


# ======================================================================================
# 6. PRESENTATION LAYER
# ======================================================================================

CSS = """
<style>
    .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px;}
    .rs-hero {background: linear-gradient(120deg,#0b2545 0%,#1f6feb 100%);
              padding: 1.4rem 1.6rem; border-radius: 14px; color: #fff; margin-bottom: 1.4rem;}
    .rs-hero h1 {margin: 0; font-size: 1.85rem; letter-spacing: -.02em;}
    .rs-hero p {margin: .35rem 0 0; opacity: .85; font-size: .95rem;}
    .kpi {border: 1px solid rgba(128,128,128,.25); border-radius: 12px;
          padding: .95rem 1.05rem; height: 100%;}
    .kpi .label {font-size: .74rem; text-transform: uppercase; letter-spacing: .08em;
                 opacity: .65;}
    .kpi .value {font-size: 1.65rem; font-weight: 700; line-height: 1.25; margin-top: .15rem;}
    .kpi .delta {font-size: .8rem; opacity: .7;}
    .card {border: 1px solid rgba(128,128,128,.25); border-left-width: 5px;
           border-radius: 10px; padding: 1rem 1.15rem; margin-bottom: .9rem;}
    .card h4 {margin: 0 0 .45rem 0; font-size: 1.02rem;}
    .card p {margin: .28rem 0; font-size: .9rem; line-height: 1.5;}
    .tag {display: inline-block; font-size: .68rem; font-weight: 700; padding: .12rem .5rem;
          border-radius: 20px; letter-spacing: .06em; text-transform: uppercase;}
</style>
"""

TYPE_COLOR = {"Risk": "#ef4444", "Opportunity": "#10b981",
              "Driver": "#f59e0b", "Action": "#1f6feb"}


def kpi_card(col, label: str, value: str, delta: str = ""):
    col.markdown(
        f"<div class='kpi'><div class='label'>{label}</div>"
        f"<div class='value'>{value}</div>"
        f"<div class='delta'>{delta}</div></div>",
        unsafe_allow_html=True,
    )


def insight_card(r: dict):
    c = TYPE_COLOR.get(r["type"], "#1f6feb")
    st.markdown(
        f"<div class='card' style='border-left-color:{c}'>"
        f"<span class='tag' style='background:{c}22;color:{c}'>{r['type']}</span>"
        f"<h4 style='margin-top:.5rem'>{r['title']}</h4>"
        f"<p><b>Fact.</b> {r['fact']}</p>"
        f"<p><b>Insight.</b> {r['insight']}</p>"
        f"<p><b>Action.</b> {r['action']}</p>"
        f"<p style='color:{c};font-weight:600'>Value at stake: {_fmt_money(r['value'])}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


def style_fig(fig, height: int = 380):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        colorway=PALETTE,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.18),
        title_font_size=15,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.18)")
    return fig


# ---------------------------------------------------------------------- Page 1
def page_overview(d: pd.DataFrame, k: dict, source: str):
    st.subheader("Executive overview - what is happening")
    st.caption(f"Data source: {source} | {k['total_bookings']:,} cleaned bookings")

    c = st.columns(4)
    kpi_card(c[0], "Realised revenue", _fmt_money(k["realised_revenue"]),
             f"of {_fmt_money(k['booked_value'])} contracted")
    kpi_card(c[1], "Revenue lost to cancellations", _fmt_money(k["lost_revenue"]),
             f"{k['revenue_leakage_pct']:.1%} leakage")
    kpi_card(c[2], "Cancellation rate", f"{k['cancellation_rate']:.1%}",
             f"{k['cancellations']:,} of {k['total_bookings']:,} bookings")
    kpi_card(c[3], "Average daily rate", f"{CUR}{k['adr']:.2f}",
             f"{k['avg_nights']:.1f} nights avg stay")

    st.write("")
    c = st.columns(4)
    kpi_card(c[0], "Confirmed bookings", f"{k['confirmed_bookings']:,}")
    kpi_card(c[1], "Avg booking value", f"{CUR}{k['avg_booking_value']:,.0f}")
    kpi_card(c[2], "Avg lead time", f"{k['avg_lead_time']:.0f} days")
    kpi_card(c[3], "Repeat guest mix", f"{k['repeat_guest_rate']:.1%}")

    st.divider()
    m = monthly_frame(d)
    left, right = st.columns([3, 2])

    with left:
        fig = go.Figure()
        fig.add_bar(x=m["label"], y=m["realised_revenue"], name="Realised revenue",
                    marker_color=PALETTE[0])
        fig.add_bar(x=m["label"], y=m["lost_revenue"], name="Lost to cancellation",
                    marker_color=PALETTE[3])
        fig.update_layout(barmode="stack", title="Monthly revenue: realised vs destroyed")
        fig.update_xaxes(categoryorder="array", categoryarray=list(m["label"]))
        st.plotly_chart(style_fig(fig, 400), use_container_width=True)

    with right:
        fig = px.line(m, x="label", y="cancellation_rate", markers=True,
                      title="Cancellation rate trend")
        fig.add_hline(y=float(m["cancellation_rate"].mean()), line_dash="dot",
                      annotation_text="period average")
        fig.update_yaxes(tickformat=".0%", title="")
        fig.update_xaxes(title="", categoryorder="array", categoryarray=list(m["label"]))
        st.plotly_chart(style_fig(fig, 400), use_container_width=True)

    left, right = st.columns(2)
    with left:
        h = segment_frame(d, "hotel", min_share=0)
        fig = px.bar(h, x="hotel", y="realised_revenue", color="hotel",
                     title="Realised revenue by property", text_auto=".2s")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)
    with right:
        lt = (d.groupby("lead_time_bucket", observed=True)["is_canceled"]
                .mean().reset_index())
        fig = px.bar(lt, x="lead_time_bucket", y="is_canceled",
                     title="Cancellation rate by booking lead time",
                     color="is_canceled", color_continuous_scale="Reds", text_auto=".1%")
        fig.update_yaxes(tickformat=".0%", title="")
        fig.update_xaxes(title="")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)


# ---------------------------------------------------------------------- Page 2
def page_segments(d: pd.DataFrame):
    st.subheader("Revenue & segment lab - where the money comes from")
    dim = st.selectbox(
        "Analyse by dimension",
        [c for c in ["market_segment", "distribution_channel", "customer_type",
                     "deposit_type", "country", "reserved_room_type", "party_type",
                     "arrival_date_month"] if c in d.columns],
        format_func=lambda c: FRIENDLY_NAMES.get(c, c.replace("_", " ").title()),
    )
    g = segment_frame(d, dim).head(15)

    fig = px.scatter(
        g, x="cancellation_rate", y="realised_revenue", size="bookings",
        color=dim, text=dim, size_max=55,
        title="Value vs risk map - big bubbles top-left are your healthy engines",
    )
    fig.update_traces(textposition="top center")
    fig.update_xaxes(tickformat=".0%", title="Cancellation rate")
    fig.update_yaxes(title=f"Realised revenue ({CURRENCY})")
    fig.update_layout(showlegend=False)
    st.plotly_chart(style_fig(fig, 460), use_container_width=True)

    left, right = st.columns(2)
    with left:
        fig = px.bar(g.sort_values("realised_revenue"), y=dim, x="realised_revenue",
                     orientation="h", title="Realised revenue contribution", text_auto=".2s")
        st.plotly_chart(style_fig(fig, 400), use_container_width=True)
    with right:
        fig = px.bar(g.sort_values("lost_revenue"), y=dim, x="lost_revenue",
                     orientation="h", title="Revenue destroyed by cancellations",
                     text_auto=".2s", color_discrete_sequence=[PALETTE[3]])
        st.plotly_chart(style_fig(fig, 400), use_container_width=True)

    st.markdown("##### Segment scorecard")
    show = g.copy()
    show["cancellation_rate"] = show["cancellation_rate"].map(lambda v: f"{v:.1%}")
    show["share_of_bookings"] = show["share_of_bookings"].map(lambda v: f"{v:.1%}")
    for c in ("realised_revenue", "lost_revenue"):
        show[c] = show[c].map(_fmt_money)
    show["adr"] = show["adr"].map(lambda v: f"{CUR}{v:.0f}")
    show["avg_lead_time"] = show["avg_lead_time"].map(lambda v: f"{v:.0f}d")
    st.dataframe(
        show.rename(columns={
            dim: FRIENDLY_NAMES.get(dim, dim), "bookings": "Bookings",
            "cancellations": "Cancellations", "realised_revenue": "Realised revenue",
            "lost_revenue": "Revenue lost", "adr": "ADR",
            "avg_lead_time": "Avg lead time", "cancellation_rate": "Cancel rate",
            "share_of_bookings": "Share of bookings"}),
        use_container_width=True, hide_index=True,
    )

    st.divider()
    st.markdown("##### Seasonality heatmap - cancellation rate by month and segment")
    if "market_segment" in d.columns:
        piv = (d.pivot_table(index="market_segment", columns="arrival_date_month",
                             values="is_canceled", aggfunc="mean")
                 .reindex(columns=[m for m in MONTH_ORDER
                                   if m in d["arrival_date_month"].unique()]))
        fig = px.imshow(piv, color_continuous_scale="RdYlGn_r", aspect="auto",
                        labels=dict(color="Cancel rate"), text_auto=".0%")
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)


# ---------------------------------------------------------------------- Page 3
def page_risk(d: pd.DataFrame, art: dict):
    st.subheader("Risk engine - what will go wrong, and how much it costs")
    m = art["metrics"]
    scored = art["scored"]

    c = st.columns(5)
    kpi_card(c[0], "ROC-AUC", f"{m['roc_auc']:.3f}", "discrimination power")
    kpi_card(c[1], "Recall (cancellations caught)", f"{m['recall']:.1%}")
    kpi_card(c[2], "Precision", f"{m['precision']:.1%}")
    kpi_card(c[3], "Accuracy", f"{m['accuracy']:.1%}",
             f"vs {max(m['baseline_rate'], 1 - m['baseline_rate']):.1%} naive baseline")
    kpi_card(c[4], "Revenue at risk",
             _fmt_money(float(scored["revenue_at_risk"].sum())),
             "probability-weighted")

    st.divider()
    left, right = st.columns([2, 3])

    with left:
        roc = art["roc"]
        fig = go.Figure()
        fig.add_scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="Model",
                        line=dict(width=3, color=PALETTE[0]))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                        line=dict(dash="dot", color="grey"))
        fig.update_layout(title=f"ROC curve (AUC = {m['roc_auc']:.3f})")
        fig.update_xaxes(title="False positive rate")
        fig.update_yaxes(title="True positive rate")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)

        cm = np.array(m["confusion"])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Predicted: keeps", "Predicted: cancels"],
                        y=["Actual: kept", "Actual: cancelled"],
                        title="Confusion matrix (hold-out set)")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, 330), use_container_width=True)

    with right:
        imp = art["importances"].head(12).sort_values("importance_pct")
        fig = px.bar(imp, x="importance_pct", y="driver", orientation="h",
                     title="What actually drives a cancellation (model importance)",
                     text_auto=".1%", color_discrete_sequence=[PALETTE[1]])
        fig.update_xaxes(tickformat=".0%", title="")
        fig.update_yaxes(title="")
        st.plotly_chart(style_fig(fig, 470), use_container_width=True)

        band = (scored.groupby("risk_band", observed=True)
                      .agg(bookings=("risk_score", "size"),
                           actual_rate=("is_canceled", "mean"),
                           value=("booking_value", "sum")).reset_index())
        fig = px.bar(band, x="risk_band", y="bookings", color="actual_rate",
                     color_continuous_scale="Reds", text_auto=".2s",
                     title="Risk bands: does the score separate reality?")
        fig.update_layout(coloraxis_colorbar_title="Actual<br>cancel rate")
        st.plotly_chart(style_fig(fig, 240), use_container_width=True)

    st.divider()
    st.markdown("##### Daily save-desk queue - highest value bookings most likely to cancel")
    thr = st.slider("Risk threshold", 0.50, 0.95, 0.70, 0.05,
                    help="Only bookings above this cancellation probability enter the queue.")
    queue_cols = [c for c in ["hotel", "market_segment", "deposit_type", "customer_type",
                              "country", "lead_time", "total_nights", "adr",
                              "booking_value", "risk_score", "revenue_at_risk"]
                  if c in scored.columns]
    queue = (scored[scored["risk_score"] >= thr]
             .sort_values("revenue_at_risk", ascending=False)[queue_cols].head(250))

    c1, c2, c3 = st.columns(3)
    kpi_card(c1, "Bookings in queue", f"{len(scored[scored['risk_score'] >= thr]):,}")
    kpi_card(c2, "Contracted value in queue",
             _fmt_money(float(scored.loc[scored['risk_score'] >= thr, 'booking_value'].sum())))
    kpi_card(c3, f"Recoverable at {DEFAULT_INTERVENTION_SUCCESS:.0%} save rate",
             _fmt_money(float(scored.loc[scored['risk_score'] >= thr, 'revenue_at_risk'].sum())
                        * DEFAULT_INTERVENTION_SUCCESS))
    st.write("")
    st.dataframe(queue.style.format({"adr": "{:.0f}", "booking_value": "{:,.0f}",
                                     "risk_score": "{:.1%}", "revenue_at_risk": "{:,.0f}"}),
                 use_container_width=True, hide_index=True)
    st.download_button("Download save-desk queue (CSV)",
                       queue.to_csv(index=False).encode(),
                       file_name="save_desk_queue.csv", mime="text/csv")


# ---------------------------------------------------------------------- Page 4
def page_brief(d: pd.DataFrame, art: dict, insights: list[dict], source: str):
    st.subheader("Decision brief - what management should do on Monday")
    k = compute_kpis(d)
    total_value = sum(r["value"] for r in insights if r["type"] in ("Action", "Opportunity"))

    c = st.columns(3)
    kpi_card(c[0], "Revenue leakage today", _fmt_money(k["lost_revenue"]),
             f"{k['revenue_leakage_pct']:.1%} of contracted value")
    kpi_card(c[1], "Revenue at risk (forward)",
             _fmt_money(float(art["scored"]["revenue_at_risk"].sum())))
    kpi_card(c[2], "Identified recoverable upside", _fmt_money(total_value),
             "across the actions below")

    st.divider()
    tabs = st.tabs(["Findings & actions", "Action register", "Generative-AI memo"])

    with tabs[0]:
        for r in insights:
            insight_card(r)

    with tabs[1]:
        table = build_action_table(insights)
        st.dataframe(table, use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        c1.download_button("Download action register (CSV)",
                           table.to_csv(index=False).encode(),
                           file_name="action_register.csv", mime="text/csv")
        c2.download_button("Download full brief (Markdown)",
                           build_brief_markdown(d, art, insights, source).encode(),
                           file_name="decision_brief.md", mime="text/markdown")

    with tabs[2]:
        st.caption(
            "The brief above is produced deterministically from the data, so the platform "
            "never depends on an external model. Optionally, a generative model can restate "
            "it as a CEO memo - paste a Google Gemini API key to enable it."
        )
        key = st.text_input("Gemini API key (optional)", type="password")
        if st.button("Generate executive memo", disabled=not key):
            with st.spinner("Asking the model to write the memo..."):
                try:
                    st.markdown(llm_narrative(build_brief_markdown(d, art, insights, source), key))
                except Exception as exc:  # noqa: BLE001 - surface any failure gracefully
                    st.error(f"Generative narration unavailable ({exc}). "
                             "The deterministic brief on the first tab remains valid.")


# ---------------------------------------------------------------------- Page 5
def page_data(raw: pd.DataFrame, d: pd.DataFrame, log: list[str]):
    st.subheader("Data quality & methodology")
    c = st.columns(4)
    kpi_card(c[0], "Raw rows ingested", f"{len(raw):,}")
    kpi_card(c[1], "Clean rows retained", f"{len(d):,}",
             f"{100 * len(d) / max(len(raw), 1):.1f}% retention")
    kpi_card(c[2], "Raw columns", f"{raw.shape[1]}")
    kpi_card(c[3], "Engineered features", f"{d.shape[1] - raw.shape[1]}")

    st.markdown("##### Cleaning audit trail")
    for line in log:
        st.markdown(f"- {line}")

    st.markdown("##### Missing values in the raw feed")
    miss = (raw.isna().sum().sort_values(ascending=False).head(12)
               .rename("missing").reset_index(name="missing"))
    miss["% of rows"] = (miss["missing"] / len(raw) * 100).round(2)
    st.dataframe(miss, use_container_width=True, hide_index=True)

    st.markdown("##### Cleaned sample")
    st.dataframe(d.head(200), use_container_width=True, hide_index=True)

    st.markdown("##### Method")
    st.markdown(textwrap.dedent(f"""
        1. **Ingest** raw booking records (CSV).
        2. **Clean** - drop leakage/identifier columns, de-duplicate, impute, remove
           impossible ADR values and ghost bookings with zero guests or zero nights.
        3. **Engineer** - booking value (ADR x nights), realised vs lost revenue,
           lead-time buckets, party type, engagement flags, arrival period.
        4. **Describe** - KPI layer and trend analysis (what is happening).
        5. **Explain** - segment value-vs-risk mapping (why it is happening).
        6. **Predict** - Random Forest on {len(MODEL_NUMERIC) + len(MODEL_CATEGORICAL)}
           features, stratified 80/20 split, class-balanced, evaluated on ROC-AUC,
           precision, recall and F1.
        7. **Decide** - rule-based insight engine converts every statistic into
           fact -> insight -> action with a Euro value attached, optionally narrated by a
           generative model.

        **Leakage control.** `reservation_status` and `reservation_status_date` encode the
        outcome directly and are dropped before modelling; the model only sees information
        available at the moment of booking.
    """))


# ======================================================================================
# 7. MAIN
# ======================================================================================

@st.cache_data(show_spinner=False)
def _prepare(file_bytes, name_hint: str):
    raw, source = load_raw(io.BytesIO(file_bytes) if file_bytes else None)
    if file_bytes:
        source = f"Uploaded CSV ({name_hint})"
    clean, log = clean_data(raw)
    return raw, clean, engineer_features(clean), log, source


@st.cache_resource(show_spinner=False)
def _model(_d: pd.DataFrame, cache_key: str):
    return train_risk_model(_d)


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="\U0001F6E1\uFE0F",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        f"<div class='rs-hero'><h1>{APP_TITLE}</h1><p>{APP_SUB} "
        "&nbsp;|&nbsp; Data &rarr; Information &rarr; Insight &rarr; Decision &rarr; Action</p></div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(f"### {APP_TITLE}")
        up = st.file_uploader("Upload hotel_bookings.csv", type=["csv"])
        st.caption("No file? The platform runs on a built-in simulator so nothing is broken.")

        file_bytes = up.getvalue() if up is not None else None
        raw, clean, d, log, source = _prepare(file_bytes, up.name if up else "")

        st.divider()
        st.markdown("**Filters**")
        hotels = sorted(d["hotel"].dropna().unique())
        pick_hotel = st.multiselect("Property", hotels, default=hotels)
        years = sorted(d["arrival_date_year"].dropna().unique())
        pick_year = st.multiselect("Arrival year", years, default=years)
        segs = sorted(d["market_segment"].dropna().unique())
        pick_seg = st.multiselect("Market segment", segs, default=segs)

        f = d[d["hotel"].isin(pick_hotel)
              & d["arrival_date_year"].isin(pick_year)
              & d["market_segment"].isin(pick_seg)]

        st.divider()
        st.caption(f"Source: {source}")
        st.caption(f"{len(f):,} bookings in current view")

    if len(f) < 500:
        st.warning("Fewer than 500 bookings in the current selection - widen the filters "
                   "for statistically meaningful results.")
        if f.empty:
            st.stop()

    with st.spinner("Training the cancellation risk model..."):
        art = _model(f, f"{len(f)}-{int(f['is_canceled'].sum())}-{source}")
    insights = generate_insights(f, art)
    k = compute_kpis(f)

    t1, t2, t3, t4, t5 = st.tabs([
        "Executive overview", "Revenue & segments", "Risk engine (ML)",
        "Decision brief", "Data & method",
    ])
    with t1:
        page_overview(f, k, source)
    with t2:
        page_segments(f)
    with t3:
        page_risk(f, art)
    with t4:
        page_brief(f, art, insights, source)
    with t5:
        page_data(raw, f, log)


if __name__ == "__main__":
    main()


# ============================================================
# REVENUE SHIELD — SAFE ENHANCEMENT LAYER
# Added without deleting/replacing original project code.
# ============================================================
def _rs_safe_float(value, default=0.0):
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default

def _rs_money(value):
    return f"€{_rs_safe_float(value):,.2f}"

def _rs_find_col(frame, names):
    lower = {str(c).strip().lower(): c for c in frame.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None

def _rs_validate_dataframe(frame):
    """Return a validation report without mutating the original dataframe."""
    required = ["is_canceled"]
    missing = [c for c in required if c not in frame.columns]
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "missing_required": missing,
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_cells": int(frame.isna().sum().sum()),
    }

def _rs_add_data_quality_panel(frame):
    """Optional, non-destructive data quality panel."""
    try:
        report = _rs_validate_dataframe(frame)
        with st.expander("🔎 Data Quality & Validation", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{report['rows']:,}")
            c2.metric("Columns", f"{report['columns']:,}")
            c3.metric("Duplicate rows", f"{report['duplicate_rows']:,}")
            c4.metric("Missing cells", f"{report['missing_cells']:,}")
            if report["missing_required"]:
                st.error("Missing required column(s): " + ", ".join(report["missing_required"]))
            else:
                st.success("Required target column is present. Original pipeline remains unchanged.")
    except Exception:
        # Never allow an enhancement to break the original app.
        pass

def _rs_add_download_button(frame, filename="revenue_shield_current_data.csv"):
    """Download the currently loaded dataset for reproducibility."""
    try:
        csv_bytes = frame.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Loaded Dataset",
            data=csv_bytes,
            file_name=filename,
            mime="text/csv",
            use_container_width=False,
        )
    except Exception:
        pass

def _rs_add_risk_export(risk_frame, filename="revenue_shield_risk_queue.csv"):
    """Export the risk/intervention table when one is available."""
    try:
        if risk_frame is not None and hasattr(risk_frame, "to_csv"):
            st.download_button(
                "⬇️ Export Risk Queue",
                data=risk_frame.to_csv(index=False).encode("utf-8"),
                file_name=filename,
                mime="text/csv",
                use_container_width=False,
            )
    except Exception:
        pass

def _rs_add_methodology_guard():
    """Clarify simulator vs real-data mode and prevent misleading presentation."""
    try:
        st.info(
            "ℹ️ **Evidence mode:** Results shown by the app depend on the loaded dataset. "
            "If the built-in simulator is active, treat figures as demonstration values; "
            "load the real Kaggle CSV for dataset-grounded findings."
        )
    except Exception:
        pass

# The following marker is intentionally kept at the end so the original
# Revenue Shield implementation above remains intact.
# ============================================================
# END SAFE ENHANCEMENT LAYER
# ============================================================
