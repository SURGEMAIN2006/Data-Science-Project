import pandas as pd
import numpy as np

# Canonical variable definitions
CANONICAL_VARS = [
    "Age",
    "Gender",
    "Occupation",
    "BEST_Usage_Frequency",
    "Travel_Purpose",
    "Travel_Time",
    "Waiting_Time",
    "Crowding_Level",
    "Punctuality_Rating",
    "Cleanliness_Rating",
    "Staff_Behavior_Rating",
    "Safety_Rating",
    "Fare_Satisfaction",
    "Delay_Frequency",
    "Satisfaction_Score"
]

# Keyword rules for intelligent column mapping
KEYWORD_MAPPING = {
    "Age": ["age", "how old"],
    "Gender": ["gender", "sex"],
    "Occupation": ["occupation", "work", "profession", "job"],
    "BEST_Usage_Frequency": ["how often", "usage_frequency", "frequency of travel", "how frequently do you travel"],
    "Travel_Purpose": ["purpose", "travel_purpose", "reason for travel", "why do you travel", "main purpose"],
    "Travel_Time": ["travel_time", "journey time", "how long", "duration"],
    "Waiting_Time": ["waiting_time", "wait time", "how long is your usual waiting", "usual waiting"],
    "Crowding_Level": ["crowd", "crowding", "how crowded"],
    "Punctuality_Rating": ["punctual", "punctuality", "on time"],
    "Cleanliness_Rating": ["clean", "cleanliness", "hygiene"],
    "Staff_Behavior_Rating": ["staff", "driver", "conductor", "behavior", "staff_behavior"],
    "Safety_Rating": ["safe", "safety", "security"],
    "Fare_Satisfaction": ["fare", "ticket price", "cost", "fare_satisfaction"],
    "Delay_Frequency": ["delay", "delays", "late", "delay_frequency"],
    "Satisfaction_Score": ["overall", "satisfaction_score", "how satisfied are you with best", "overall satisfaction", "overall, how satisfied"]
}

def map_columns(df):
    """
    Intelligently map uploaded DataFrame columns to canonical variables.
    """
    mapped_df = pd.DataFrame()
    mapping_report = {}
    missing_vars = []

    existing_cols = list(df.columns)
    used_cols = set()

    for canonical in CANONICAL_VARS:
        matched_col = None
        for col in existing_cols:
            if col in used_cols:
                continue
            if col.strip().lower() == canonical.lower():
                matched_col = col
                break
        
        if not matched_col:
            keywords = KEYWORD_MAPPING.get(canonical, [])
            for col in existing_cols:
                if col in used_cols:
                    continue
                col_lower = col.strip().lower()
                if "timestamp" in col_lower:
                    continue
                if any(kw in col_lower for kw in keywords):
                    matched_col = col
                    break
        
        if matched_col:
            mapped_df[canonical] = df[matched_col]
            used_cols.add(matched_col)
            mapping_report[canonical] = matched_col
        else:
            missing_vars.append(canonical)

    return mapped_df, mapping_report, missing_vars

def preprocess_dataset(df):
    """
    Clean, impute, map ordinal categories, and create target binary column.
    """
    raw_rows = len(df)
    
    clean_df, mapping_report, missing_vars = map_columns(df)
    
    if missing_vars:
        return None, {
            "error": f"Missing required survey variables: {', '.join(missing_vars)}",
            "missing_vars": missing_vars,
            "mapping_report": mapping_report
        }

    duplicates_count = int(clean_df.duplicated().sum())
    clean_df = clean_df.drop_duplicates()

    missing_summary = clean_df.isnull().sum().to_dict()
    total_missing = sum(missing_summary.values())

    rating_cols = [
        "Crowding_Level", "Punctuality_Rating", "Cleanliness_Rating",
        "Staff_Behavior_Rating", "Safety_Rating", "Fare_Satisfaction",
        "Satisfaction_Score"
    ]

    for col in rating_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
        if clean_df[col].isnull().any():
            clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    freq_map = {
        "Daily": 5,
        "4–6 times a week": 4, "4-6 times a week": 4,
        "2–3 times a week": 3, "2-3 times a week": 3,
        "Once a week": 2,
        "Occasionally": 1
    }
    clean_df["BEST_Usage_Frequency_Num"] = clean_df["BEST_Usage_Frequency"].astype(str).str.strip().map(freq_map).fillna(3)

    time_map = {
        "Less than 15 minutes": 10, "Under 15 mins": 10, "<15 mins": 10,
        "15–30 minutes": 22.5, "15-30 minutes": 22.5, "15–30 mins": 22.5, "15-30 mins": 22.5,
        "31–45 minutes": 38, "31-45 minutes": 38,
        "46–60 minutes": 53, "46-60 minutes": 53, "30–60 mins": 45, "30-60 mins": 45,
        "More than 60 minutes": 75, "Over 60 mins": 75, ">60 mins": 75
    }
    clean_df["Travel_Time_Num"] = clean_df["Travel_Time"].astype(str).str.strip().map(time_map).fillna(30)

    wait_map = {
        "Less than 5 minutes": 3, "Under 5 mins": 3, "<5 mins": 3,
        "5–10 minutes": 7.5, "5-10 minutes": 7.5, "5–15 mins": 10, "5-15 mins": 10,
        "11–15 minutes": 13, "11-15 minutes": 13,
        "16–30 minutes": 23, "16-30 minutes": 23, "15–30 mins": 22.5, "15-30 mins": 22.5,
        "More than 30 minutes": 40, "Over 30 mins": 40, ">30 mins": 40
    }
    clean_df["Waiting_Time_Num"] = clean_df["Waiting_Time"].astype(str).str.strip().map(wait_map).fillna(15)

    delay_map = {
        "Never": 1,
        "Rarely": 2,
        "Sometimes": 3,
        "Often": 4, "Frequently": 4,
        "Very Often": 5, "Always": 5
    }
    clean_df["Delay_Frequency_Num"] = clean_df["Delay_Frequency"].astype(str).str.strip().map(delay_map).fillna(3)

    clean_df["Satisfaction_Label"] = clean_df["Satisfaction_Score"].apply(lambda x: 1 if x >= 4 else 0)
    clean_df["Satisfaction_Text"] = clean_df["Satisfaction_Label"].apply(lambda x: "Satisfied" if x == 1 else "Not Satisfied")

    # Calculate Box Plot 5-number summary stats for rating variables
    box_plots = {}
    box_cols = ["Punctuality_Rating", "Cleanliness_Rating", "Staff_Behavior_Rating", "Safety_Rating", "Fare_Satisfaction", "Crowding_Level"]
    for col in box_cols:
        series = clean_df[col]
        box_plots[col.replace("_Rating", "").replace("_", " ")] = {
            "min": float(series.min()),
            "q1": float(series.quantile(0.25)),
            "median": float(series.median()),
            "q3": float(series.quantile(0.75)),
            "max": float(series.max())
        }

    processed_rows = len(clean_df)

    summary_info = {
        "raw_rows": raw_rows,
        "processed_rows": processed_rows,
        "cols_count": len(clean_df.columns),
        "duplicates_removed": duplicates_count,
        "total_missing_imputed": total_missing,
        "mapping_report": mapping_report,
        "missing_vars": missing_vars,
        "box_plots": box_plots,
        "satisfied_count": int((clean_df["Satisfaction_Label"] == 1).sum()),
        "not_satisfied_count": int((clean_df["Satisfaction_Label"] == 0).sum()),
        "satisfaction_pct": round(((clean_df["Satisfaction_Label"] == 1).mean() * 100), 1),
        "avg_waiting_time": round(clean_df["Waiting_Time_Num"].mean(), 1),
        "avg_satisfaction_score": round(clean_df["Satisfaction_Score"].mean(), 2)
    }

    return clean_df, summary_info
