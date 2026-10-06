import os
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

from src.data_processor import preprocess_dataset
from src.ml_models import train_all_models, perform_pca_analysis, perform_kmeans_clustering, predict_all_models
from src.insights import generate_automated_insights

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "best_bus_data_science_secret_key"

ACTIVE_SESSION = {
    "df": None,
    "pipelines": None,
    "analysis_cache": None
}

def analyze_df(df):
    """
    Execute end-to-end data processing, EDA statistics, ML modeling, PCA, clustering, and insights.
    """
    clean_df, summary_info = preprocess_dataset(df)

    if clean_df is None:
        return {"success": False, "error": summary_info["error"]}

    num_cols = [
        "Crowding_Level", "Punctuality_Rating", "Cleanliness_Rating",
        "Staff_Behavior_Rating", "Safety_Rating", "Fare_Satisfaction",
        "Satisfaction_Score", "Waiting_Time_Num", "Travel_Time_Num"
    ]
    
    desc_stats = []
    for col in num_cols:
        if col in clean_df.columns:
            series = clean_df[col]
            mode_val = float(series.mode()[0]) if not series.mode().empty else float(series.mean())
            desc_stats.append({
                "variable": col.replace("_Num", "").replace("_", " "),
                "mean": round(float(series.mean()), 2),
                "median": round(float(series.median()), 2),
                "mode": round(mode_val, 2),
                "std": round(float(series.std()), 2),
                "min": round(float(series.min()), 2),
                "max": round(float(series.max()), 2),
                "p25": round(float(series.quantile(0.25)), 2),
                "p50": round(float(series.quantile(0.50)), 2),
                "p75": round(float(series.quantile(0.75)), 2)
            })

    def get_counts(col_name):
        if col_name not in clean_df.columns:
            return {"labels": [], "counts": []}
        vc = clean_df[col_name].value_counts()
        return {"labels": [str(x) for x in vc.index.tolist()], "counts": [int(x) for x in vc.values.tolist()]}

    # Scores 1-5 distribution
    score_vc = clean_df["Satisfaction_Score"].value_counts().sort_index()
    score_dist = {
        "labels": [f"Score {int(k)}" for k in score_vc.index],
        "counts": [int(v) for v in score_vc.values]
    }

    crowd_vc = clean_df["Crowding_Level"].value_counts().sort_index()
    crowd_dist = {
        "labels": [f"Rating {int(k)}" for k in crowd_vc.index],
        "counts": [int(v) for v in crowd_vc.values]
    }

    eda_data = {
        "age_dist": get_counts("Age"),
        "gender_dist": get_counts("Gender"),
        "occupation_dist": get_counts("Occupation"),
        "usage_freq_dist": get_counts("BEST_Usage_Frequency"),
        "travel_purpose_dist": get_counts("Travel_Purpose"),
        "travel_time_dist": get_counts("Travel_Time"),
        "waiting_time_dist": get_counts("Waiting_Time"),
        "delay_freq_dist": get_counts("Delay_Frequency"),
        "crowding_dist": crowd_dist,
        "score_dist": score_dist,
        "satisfaction_dist": {
            "labels": ["Not Satisfied (1-3)", "Satisfied (4-5)"],
            "counts": [
                summary_info["not_satisfied_count"],
                summary_info["satisfied_count"]
            ]
        },
        "rating_averages": {
            "categories": ["Crowding", "Punctuality", "Cleanliness", "Staff Behavior", "Safety", "Fare", "Satisfaction Score"],
            "values": [
                round(float(clean_df["Crowding_Level"].mean()), 2),
                round(float(clean_df["Punctuality_Rating"].mean()), 2),
                round(float(clean_df["Cleanliness_Rating"].mean()), 2),
                round(float(clean_df["Staff_Behavior_Rating"].mean()), 2),
                round(float(clean_df["Safety_Rating"].mean()), 2),
                round(float(clean_df["Fare_Satisfaction"].mean()), 2),
                round(float(clean_df["Satisfaction_Score"].mean()), 2)
            ]
        }
    }

    corr_cols = [
        "Crowding_Level", "Punctuality_Rating", "Cleanliness_Rating",
        "Staff_Behavior_Rating", "Safety_Rating", "Fare_Satisfaction",
        "Satisfaction_Score", "Waiting_Time_Num"
    ]
    corr_matrix = clean_df[corr_cols].corr().round(2)
    corr_data = {
        "labels": [c.replace("_Num", "").replace("_", " ") for c in corr_cols],
        "values": corr_matrix.values.tolist()
    }

    pipelines, ml_summary = train_all_models(clean_df)
    pca_data = perform_pca_analysis(clean_df)
    kmeans_data = perform_kmeans_clustering(clean_df, k=3)

    ACTIVE_SESSION["df"] = clean_df
    ACTIVE_SESSION["pipelines"] = pipelines

    insights = generate_automated_insights(clean_df, ml_summary.get("logistic_regression", {}), kmeans_data)

    sample_preview = clean_df[
        ["Age", "Gender", "Occupation", "BEST_Usage_Frequency", "Travel_Purpose", "Waiting_Time", "Crowding_Level", "Punctuality_Rating", "Satisfaction_Score", "Satisfaction_Text"]
    ].head(10).to_dict(orient="records")

    analysis_result = {
        "success": True,
        "summary": summary_info,
        "sample_preview": sample_preview,
        "descriptive_stats": desc_stats,
        "eda": eda_data,
        "correlation": corr_data,
        "ml_models": ml_summary,
        "pca": pca_data,
        "kmeans": kmeans_data,
        "insights": insights
    }

    ACTIVE_SESSION["analysis_cache"] = analysis_result
    return analysis_result

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/load-sample", methods=["GET"])
def load_sample():
    primary_path = os.path.join(os.path.dirname(__file__), "data", "BEST_Bus_Survey_100_Augmented_Responses.csv")
    fallback_path = os.path.join(os.path.dirname(__file__), "data", "sample_best_bus_dataset.csv")

    target_path = primary_path if os.path.exists(primary_path) else fallback_path

    if not os.path.exists(target_path):
        return jsonify({"success": False, "error": "Dataset CSV file not found on server."}), 404
    
    df = pd.read_csv(target_path)
    result = analyze_df(df)
    result["dataset_filename"] = os.path.basename(target_path)
    return jsonify(result)

@app.route("/api/upload", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No CSV file uploaded in request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    try:
        df = pd.read_csv(file)
        result = analyze_df(df)
        result["dataset_filename"] = file.filename
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to parse CSV file: {str(e)}"}), 500

@app.route("/api/predict", methods=["POST"])
def predict():
    if ACTIVE_SESSION["pipelines"] is None:
        return jsonify({"success": False, "error": "No trained ML models available. Please upload a dataset first."}), 400

    data = request.json
    if not data:
        return jsonify({"success": False, "error": "No input passenger parameters provided."}), 400

    try:
        prediction_result = predict_all_models(ACTIVE_SESSION["pipelines"], data)
        return jsonify({"success": True, "result": prediction_result})
    except Exception as e:
        return jsonify({"success": False, "error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
