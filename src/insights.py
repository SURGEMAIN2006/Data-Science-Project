import pandas as pd
import numpy as np

def generate_automated_insights(df, ml_metrics=None, cluster_data=None):
    """
    Generate dynamic bulleted insights directly calculated from the uploaded dataset.
    """
    insights = []
    total = len(df)

    # 1. Respondent Demographics Insight
    if "Age" in df.columns:
        top_age = df["Age"].mode()[0]
        top_age_pct = round((df["Age"] == top_age).mean() * 100, 1)
        insights.append({
            "category": "Demographics",
            "icon": "user",
            "title": "Primary Passenger Demographics",
            "detail": f"The majority of survey respondents ({top_age_pct}%, n={int((df['Age'] == top_age).sum())}) belong to the '{top_age}' age bracket."
        })

    # 2. Travel Behavior Insight
    if "BEST_Usage_Frequency" in df.columns:
        top_freq = df["BEST_Usage_Frequency"].mode()[0]
        top_freq_pct = round((df["BEST_Usage_Frequency"] == top_freq).mean() * 100, 1)
        insights.append({
            "category": "Travel Habits",
            "icon": "bus",
            "title": "Usage Frequency Pattern",
            "detail": f"'{top_freq}' is the most dominant commuter travel frequency, representing {top_freq_pct}% of total surveyed BEST bus riders."
        })

    # 3. Overall Satisfaction & Key Driver Insight
    if "Satisfaction_Score" in df.columns and "Satisfaction_Label" in df.columns:
        avg_sat = round(df["Satisfaction_Score"].mean(), 2)
        sat_pct = round((df["Satisfaction_Label"] == 1).mean() * 100, 1)
        insights.append({
            "category": "Satisfaction Breakdown",
            "icon": "smile",
            "title": "Overall Passenger Satisfaction Rate",
            "detail": f"The overall average passenger satisfaction score across all respondents is {avg_sat} / 5.0, with {sat_pct}% of commuters reporting overall satisfaction (rating score 4 or 5)."
        })

    # 4. Service Quality Rating Impact (Punctuality vs Satisfaction)
    if "Punctuality_Rating" in df.columns and "Satisfaction_Label" in df.columns:
        high_punct_sat = round((df[df["Punctuality_Rating"] >= 4]["Satisfaction_Label"] == 1).mean() * 100, 1)
        low_punct_sat = round((df[df["Punctuality_Rating"] <= 2]["Satisfaction_Label"] == 1).mean() * 100, 1)
        insights.append({
            "category": "Service Performance",
            "icon": "clock",
            "title": "Punctuality & Satisfaction Relationship",
            "detail": f"Passenders who rate punctuality 4 or 5 have a {high_punct_sat}% satisfaction rate, whereas those rating punctuality 1 or 2 exhibit a {low_punct_sat}% satisfaction rate."
        })

    # 5. Waiting Time Impact
    if "Waiting_Time" in df.columns and "Satisfaction_Score" in df.columns:
        wait_grouped = df.groupby("Waiting_Time")["Satisfaction_Score"].mean().to_dict()
        if wait_grouped:
            sorted_wait = sorted(wait_grouped.items(), key=lambda x: x[1], reverse=True)
            highest_wait_cat, highest_score = sorted_wait[0]
            lowest_wait_cat, lowest_score = sorted_wait[-1]
            insights.append({
                "category": "Waiting Experience",
                "icon": "hourglass",
                "title": "Impact of Bus Waiting Time",
                "detail": f"Passengers with '{highest_wait_cat}' experience the highest average satisfaction ({round(highest_score, 2)}/5.0), compared to {round(lowest_score, 2)}/5.0 for passengers in the '{lowest_wait_cat}' category."
            })

    # 6. ML Model Insight
    if ml_metrics and "accuracy" in ml_metrics:
        acc = ml_metrics["accuracy"]
        f1 = ml_metrics["f1_score"]
        top_feat = ml_metrics.get("top_features", [])
        top_name = top_feat[0]["feature"] if top_feat else "Punctuality & Safety"
        insights.append({
            "category": "Machine Learning",
            "icon": "cpu",
            "title": "Logistic Regression ML Classification",
            "detail": f"The trained Logistic Regression classifier achieved an evaluation Accuracy of {acc}% (F1 Score: {f1}%). The model identified '{top_name}' as a key influencing predictor."
        })

    # 7. Unsupervised Clustering Insight
    if cluster_data and "cluster_profiles" in cluster_data:
        profiles = cluster_data["cluster_profiles"]
        if profiles:
            best_cluster = max(profiles, key=lambda x: x["avg_satisfaction"])
            worst_cluster = min(profiles, key=lambda x: x["avg_satisfaction"])
            insights.append({
                "category": "Passenger Segmentation",
                "icon": "users",
                "title": "K-Means Behavioral Segments",
                "detail": f"K-Means identified '{best_cluster['cluster_name']}' as the highest-satisfaction segment ({best_cluster['avg_satisfaction']}/5.0 avg score), while '{worst_cluster['cluster_name']}' represents the most dissatisfied segment needing service intervention."
            })

    return insights
