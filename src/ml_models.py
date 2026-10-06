import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_curve, roc_auc_score
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def train_all_models(df):
    """
    Train Logistic Regression, Decision Tree, Random Forest, and Linear Regression
    with strict train/test split (80/20) and data leakage prevention.
    Generate complete evaluation data including ROC curves, residuals, feature importances, and tree rules.
    """
    # Check target class distribution for classification
    class_counts = df["Satisfaction_Label"].value_counts().to_dict()
    if len(class_counts) < 2:
        return None, {
            "error": "Dataset contains only one satisfaction outcome class. Classification models require both 'Satisfied' (Score 4-5) and 'Not Satisfied' (Score 1-3) responses to train.",
            "class_counts": class_counts
        }

    # Define feature sets
    num_features = [
        "Crowding_Level", "Punctuality_Rating", "Cleanliness_Rating",
        "Staff_Behavior_Rating", "Safety_Rating", "Fare_Satisfaction",
        "Delay_Frequency_Num", "Waiting_Time_Num", "Travel_Time_Num",
        "BEST_Usage_Frequency_Num"
    ]
    
    cat_features = ["Age", "Gender", "Occupation", "Travel_Purpose"]
    all_features = num_features + cat_features

    X = df[all_features]
    y_class = df["Satisfaction_Label"]
    y_score = df["Satisfaction_Score"]

    # 80/20 Train/Test Split (random_state=42)
    test_size = 0.2 if len(df) >= 15 else 0.3
    test_rows = int(np.ceil(len(df) * test_size))
    can_stratify = (
        len(df) >= 20
        and min(class_counts.values()) >= 2
        and test_rows >= len(class_counts)
    )
    X_train, X_test, y_train_class, y_test_class, y_train_score, y_test_score = train_test_split(
        X, y_class, y_score, test_size=test_size, random_state=42, stratify=y_class if can_stratify else None
    )

    # Preprocessor fitted ONLY on training set to avoid data leakage
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_features)
        ]
    )

    # 1. LOGISTIC REGRESSION (Binary Classification)
    log_reg_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000))
    ])
    log_reg_pipeline.fit(X_train, y_train_class)
    
    y_pred_log = log_reg_pipeline.predict(X_test)
    y_prob_log = log_reg_pipeline.predict_proba(X_test)[:, 1] if hasattr(log_reg_pipeline, "predict_proba") else None

    # ROC Curve for Logistic Regression
    log_roc = {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "auc": 0.5}
    if y_prob_log is not None and len(np.unique(y_test_class)) > 1:
        fpr, tpr, _ = roc_curve(y_test_class, y_prob_log)
        auc_val = roc_auc_score(y_test_class, y_prob_log)
        log_roc = {
            "fpr": [round(float(x), 3) for x in fpr],
            "tpr": [round(float(x), 3) for x in tpr],
            "auc": round(float(auc_val), 3)
        }

    log_metrics = {
        "accuracy": round(float(accuracy_score(y_test_class, y_pred_log)) * 100, 2),
        "precision": round(float(precision_score(y_test_class, y_pred_log, zero_division=0)) * 100, 2),
        "recall": round(float(recall_score(y_test_class, y_pred_log, zero_division=0)) * 100, 2),
        "f1_score": round(float(f1_score(y_test_class, y_pred_log, zero_division=0)) * 100, 2),
        "confusion_matrix": confusion_matrix(y_test_class, y_pred_log).tolist(),
        "roc": log_roc,
        "class_dist": {
            "actual": [int((y_test_class == 0).sum()), int((y_test_class == 1).sum())],
            "predicted": [int((y_pred_log == 0).sum()), int((y_pred_log == 1).sum())]
        }
    }

    # Feature Importance for Logistic Regression
    classifier_log = log_reg_pipeline.named_steps["classifier"]
    encoded_cat = log_reg_pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(cat_features)
    feature_names = num_features + list(encoded_cat)
    coefs_log = classifier_log.coef_[0]
    
    log_top_features = pd.DataFrame({
        "feature": feature_names,
        "importance": coefs_log
    }).sort_values(by="importance", key=abs, ascending=False).head(8).to_dict(orient="records")

    # 2. DECISION TREE CLASSIFIER (Binary Classification)
    dt_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42, max_depth=4))
    ])
    dt_pipeline.fit(X_train, y_train_class)
    
    y_pred_dt = dt_pipeline.predict(X_test)
    y_prob_dt = dt_pipeline.predict_proba(X_test)[:, 1] if hasattr(dt_pipeline, "predict_proba") else None

    dt_metrics = {
        "accuracy": round(float(accuracy_score(y_test_class, y_pred_dt)) * 100, 2),
        "precision": round(float(precision_score(y_test_class, y_pred_dt, zero_division=0)) * 100, 2),
        "recall": round(float(recall_score(y_test_class, y_pred_dt, zero_division=0)) * 100, 2),
        "f1_score": round(float(f1_score(y_test_class, y_pred_dt, zero_division=0)) * 100, 2),
        "confusion_matrix": confusion_matrix(y_test_class, y_pred_dt).tolist(),
        "class_dist": {
            "actual": [int((y_test_class == 0).sum()), int((y_test_class == 1).sum())],
            "predicted": [int((y_pred_dt == 0).sum()), int((y_pred_dt == 1).sum())]
        }
    }

    classifier_dt = dt_pipeline.named_steps["classifier"]
    dt_top_features = pd.DataFrame({
        "feature": feature_names,
        "importance": classifier_dt.feature_importances_
    }).sort_values(by="importance", ascending=False).head(8).to_dict(orient="records")

    # Extract actual Decision Tree text rules
    tree_rules_text = export_text(classifier_dt, feature_names=feature_names, max_depth=3)

    # 3. RANDOM FOREST CLASSIFIER (Binary Classification)
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    rf_pipeline.fit(X_train, y_train_class)

    y_pred_rf = rf_pipeline.predict(X_test)
    y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1] if hasattr(rf_pipeline, "predict_proba") else None

    rf_roc = {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "auc": 0.5}
    if y_prob_rf is not None and len(np.unique(y_test_class)) > 1:
        fpr, tpr, _ = roc_curve(y_test_class, y_prob_rf)
        auc_val = roc_auc_score(y_test_class, y_prob_rf)
        rf_roc = {
            "fpr": [round(float(x), 3) for x in fpr],
            "tpr": [round(float(x), 3) for x in tpr],
            "auc": round(float(auc_val), 3)
        }

    rf_metrics = {
        "accuracy": round(float(accuracy_score(y_test_class, y_pred_rf)) * 100, 2),
        "precision": round(float(precision_score(y_test_class, y_pred_rf, zero_division=0)) * 100, 2),
        "recall": round(float(recall_score(y_test_class, y_pred_rf, zero_division=0)) * 100, 2),
        "f1_score": round(float(f1_score(y_test_class, y_pred_rf, zero_division=0)) * 100, 2),
        "confusion_matrix": confusion_matrix(y_test_class, y_pred_rf).tolist(),
        "roc": rf_roc,
        "class_dist": {
            "actual": [int((y_test_class == 0).sum()), int((y_test_class == 1).sum())],
            "predicted": [int((y_pred_rf == 0).sum()), int((y_pred_rf == 1).sum())]
        }
    }

    classifier_rf = rf_pipeline.named_steps["classifier"]
    rf_top_features = pd.DataFrame({
        "feature": feature_names,
        "importance": classifier_rf.feature_importances_
    }).sort_values(by="importance", ascending=False).head(8).to_dict(orient="records")

    # 4. LINEAR REGRESSION (Continuous Regression on Satisfaction_Score)
    lin_reg_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    lin_reg_pipeline.fit(X_train, y_train_score)
    
    y_pred_lin = lin_reg_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test_score, y_pred_lin)
    mse = mean_squared_error(y_test_score, y_pred_lin)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_score, y_pred_lin)

    # Actual vs Predicted scatter points & Residuals
    actual_vs_pred = []
    residuals = []
    for a, p in zip(y_test_score, y_pred_lin):
        act_val = round(float(a), 2)
        pred_val = round(float(p), 2)
        actual_vs_pred.append({"x": act_val, "y": pred_val})
        residuals.append({"x": pred_val, "y": round(float(act_val - pred_val), 2)})

    lin_metrics = {
        "mae": round(float(mae), 3),
        "mse": round(float(mse), 3),
        "rmse": round(float(rmse), 3),
        "r2_score": round(float(r2), 3),
        "actual_vs_pred": actual_vs_pred,
        "residuals": residuals
    }

    # Target Feature Correlations
    num_cols = [
        "Crowding_Level", "Punctuality_Rating", "Cleanliness_Rating",
        "Staff_Behavior_Rating", "Safety_Rating", "Fare_Satisfaction",
        "Delay_Frequency_Num", "Waiting_Time_Num", "Travel_Time_Num",
        "BEST_Usage_Frequency_Num"
    ]
    target_corrs = df[num_cols].corrwith(df["Satisfaction_Score"]).round(3).to_dict()
    target_corrs_clean = [
        {"feature": k.replace("_Num", "").replace("_", " "), "correlation": float(v)}
        for k, v in sorted(target_corrs.items(), key=lambda item: abs(item[1]), reverse=True)
    ]

    # Additional classifiers commonly taught in introductory ML courses.
    # Keep all preprocessing inside each pipeline so the holdout stays unseen.
    extra_estimators = {
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=min(5, len(X_train))),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42)
    }
    extra_models = {}
    extra_pipelines = {}
    for name, estimator in extra_estimators.items():
        model_key = name.lower().replace(" ", "_")
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
        pipeline.fit(X_train, y_train_class)
        extra_pipelines[model_key] = pipeline
        pred = pipeline.predict(X_test)
        extra_models[model_key] = {
            "model": name,
            "accuracy": round(float(accuracy_score(y_test_class, pred)) * 100, 2),
            "precision": round(float(precision_score(y_test_class, pred, zero_division=0)) * 100, 2),
            "recall": round(float(recall_score(y_test_class, pred, zero_division=0)) * 100, 2),
            "f1_score": round(float(f1_score(y_test_class, pred, zero_division=0)) * 100, 2),
            "confusion_matrix": confusion_matrix(y_test_class, pred, labels=[0, 1]).tolist()
        }

    # Model Comparison Table Summary
    comparison_table = [
        {
            "model": "Logistic Regression",
            "task": "Classification",
            "accuracy": f"{log_metrics['accuracy']}%",
            "precision": f"{log_metrics['precision']}%",
            "recall": f"{log_metrics['recall']}%",
            "f1_score": f"{log_metrics['f1_score']}%"
        },
        {
            "model": "Decision Tree",
            "task": "Classification",
            "accuracy": f"{dt_metrics['accuracy']}%",
            "precision": f"{dt_metrics['precision']}%",
            "recall": f"{dt_metrics['recall']}%",
            "f1_score": f"{dt_metrics['f1_score']}%"
        },
        {
            "model": "Random Forest",
            "task": "Classification",
            "accuracy": f"{rf_metrics['accuracy']}%",
            "precision": f"{rf_metrics['precision']}%",
            "recall": f"{rf_metrics['recall']}%",
            "f1_score": f"{rf_metrics['f1_score']}%"
        },
        {
            "model": "Linear Regression",
            "task": "Score Regression",
            "accuracy": f"R²: {lin_metrics['r2_score']}",
            "precision": f"MAE: {lin_metrics['mae']}",
            "recall": f"MSE: {lin_metrics['mse']}",
            "f1_score": f"RMSE: {lin_metrics['rmse']}"
        }
    ]
    for model in extra_models.values():
        comparison_table.append({
            "model": model["model"], "task": "Classification",
            "accuracy": f"{model['accuracy']}%", "precision": f"{model['precision']}%",
            "recall": f"{model['recall']}%", "f1_score": f"{model['f1_score']}%"
        })

    # Grouped Metrics Data for Chart.js Comparison Chart
    grouped_comparison = {
        "metrics": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "logistic": [log_metrics['accuracy'], log_metrics['precision'], log_metrics['recall'], log_metrics['f1_score']],
        "decision_tree": [dt_metrics['accuracy'], dt_metrics['precision'], dt_metrics['recall'], dt_metrics['f1_score']],
        "random_forest": [rf_metrics['accuracy'], rf_metrics['precision'], rf_metrics['recall'], rf_metrics['f1_score']]
    }
    for key, model in extra_models.items():
        grouped_comparison[key] = [model[metric] for metric in ("accuracy", "precision", "recall", "f1_score")]

    models_summary = {
        "logistic_regression": log_metrics,
        "decision_tree": dt_metrics,
        "random_forest": rf_metrics,
        "additional_classifiers": extra_models,
        "linear_regression": lin_metrics,
        "log_top_features": log_top_features,
        "dt_top_features": dt_top_features,
        "rf_top_features": rf_top_features,
        "tree_rules_text": tree_rules_text,
        "comparison_table": comparison_table,
        "grouped_comparison": grouped_comparison,
        "target_correlations": target_corrs_clean,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "disclaimer": "Model metrics are evaluated strictly on unseen test dataset (20% split) with preprocessors fitted on training data."
    }

    pipelines = {
        "logistic": log_reg_pipeline,
        "decision_tree": dt_pipeline,
        "random_forest": rf_pipeline,
        "linear_regression": lin_reg_pipeline
    }
    pipelines.update(extra_pipelines)

    return pipelines, models_summary

def perform_pca_analysis(df):
    """
    Perform Principal Component Analysis (PCA) for dimensionality reduction.
    """
    num_features = [
        "Crowding_Level", "Punctuality_Rating", "Cleanliness_Rating",
        "Staff_Behavior_Rating", "Safety_Rating", "Fare_Satisfaction",
        "Delay_Frequency_Num", "Waiting_Time_Num", "Travel_Time_Num",
        "BEST_Usage_Frequency_Num"
    ]

    X_num = df[num_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    n_comp = min(5, len(num_features))
    pca = PCA(n_components=n_comp, random_state=42)
    pca_transformed = pca.fit_transform(X_scaled)

    var_ratio = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]
    cum_var = [round(float(v) * 100, 2) for v in np.cumsum(pca.explained_variance_ratio_)]

    loadings = pca.components_
    pc1_loadings = pd.DataFrame({
        "feature": [f.replace("_Num", "").replace("_", " ") for f in num_features],
        "weight": np.round(loadings[0], 3)
    }).sort_values(by="weight", key=abs, ascending=False).to_dict(orient="records")

    pc2_loadings = pd.DataFrame({
        "feature": [f.replace("_Num", "").replace("_", " ") for f in num_features],
        "weight": np.round(loadings[1], 3)
    }).sort_values(by="weight", key=abs, ascending=False).to_dict(orient="records")

    pca_scatter = []
    for i in range(len(df)):
        pca_scatter.append({
            "pc1": round(float(pca_transformed[i, 0]), 3),
            "pc2": round(float(pca_transformed[i, 1]), 3),
            "satisfaction": str(df["Satisfaction_Text"].iloc[i])
        })

    return {
        "n_components": n_comp,
        "explained_variance_ratio": var_ratio,
        "cumulative_variance": cum_var,
        "pc1_loadings": pc1_loadings,
        "pc2_loadings": pc2_loadings,
        "pca_scatter": pca_scatter
    }

def perform_kmeans_clustering(df, k=3):
    """
    Apply K-Means clustering and dynamic profiling on numerical travel & rating features.
    """
    cluster_features = [
        "Waiting_Time_Num", "Travel_Time_Num", "Crowding_Level",
        "Punctuality_Rating", "Safety_Rating", "Satisfaction_Score",
        "BEST_Usage_Frequency_Num"
    ]

    X_cluster = df[cluster_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    max_k = min(6, len(df))
    inertias = []
    k_range = list(range(1, max_k + 1))
    for i in k_range:
        km = KMeans(n_clusters=i, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(round(float(km.inertia_), 2))

    final_k = min(k, len(df))
    kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
    df_clustered = df.copy()
    df_clustered["Cluster"] = kmeans.fit_predict(X_scaled)

    cluster_profiles = []
    cluster_sizes = []
    for c in range(final_k):
        sub = df_clustered[df_clustered["Cluster"] == c]
        cnt = len(sub)
        avg_wait = sub["Waiting_Time_Num"].mean()
        avg_crowd = sub["Crowding_Level"].mean()
        avg_sat = sub["Satisfaction_Score"].mean()
        avg_punct = sub["Punctuality_Rating"].mean()

        traits = []
        if avg_sat >= 3.8:
            traits.append("High Satisfaction Riders")
        elif avg_sat <= 2.5:
            traits.append("Dissatisfied Commuters")
        else:
            traits.append("Moderate Satisfaction Commuters")

        if avg_wait >= 20:
            traits.append("High Wait Time")
        elif avg_wait <= 8:
            traits.append("Short Wait Time")

        if avg_crowd >= 4.0:
            traits.append("Peak Rush Riders")

        name = " - ".join(traits) if traits else f"Cluster {c+1}"

        cluster_sizes.append({"label": f"Cluster {c+1}", "count": int(cnt)})

        profile = {
            "cluster_id": c + 1,
            "cluster_name": name,
            "count": int(cnt),
            "percentage": round(float(cnt / len(df) * 100), 1),
            "avg_satisfaction": round(float(avg_sat), 2),
            "avg_waiting_time": round(float(avg_wait), 1),
            "avg_crowding": round(float(avg_crowd), 2),
            "avg_punctuality": round(float(avg_punct), 2)
        }
        cluster_profiles.append(profile)

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)
    
    pca_data = []
    for i in range(len(df_clustered)):
        pca_data.append({
            "x": round(float(components[i, 0]), 3),
            "y": round(float(components[i, 1]), 3),
            "cluster": int(df_clustered["Cluster"].iloc[i] + 1),
            "satisfaction": str(df_clustered["Satisfaction_Text"].iloc[i])
        })

    return {
        "k_range": k_range,
        "inertias": inertias,
        "selected_k": final_k,
        "cluster_sizes": cluster_sizes,
        "cluster_profiles": cluster_profiles,
        "pca_scatter": pca_data
    }

def predict_all_models(pipelines_dict, input_dict):
    """
    Predict satisfaction for a new passenger profile using all four trained models.
    """
    freq_map = {
        "Daily": 5, "4–6 times a week": 4, "4-6 times a week": 4,
        "2–3 times a week": 3, "2-3 times a week": 3, "Once a week": 2, "Occasionally": 1
    }
    time_map = {
        "Less than 15 minutes": 10, "Under 15 mins": 10,
        "15–30 minutes": 22.5, "15-30 minutes": 22.5, "15–30 mins": 22.5, "15-30 mins": 22.5,
        "31–45 minutes": 38, "46–60 minutes": 53, "30–60 mins": 45, "30-60 mins": 45,
        "More than 60 minutes": 75, "Over 60 mins": 75
    }
    wait_map = {
        "Less than 5 minutes": 3, "Under 5 mins": 3,
        "5–10 minutes": 7.5, "5-10 minutes": 7.5, "5–15 mins": 10, "5-15 mins": 10,
        "11–15 minutes": 13, "11-15 minutes": 13,
        "16–30 minutes": 23, "16-30 minutes": 23, "15–30 mins": 22.5, "15-30 mins": 22.5,
        "More than 30 minutes": 40, "Over 30 mins": 40
    }
    delay_map = {
        "Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4, "Frequently": 4, "Very Often": 5, "Always": 5
    }

    row = {
        "Age": input_dict.get("Age", "21–25"),
        "Gender": input_dict.get("Gender", "Male"),
        "Occupation": input_dict.get("Occupation", "Student"),
        "Travel_Purpose": input_dict.get("Travel_Purpose", "College / Education"),
        "Crowding_Level": float(input_dict.get("Crowding_Level", 3)),
        "Punctuality_Rating": float(input_dict.get("Punctuality_Rating", 3)),
        "Cleanliness_Rating": float(input_dict.get("Cleanliness_Rating", 3)),
        "Staff_Behavior_Rating": float(input_dict.get("Staff_Behavior_Rating", 3)),
        "Safety_Rating": float(input_dict.get("Safety_Rating", 4)),
        "Fare_Satisfaction": float(input_dict.get("Fare_Satisfaction", 4)),
        "Delay_Frequency_Num": delay_map.get(str(input_dict.get("Delay_Frequency", "Sometimes")).strip(), 3),
        "Waiting_Time_Num": wait_map.get(str(input_dict.get("Waiting_Time", "11–15 minutes")).strip(), 13),
        "Travel_Time_Num": time_map.get(str(input_dict.get("Travel_Time", "15–30 minutes")).strip(), 22.5),
        "BEST_Usage_Frequency_Num": freq_map.get(str(input_dict.get("BEST_Usage_Frequency", "Daily")).strip(), 5)
    }

    input_df = pd.DataFrame([row])

    # 1. Logistic Regression Prediction
    log_pipe = pipelines_dict["logistic"]
    log_label = log_pipe.predict(input_df)[0]
    log_probs = log_pipe.predict_proba(input_df)[0]
    log_conf = round(float(log_probs[1] if log_label == 1 else log_probs[0]) * 100, 1)

    # 2. Decision Tree Prediction
    dt_pipe = pipelines_dict["decision_tree"]
    dt_label = dt_pipe.predict(input_df)[0]
    dt_probs = dt_pipe.predict_proba(input_df)[0] if hasattr(dt_pipe, "predict_proba") else [0.5, 0.5]
    dt_conf = round(float(dt_probs[1] if dt_label == 1 else dt_probs[0]) * 100, 1)

    # 3. Random Forest Prediction
    rf_pipe = pipelines_dict["random_forest"]
    rf_label = rf_pipe.predict(input_df)[0]
    rf_probs = rf_pipe.predict_proba(input_df)[0] if hasattr(rf_pipe, "predict_proba") else [0.5, 0.5]
    rf_conf = round(float(rf_probs[1] if rf_label == 1 else rf_probs[0]) * 100, 1)

    # 4. Linear Regression Continuous Prediction (Bounded 1.0 to 5.0)
    lin_pipe = pipelines_dict["linear_regression"]
    raw_score = lin_pipe.predict(input_df)[0]
    bounded_score = round(float(np.clip(raw_score, 1.0, 5.0)), 2)

    return {
        "logistic_regression": {
            "prediction": "SATISFIED" if log_label == 1 else "NOT SATISFIED",
            "confidence": log_conf,
            "prob_satisfied": round(float(log_probs[1]) * 100, 1),
            "prob_not_satisfied": round(float(log_probs[0]) * 100, 1)
        },
        "decision_tree": {
            "prediction": "SATISFIED" if dt_label == 1 else "NOT SATISFIED",
            "confidence": dt_conf,
            "prob_satisfied": round(float(dt_probs[1]) * 100, 1),
            "prob_not_satisfied": round(float(dt_probs[0]) * 100, 1)
        },
        "random_forest": {
            "prediction": "SATISFIED" if rf_label == 1 else "NOT SATISFIED",
            "confidence": rf_conf,
            "prob_satisfied": round(float(rf_probs[1]) * 100, 1),
            "prob_not_satisfied": round(float(rf_probs[0]) * 100, 1)
        },
        "linear_regression": {
            "predicted_score": bounded_score,
            "max_score": 5.0
        },
        "disclaimer": "This prediction is generated based on mathematical model estimation for the entered passenger profile."
    }
