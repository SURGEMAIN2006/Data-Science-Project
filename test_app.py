import os
import sys
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_processor import preprocess_dataset
from src.ml_models import train_all_models, perform_pca_analysis, perform_kmeans_clustering, predict_all_models
from src.insights import generate_automated_insights
from app import app

def run_tests():
    print("==========================================")
    print("Testing 5 ML Techniques Pipeline...")
    print("==========================================")

    google_form_csv = os.path.join(os.path.dirname(__file__), "data", "BEST_Bus_Survey_100_Augmented_Responses.csv")
    assert os.path.exists(google_form_csv), "BEST_Bus_Survey_100_Augmented_Responses.csv missing!"
    print(f"[OK] Found Google Form dataset at: {google_form_csv}")

    df = pd.read_csv(google_form_csv)
    print(f"[OK] Raw CSV loaded: {len(df)} rows, {len(df.columns)} columns")

    # 1. Test Preprocessing & Column Mapper
    clean_df, summary = preprocess_dataset(df)
    assert clean_df is not None, f"Preprocessing failed: {summary.get('error')}"
    print(f"[OK] Preprocessing successful: {summary['processed_rows']} unique rows processed")
    print(f"  - Satisfied count: {summary['satisfied_count']} ({summary['satisfaction_pct']}%)")
    print(f"  - Not Satisfied count: {summary['not_satisfied_count']}")

    # 2. Test Supervised Multi-Models (Logistic Regression, Decision Tree, Random Forest, Linear Regression)
    pipelines, ml_summary = train_all_models(clean_df)
    assert pipelines is not None, f"Multi-model training failed: {ml_summary.get('error')}"
    print("[OK] All 4 supervised models trained successfully!")
    
    log = ml_summary["logistic_regression"]
    print(f"  1. Logistic Regression -> Acc: {log['accuracy']}%, Prec: {log['precision']}%, Rec: {log['recall']}%, F1: {log['f1_score']}%")
    
    dt = ml_summary["decision_tree"]
    print(f"  2. Decision Tree       -> Acc: {dt['accuracy']}%, Prec: {dt['precision']}%, Rec: {dt['recall']}%, F1: {dt['f1_score']}%")

    rf = ml_summary["random_forest"]
    print(f"  3. Random Forest        -> Acc: {rf['accuracy']}%, Prec: {rf['precision']}%, Rec: {rf['recall']}%, F1: {rf['f1_score']}%")
    
    lin = ml_summary["linear_regression"]
    print(f"  4. Linear Regression   -> MAE: {lin['mae']}, MSE: {lin['mse']}, RMSE: {lin['rmse']}, R2: {lin['r2_score']}")

    # 3. Test PCA (Principal Component Analysis)
    pca_res = perform_pca_analysis(clean_df)
    assert pca_res is not None, "PCA analysis failed!"
    print(f"[OK] 5. PCA Analysis executed successfully! (Explained Variance: {pca_res['explained_variance_ratio']})")

    # 4. Test Unsupervised ML: K-Means Clustering
    kmeans_res = perform_kmeans_clustering(clean_df, k=3)
    assert kmeans_res is not None, "K-Means clustering failed!"
    print(f"[OK] K-Means Clustering executed successfully! ({len(kmeans_res['cluster_profiles'])} clusters derived)")

    # 5. Test Live Multi-Model Predictor
    test_passenger = {
        "Age": "21–25", "Gender": "Male", "Occupation": "Student",
        "BEST_Usage_Frequency": "Daily", "Travel_Purpose": "College / Education",
        "Travel_Time": "15–30 minutes", "Waiting_Time": "11–15 minutes",
        "Delay_Frequency": "Often", "Crowding_Level": 4,
        "Punctuality_Rating": 3, "Cleanliness_Rating": 3,
        "Staff_Behavior_Rating": 3, "Safety_Rating": 4, "Fare_Satisfaction": 4
    }
    pred_res = predict_all_models(pipelines, test_passenger)
    print("[OK] Live Multi-Model Predictor Test Result:")
    print(f"  - Logistic Reg Prediction:  {pred_res['logistic_regression']['prediction']} ({pred_res['logistic_regression']['confidence']}%)")
    print(f"  - Decision Tree Prediction:  {pred_res['decision_tree']['prediction']} ({pred_res['decision_tree']['confidence']}%)")
    print(f"  - Random Forest Prediction:  {pred_res['random_forest']['prediction']} ({pred_res['random_forest']['confidence']}%)")
    print(f"  - Linear Reg Score Prediction: {pred_res['linear_regression']['predicted_score']} / 5.0")

    # 6. Test Flask API Endpoints
    print("\n------------------------------------------")
    print("Testing Flask REST API Endpoints...")
    print("------------------------------------------")
    with app.test_client() as client:
        # Load sample route
        res_sample = client.get("/api/load-sample")
        assert res_sample.status_code == 200, "API /api/load-sample failed!"
        json_data = res_sample.get_json()
        assert json_data["success"] == True
        assert "random_forest" in json_data["ml_models"]
        print("[OK] Flask GET /api/load-sample returned 200 OK")

        # Predict route
        res_pred = client.post("/api/predict", json=test_passenger)
        assert res_pred.status_code == 200, "API /api/predict failed!"
        pred_json = res_pred.get_json()
        assert pred_json["success"] == True
        assert "random_forest" in pred_json["result"]
        print("[OK] Flask POST /api/predict returned 200 OK (All 4 Models Evaluated)")

    print("\n==========================================")
    print("ALL 5 REQUIRED TECHNIQUES TESTED & PASSED!")
    print("==========================================")

if __name__ == "__main__":
    run_tests()
