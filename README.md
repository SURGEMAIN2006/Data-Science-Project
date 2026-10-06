# BEST Bus Passenger Experience & Satisfaction Analysis Using Data Science and Machine Learning

A complete, college-level Data Science and Machine Learning web application designed to analyze Mumbai's BEST bus commuter travel behavior, service experience, and overall passenger satisfaction.

Built using **Python (Flask, Pandas, Scikit-Learn)** and **HTML5/CSS3/JavaScript (Chart.js)**, this project is fully dynamic and ready for deployment on **Vercel** or local execution.

---

## 📌 Project Overview

Brihanmumbai Electric Supply and Transport (BEST) buses serve millions of commuters daily across Mumbai. Understanding passenger perception regarding waiting times, crowding levels, punctuality, staff behavior, and safety is critical for public transit planning and service optimization.

This application accepts Google Forms survey responses exported as CSV. It executes data preprocessing, column auto-mapping, exploratory data analysis (EDA), descriptive statistics, correlation heatmaps, supervised classification model comparisons, unsupervised segmentation (K-Means Clustering), and live passenger satisfaction predictions. Export the Google Form responses as CSV and upload that file in the dashboard.

The bundled `BEST_Bus_Survey_100_Augmented_Responses.csv` is an augmented/demo dataset. Treat its analysis as a software demonstration, not as evidence about actual BEST passengers. Replace it with the genuine Google Forms export for the project results, and report clearly if any responses were synthetically augmented.

---

## 🚀 Key Features & Modules

- **Dynamic CSV Dataset Upload**: Prominent uploader accepting Google Form CSV responses at any time.
- **Intelligent Google Form Column Mapper**: Auto-detects verbatim survey question text and maps them to standard canonical variables.
- **Data Preprocessing Pipeline**: Automatic duplicate removal, missing value imputation, categorical encoding, ordinal preservation, and binary target calculation.
- **KPI Summary Cards**: Real-time display of Total Respondents, Average Satisfaction Score, Satisfied %, Not Satisfied %, and Average Waiting Time.
- **Exploratory Data Analysis (EDA)**: Dynamic interactive charts powered by Chart.js (Demographics, Usage Frequency, Travel Purpose, Journey Time, Waiting Time, Service Ratings, Delays).
- **Descriptive Statistics**: Calculates Mean, Median, Mode, Standard Deviation, Min, Max, 25th, 50th, and 75th percentiles for numerical parameters.
- **Correlation Matrix & Heatmap**: Analyzes linear associations between service parameters and satisfaction with academic disclaimers.
- **Supervised ML model comparison**: Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Gaussian Naive Bayes, Gradient Boosting, Extra Trees, and AdaBoost classify satisfaction (scores 4–5 vs 1–3). Linear Regression estimates the original 1–5 score. Classifiers use a stratified 80/20 holdout split, `random_state=42`, and preprocessing pipelines that impute missing predictors and fit encoders/scalers on training data only. Metrics include Accuracy, Precision, Recall, F1 Score, and confusion matrices.
- **Unsupervised ML (K-Means Clustering)**: Evaluates Elbow curve (Inertia vs K), projects clusters in 2D PCA space, and derives dynamic segment descriptions based on cluster statistics.
- **Live Passenger Satisfaction Predictor**: Interactive input form (sliders & dropdowns) allowing users to enter a new passenger profile and receive immediate probability-based predictions.
- **Automated Dynamic Insights**: Text findings generated strictly from the active dataset.
- **Academic Viva Q&A Guide**: Comprehensive built-in accordion section explaining key machine learning and statistics concepts for viva preparation.

---

## 🛠️ Technology Stack

- **Backend**: Python 3.9+, Flask (REST API)
- **Data Science & ML**: Pandas, NumPy, Scikit-Learn
- **Frontend**: HTML5, CSS3, JavaScript (ES6), Chart.js, RemixIcon
- **Deployment**: Vercel Serverless Functions (`@vercel/python`) / Local WSGI

---

## 📁 Project File Structure

```
BEST-Bus-Data-Science/
│
├── app.py                      # Flask backend & REST API endpoints
├── requirements.txt            # Python dependencies
├── vercel.json                 # Vercel serverless deployment config
├── README.md                   # Complete academic documentation
│
├── api/
│   └── index.py                # Vercel serverless entry point
│
├── data/
│   └── sample_best_bus_dataset.csv # Realistic sample survey dataset (~60 rows)
│
├── src/
│   ├── data_processor.py       # Column mapper, cleaning & target builder
│   ├── ml_models.py            # Logistic Regression, K-Means & Predictor
│   └── insights.py             # Automated text insights generator
│
├── templates/
│   └── index.html              # Modern responsive Web Dashboard
│
└── static/
    ├── css/
    │   └── style.css           # Styling, grid layout, KPI cards & themes
    └── js/
        └── main.js             # Dynamic Chart.js rendering & API client
```

---

## 📊 Dataset & Column Mapping

Use Google Forms → Responses → Download responses (.csv), then upload the CSV in the dashboard. The app does not connect directly to a Google account or fetch form responses automatically. The CSV should include one column for each survey variable below. The built-in sample is augmented and must not be presented as 100 genuine passenger responses.

The Google Form is designed around 15 primary survey variables:

| Canonical Variable | Example Google Form Question Header |
| :--- | :--- |
| `Age` | "What is your age?" |
| `Gender` | "Gender" |
| `Occupation` | "What is your current occupation?" |
| `BEST_Usage_Frequency` | "How often do you travel by BEST bus?" |
| `Travel_Purpose` | "What is your primary travel purpose?" |
| `Travel_Time` | "How long is your typical BEST bus journey?" |
| `Waiting_Time` | "What is your usual waiting time for a BEST bus?" |
| `Crowding_Level` | "How crowded is the BEST bus usually when you travel?" |
| `Punctuality_Rating` | "How would you rate BEST bus punctuality?" |
| `Cleanliness_Rating` | "How would you rate the cleanliness of BEST buses?" |
| `Staff_Behavior_Rating` | "How would you rate the behavior of BEST drivers and conductors?" |
| `Safety_Rating` | "How safe do you feel while travelling by BEST?" |
| `Fare_Satisfaction` | "How satisfied are you with BEST bus fares?" |
| `Delay_Frequency` | "How frequently do you experience delays while travelling by BEST?" |
| `Satisfaction_Score` | "Overall how satisfied are you with BEST bus services?" |

---

## ⚡ How to Run Locally

1. **Clone or Navigate to the Project Directory**:
   ```bash
   cd BEST-Bus-Data-Science
   ```

2. **Create a Virtual Environment (Optional but Recommended)**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Flask Application**:
   ```bash
   python app.py
   ```

5. **Open Browser**:
   Navigate to `http://127.0.0.1:5000` to interact with the dashboard.

---

## ☁️ How to Deploy on Vercel

1. **Install Vercel CLI** (if not installed):
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**:
   ```bash
   vercel login
   ```

3. **Deploy from Project Root**:
   ```bash
   vercel
   ```
   Follow the CLI prompts (accept defaults). Vercel will automatically detect `vercel.json` and deploy the Python app serverlessly!

4. **Deploy to Production**:
   ```bash
   vercel --prod
   ```

---

## 🎓 Academic Viva Questions & Answers

### 1. Why is Logistic Regression suitable for this problem?
> Logistic Regression is designed for binary classification target variables (Satisfied vs Not Satisfied). It provides direct interpretability via feature coefficients, showing how each rating parameter impacts the log-odds of satisfaction.

### 2. What is binary classification and train-test split?
> Binary classification categorizes instances into 0 (Not Satisfied: Score 1-3) or 1 (Satisfied: Score 4-5). A train-test split (80% training / 20% testing) ensures we evaluate model generalization on unseen data.

### 3. What do Accuracy, Precision, Recall, and F1 Score measure?
> - **Accuracy**: Percentage of correct overall predictions.
> - **Precision**: Ratio of true satisfied passengers among all predicted satisfied.
> - **Recall**: Ratio of true satisfied passengers correctly identified by the model.
> - **F1 Score**: Harmonic mean of Precision and Recall.

### 4. Why scale data before K-Means Clustering?
> K-Means calculates Euclidean distances between data points. Feature scaling (StandardScaler) standardizes features to zero mean and unit variance so variables with larger numeric scales (e.g. travel minutes vs 1-5 rating scale) do not dominate clustering distance math.

---

## 📄 License
Academic CS Project & Copyleft © 2026.
