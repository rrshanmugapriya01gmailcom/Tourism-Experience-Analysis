# 🌍 TourEase – Smart Tourism Intelligence System

TourEase is a  machine learning project built to revolutionize the way travelers plan their trips. It predicts how users travel (Visit Mode), what ratings they'll give to attractions, and recommends places they’re most likely to enjoy — all through an interactive Streamlit dashboard.

---

##  Project Highlights

-  **Predict Visit Mode**: Understand how users prefer to travel — Solo, Family, or with Friends.
-  **Attraction Rating Prediction**: Estimate the rating a user might give to a tourist spot.
-  **Personalized Recommendations**: Suggest attractions using hybrid recommendation logic.
-  **Interactive UI**: Built with Streamlit for real-time predictions and insights.

---

## 🧠 ML Models & Techniques

| Task                  | Model Used               | Justification                                     |
|-----------------------|--------------------------|--------------------------------------------------|
| Visit Mode Prediction | Random Forest Classifier | Handles feature interactions, good accuracy      |
| Rating Prediction     | Random Forest Regressor  | Robust and interpretable                         |
| Recommendation System| Cosine Similarity (CB & CF)| Ideal for sparse user-item data                 |

✔️ Hyperparameter tuning with **GridSearchCV**  
✔️ Feature Scaling using **MinMaxScaler**  
✔️ Encoding: **LabelEncoder**, **OneHotEncoder**

---

## 📊 EDA & Feature Engineering

- **Univariate**: Rating & travel mode distribution
- **Bivariate**: Season vs Ratings, Region vs Visit Type
- **Multivariate**: Interactions between popularity, seasonality, and user behavior

🔧 Feature Engineered:
- `AttractionPopularityIndex`
- `UserVisitFrequency`
- `SeasonalScore`
- `RatingDeviation`

---

## 🧪 Statistical Validation

- **Chi-Square Test** used to identify meaningful associations between categorical variables (e.g., Region ↔ Visit Mode)

---

## 📏 Evaluation Metrics

- **Classification**: `Accuracy`, `F1-Score`
- **Regression**: `Root Mean Squared Error (RMSE)`

---

🛠️ Technologies Used
Python

Streamlit

Scikit-learn

Pandas, NumPy

Pickle (model serialization)

Matplotlib/Seaborn (EDA)

📁 Folder Structure
TourEase/
│
├── app.py                      # Streamlit web app
├── RatingPrediction.ipynb      # Rating model development
├── VisitModePrediction.ipynb   # Visit mode model
├── recommendationsystem.ipynb  # Recommendation logic
├── *.pkl                       # Saved models and encoders
├── *.csv                       # Processed datasets
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
✅ How to Run Locally
Clone the repo:


git clone https://github.com/yourusername/TourEase.git
cd TourEase
Install dependencies:


pip install -r requirements.txt
Launch the app:


streamlit run app.py
💼 Business Use Case
This project empowers tourism businesses to:

Deliver personalized travel suggestions

Improve user engagement & retention

Optimize marketing with data-driven insights

