# Loan-prediction
🏦 End-to-End Loan Approval Prediction System
An end-to-end Machine Learning pipeline and production-ready web application designed to automate credit risk assessment. By analyzing financial and demographic applicant profiles, the system minimizes a bank's risk exposure.

Through robust feature engineering, class imbalance treatment, and rigorous hyperparameter tuning, the final model's performance was optimized from a 66% baseline to a 90% production-ready accuracy rate.

🔗 [Live Demo Link Here] -  https://loan-prediction-j2i4.onrender.com/

📈 Performance Journey & Optimization
Real-world credit datasets are heavily skewed. This project highlights the journey of diagnosing model deficiencies and systematically engineering a high-accuracy solution:

The Baseline (66% Accuracy): The initial model suffered from the "accuracy paradox" due to a severe 70/30 class imbalance. It routinely misclassified high-risk defaults because it lacked sufficient minority-class data to learn from.

The Breakthrough (90% Accuracy): By implementing data-centric and algorithmic optimizations, the model achieved a massive performance leap, drastically lowering False Negatives (the costliest error for credit lenders).

🛠️ System Architecture & ML Pipeline
The data flows through a structured, modular pipeline to guarantee zero data leakage during training and flawless inference during live deployment:

Data Preprocessing: * Handled missing values cleanly using Median Imputation for continuous numerical data and Mode Imputation for categorical entries.

Managed massive variance in applicant and co-applicant incomes using a RobustScaler to mitigate outlier distortion.

Feature Engineering: * Engineered a combined Total_Income metric to capture household purchasing power and eliminate multicollinearity issues between individual income variables.

Imbalance Mitigation: * Applied SMOTE (Synthetic Minority Over-sampling Technique) exclusively to the training split to synthesize logical profiles for the rejected loan class.

Model Exploration & Tuning:

Evaluated multiple architectures including Logistic Regression and XGBoost.

Selected Random Forest Classifier as the champion model due to its exceptional stability and ensemble variance-reduction on tabular feature spaces.

Executed exhaustive hyperparameter tuning using GridSearchCV to optimize n_estimators, max_depth, and min_samples_split.

📉 Model Evaluation (Confusion Matrix Insights)
During testing, evaluation focused heavily on the trade-off between Precision and Recall:

False Positives (Type I Error): Risking bank capital by predicting a default-bound applicant as "Eligible". The final Random Forest model was tightly tuned to suppress this specific error.

False Negatives (Type II Error): Turning away safe, credit-worthy customers. SMOTE oversampling directly minimized this issue, teaching the model to accurately differentiate subtle boundaries between high-risk and low-risk applications.

🧰 Tech Stack & Tools
Core Languages & Frameworks: Python, Streamlit

Data Science Ecosystem: Scikit-Learn, Pandas, NumPy, Imbalanced-Learn (imblearn)

Model Serialization: Joblib

DevOps & Deployment: Git, GitHub, Render (PaaS Container Hosting)

💻 Local Installation & Setup
To replicate this environment locally, clone the repository and install the production dependencies:

Bash
# Clone the repository
git clone https://github.com/Muthuworkspace/Loan-prediction.git

# Navigate into the project folder
cd Loan-prediction

# Install required packages
pip install -r requirements.txt

# Run the live Streamlit dashboard locally
streamlit run app.py


📁 Repository Structure
Plaintext
Loan-prediction/
│
├── app.py                 # Streamlit web application interface script
├── loan_model_v1.pkl       # Serialized 90% accuracy Random Forest pipeline
├── requirements.txt       # Strict version-controlled project dependencies
├── .gitignore             # Excludes pycache, local environments, and checkpoint data
└── README.md              # Project documentation
