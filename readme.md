# 💼 ConnectSphere Telecom – Customer Churn Prediction
Project Goal:
This project aims to help ConnectSphere Telecom, a regional telecom provider, reduce customer churn by building a binary classification model that predicts whether a customer is likely to leave (churn) based on service usage patterns and account-level information.

# 🔍 Problem Statement
Customer churn leads to significant revenue loss and increased customer acquisition costs. The goal is to develop a machine learning solution—specifically an Artificial Neural Network (ANN)—to:

Predict customer churn using historical data

Identify high-risk customers early

Help the business implement effective retention strategies

🧠 Machine Learning Approach
We use a feedforward ANN (Artificial Neural Network) built using TensorFlow/Keras for classification. The model outputs a churn probability between 0 and 1 for each customer.

# 📊 Dataset
The dataset used is publicly available and contains telecom customer information such as:

Demographics: Gender, Senior Citizen, Partner, Dependents

Account info: Tenure, Contract, Payment method, Monthly & Total Charges

Services signed up for: Internet service, Streaming, Device protection, etc.

Target variable: Churn (Yes/No)

Note: The dataset is pre-cleaned in the notebook (.ipynb) before training.

# 🛠️ Tools & Libraries
Category	Libraries/Tools
Data Handling	pandas, numpy
Preprocessing	sklearn.preprocessing, LabelEncoder, StandardScaler
Model Training	tensorflow.keras, Sequential, Dense
Evaluation	accuracy_score, f1_score, classification_report
Web App (UI)	streamlit

# 🧪 Model Evaluation Metrics
Metric	Description
Accuracy	Overall correctness of the model
F1 Score	Harmonic mean of precision and recall (useful for imbalanced data)
Confusion Matrix	To visualize True Positives, False Positives, etc.

Final Model Performance:

Accuracy: ~85%

F1-Score: Balanced for both churners and non-churners

# 🖥️ Streamlit App (UI)
You can run the included Streamlit app to test predictions interactively:

✅ Features:
Upload customer data via CSV

Instant churn probability predictions

Highlight high-risk customers

Download results with churn labels

📦 How to Run
bash
Copy
Edit
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the Streamlit app
streamlit run streamlit_app.py
📁 Repository Structure
├── Churn_Prediction_for_ConnectSphere_Telecom.ipynb  # Colab notebook
├── churn_model.h5                                    # Trained ANN model
├── streamlit_app.py                                  # Streamlit web app
├── requirements.txt                                  # Python dependencies
└── README.md                                          # Project documentation
📥 Input Format (CSV)
The input CSV should include customer records in the format of the original dataset. Required columns include:

gender, SeniorCitizen, Partner, tenure, InternetService, etc.

Do not include the customerID column (or it will be dropped automatically).

📤 Output Format
The app outputs a CSV file with:

Churn Probability: Likelihood of churn (0.0 to 1.0)

Churn Risk: Categorized as High (≥ 0.5) or Low (< 0.5)

✅ Future Improvements
Hyperparameter tuning for better model performance

Add SHAP or LIME for model interpretability

Add deployment support (Docker or Hugging Face Spaces)

Automated model retraining pipeline

🙏 Acknowledgements
IBM Telco Churn Dataset on Kaggle

TensorFlow / Keras

Streamlit Community

📬 Contact
For any questions, reach out to the project author:

Suman Banerjee
📧 Email: sumaneducator10@gmail.com
🌐 GitHub: https://github.com/codewithsumanbyte/