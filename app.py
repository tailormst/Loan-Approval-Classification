import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .approved {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .rejected {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.1rem;
        padding: 0.5rem;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">💰 Loan Approval Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict loan approval using machine learning models</div>', unsafe_allow_html=True)

# Model mapping
MODEL_FILES = {
    "LightGBM": "lightgbm_pipeline.joblib",
    "Random Forest": "randomforest_pipeline.joblib",
    "Decision Tree": "decisiontree_pipeline.joblib",
    "Logistic Regression": "logisticregression_pipeline.joblib",
    "K-Nearest Neighbors": "knn_pipeline.joblib",
    "Support Vector Machine": "svm_pipeline.joblib"
}


# Load model accuracies
@st.cache_resource
def load_accuracies():
    try:
        accuracies = joblib.load('models_save/model_accuracies.pkl')
        # Map the keys from the file to our display names
        mapped_accuracies = {}
        key_mapping = {
            'LightGBM': ['LightGBM', 'lightgbm', 'LGBM'],
            'Random Forest': ['RandomForest', 'random_forest', 'Random Forest', 'randomforest'],
            'Decision Tree': ['DecisionTree', 'decision_tree', 'Decision Tree', 'decisiontree'],
            'Logistic Regression': ['LogisticRegression', 'logistic_regression', 'Logistic Regression',
                                    'logisticregression'],
            'K-Nearest Neighbors': ['KNN', 'knn', 'K-Nearest Neighbors', 'KNeighbors'],
            'Support Vector Machine': ['SVM', 'svm', 'Support Vector Machine', 'SVC']
        }

        for display_name, possible_keys in key_mapping.items():
            for key in possible_keys:
                if key in accuracies:
                    mapped_accuracies[display_name] = accuracies[key]
                    break
            if display_name not in mapped_accuracies:
                mapped_accuracies[display_name] = 0.0

        return mapped_accuracies
    except Exception as e:
        st.sidebar.warning(f"Could not load accuracies: {str(e)}")
        # Default accuracies if file not found
        return {
            "LightGBM": 0.9341,
            "Random Forest": 0.9267,
            "Decision Tree": 0.9097,
            "Support Vector Machine": 0.8762,
            "Logistic Regression": 0.8596,
            "K-Nearest Neighbors": 0.8354
        }


# Load selected model
@st.cache_resource
def load_model(model_name):
    try:
        model_file = MODEL_FILES[model_name]
        model_path = f'models_save/{model_file}'
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


# Sidebar for model selection
st.sidebar.header("⚙️ Model Configuration")

# Load accuracies
accuracies = load_accuracies()

# Sort models by accuracy
sorted_models = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
default_model = sorted_models[0][0]

# Display model accuracies
st.sidebar.subheader("📊 Model Accuracies")
for model, acc in sorted_models:
    st.sidebar.metric(model, f"{acc * 100:.2f}%")

st.sidebar.markdown("---")

# Model selection
selected_model = st.sidebar.selectbox(
    "Select Model:",
    options=list(MODEL_FILES.keys()),
    index=list(MODEL_FILES.keys()).index(default_model),
    help="Choose the machine learning model for prediction"
)

st.sidebar.info(f"✨ **{default_model}** has the highest accuracy!")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Enter Loan Application Details")

    # Create input form
    with st.form("loan_form"):
        st.subheader("👤 Personal Information")
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            person_age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=30,
                step=1,
                help="Applicant's age"
            )

        with col_b:
            person_gender = st.selectbox(
                "Gender",
                options=["male", "female"],
                help="Applicant's gender"
            )

        with col_c:
            person_education = st.selectbox(
                "Education Level",
                options=["High School", "Associate", "Bachelor", "Master", "Doctorate"],
                help="Highest education level"
            )

        st.markdown("---")
        st.subheader("💼 Employment & Income")

        col_d, col_e = st.columns(2)

        with col_d:
            person_income = st.number_input(
                "Annual Income ($)",
                min_value=0,
                value=50000,
                step=1000,
                help="Annual income of the applicant"
            )

        with col_e:
            person_emp_exp = st.number_input(
                "Employment Experience (years)",
                min_value=0,
                max_value=50,
                value=5,
                step=1,
                help="Years of employment experience"
            )

        person_home_ownership = st.selectbox(
            "Home Ownership",
            options=["RENT", "OWN", "MORTGAGE", "OTHER"],
            help="Current home ownership status"
        )

        st.markdown("---")
        st.subheader("💳 Loan Details")

        col_f, col_g = st.columns(2)

        with col_f:
            loan_amnt = st.number_input(
                "Loan Amount ($)",
                min_value=500,
                value=10000,
                step=500,
                help="Total loan amount requested"
            )

        with col_g:
            loan_int_rate = st.number_input(
                "Interest Rate (%)",
                min_value=0.0,
                max_value=30.0,
                value=10.0,
                step=0.1,
                help="Loan interest rate"
            )

        loan_intent = st.selectbox(
            "Loan Purpose",
            options=["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
            help="Purpose of the loan"
        )

        st.markdown("---")
        st.subheader("📊 Credit History")

        col_i, col_j, col_k = st.columns(3)

        with col_i:
            credit_score = st.number_input(
                "Credit Score",
                min_value=300,
                max_value=850,
                value=650,
                step=10,
                help="Credit score (300-850)"
            )

        with col_j:
            cb_person_cred_hist_length = st.number_input(
                "Credit History Length (years)",
                min_value=0,
                max_value=50,
                value=5,
                step=1,
                help="Length of credit history in years"
            )

        with col_k:
            previous_loan_defaults_on_file = st.selectbox(
                "Previous Defaults",
                options=["No", "Yes"],
                help="Any previous loan defaults on file?"
            )

        submit_button = st.form_submit_button("🔍 Predict Loan Approval")

with col2:
    st.header("📈 Model Information")

    # Display selected model info
    st.info(f"""
    **Selected Model:** {selected_model}

    **Accuracy:** {accuracies.get(selected_model, 0.0) * 100:.2f}%

    This model will analyze your loan application and predict the likelihood of approval.
    """)

    # Show warning if accuracy is 0
    if accuracies.get(selected_model, 0.0) == 0.0:
        st.warning("⚠️ Accuracy data not available for this model")

    # Load the selected model
    if selected_model:
        model = load_model(selected_model)
        if model:
            st.success("✅ Model loaded successfully!")
        else:
            st.error("❌ Failed to load model")

# Process prediction
if submit_button:
    # Load the model
    model = load_model(selected_model)

    if model:
        try:
            # Feature engineering - match the training preprocessing exactly

            # Calculate derived features
            loan_percent_income = loan_amnt / (person_income + 1e-6)
            income_per_experience = person_income / (person_emp_exp + 1)
            cred_length_per_age = cb_person_cred_hist_length / (person_age + 1)

            # Log transforms for skewed features
            person_income_log = np.log1p(person_income)
            loan_amnt_log = np.log1p(loan_amnt)
            loan_percent_income_log = np.log1p(loan_percent_income)

            # Create input dataframe WITH raw person_income and loan_amnt
            # The pipeline expects these columns even though they get dropped later
            # IMPORTANT: Maintain column order to match training
            input_data = pd.DataFrame({
                'person_age': [person_age],
                'person_gender': [person_gender],
                'person_education': [person_education],
                'person_income': [person_income],  # Raw column needed by pipeline
                'person_emp_exp': [person_emp_exp],
                'person_home_ownership': [person_home_ownership],
                'loan_amnt': [loan_amnt],  # Raw column needed by pipeline
                'loan_intent': [loan_intent],
                'loan_int_rate': [loan_int_rate],
                'cb_person_cred_hist_length': [cb_person_cred_hist_length],
                'credit_score': [credit_score],
                'previous_loan_defaults_on_file': [previous_loan_defaults_on_file],
                # Engineered features
                'loan_percent_income': [loan_percent_income],
                'income_per_experience': [income_per_experience],
                'cred_length_per_age': [cred_length_per_age],
                # Log-transformed features
                'person_income_log': [person_income_log],
                'loan_amnt_log': [loan_amnt_log],
                'loan_percent_income_log': [loan_percent_income_log]
            })

            # Ensure consistent dtypes
            input_data = input_data.astype({
                'person_age': 'int64',
                'person_emp_exp': 'int64',
                'loan_int_rate': 'float64',
                'cb_person_cred_hist_length': 'int64',
                'credit_score': 'int64',
                'loan_percent_income': 'float64',
                'income_per_experience': 'float64',
                'cred_length_per_age': 'float64',
                'person_income_log': 'float64',
                'loan_amnt_log': 'float64',
                'loan_percent_income_log': 'float64'
            })

            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]

            # Display results
            st.markdown("---")
            st.header("🎯 Prediction Results")

            # Determine confidence level
            confidence = max(prediction_proba[0], prediction_proba[1])

            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box approved">
                    <h2>✅ LOAN APPROVED</h2>
                    <p style="font-size: 1.5rem; margin-top: 1rem;">
                        Confidence: {prediction_proba[1] * 100:.4f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="prediction-box rejected">
                    <h2>❌ LOAN REJECTED</h2>
                    <p style="font-size: 1.5rem; margin-top: 1rem;">
                        Confidence: {prediction_proba[0] * 100:.4f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Show detailed probabilities with more precision
            col_prob1, col_prob2 = st.columns(2)
            with col_prob1:
                st.metric("Probability of Rejection", f"{prediction_proba[0] * 100:.4f}%")
            with col_prob2:
                st.metric("Probability of Approval", f"{prediction_proba[1] * 100:.4f}%")

            # Show raw probabilities for debugging
            with st.expander("🔍 Raw Probability Values (Debug)"):
                st.write(f"Raw probabilities: {prediction_proba}")
                st.write(f"Rejection (class 0): {prediction_proba[0]}")
                st.write(f"Approval (class 1): {prediction_proba[1]}")
                st.write(f"Sum: {prediction_proba[0] + prediction_proba[1]}")

                if confidence > 0.99:
                    st.info("ℹ️ The model is highly confident (>99%) in this prediction. This could indicate:\n"
                            "- Very clear-cut case based on the input features\n"
                            "- The model might be overfit to the training data\n"
                            "- Try adjusting input values to see how sensitive the model is")

            # Show input summary
            with st.expander("📋 View Input Summary"):
                st.subheader("Raw Input Values")
                raw_data = {
                    'Age': person_age,
                    'Gender': person_gender,
                    'Education': person_education,
                    'Annual Income': f"${person_income:,}",
                    'Employment Experience': f"{person_emp_exp} years",
                    'Home Ownership': person_home_ownership,
                    'Loan Amount': f"${loan_amnt:,}",
                    'Loan Purpose': loan_intent,
                    'Interest Rate': f"{loan_int_rate}%",
                    'Credit Score': credit_score,
                    'Credit History Length': f"{cb_person_cred_hist_length} years",
                    'Previous Defaults': previous_loan_defaults_on_file
                }
                for key, value in raw_data.items():
                    st.text(f"{key}: {value}")

                st.subheader("Engineered Features (Used by Model)")
                eng_features = {
                    'loan_percent_income': f"{loan_percent_income:.4f}",
                    'income_per_experience': f"{income_per_experience:.2f}",
                    'cred_length_per_age': f"{cred_length_per_age:.4f}",
                    'person_income_log': f"{person_income_log:.4f}",
                    'loan_amnt_log': f"{loan_amnt_log:.4f}",
                    'loan_percent_income_log': f"{loan_percent_income_log:.4f}"
                }
                for key, value in eng_features.items():
                    st.text(f"{key}: {value}")

            # Additional insights
            with st.expander("💡 Key Factors"):
                st.write("**Your Application Summary:**")
                st.write(f"- **Loan Amount:** ${loan_amnt:,}")
                st.write(f"- **Annual Income:** ${person_income:,}")
                st.write(f"- **Loan-to-Income Ratio:** {loan_percent_income * 100:.2f}%")
                st.write(f"- **Income per Year of Experience:** ${income_per_experience:,.2f}")
                st.write(f"- **Credit History / Age Ratio:** {cred_length_per_age:.3f}")
                st.write(f"- **Credit Score:** {credit_score}")
                st.write(f"- **Employment Experience:** {person_emp_exp} years")
                st.write(f"- **Credit History:** {cb_person_cred_hist_length} years")
                st.write(f"- **Previous Defaults:** {previous_loan_defaults_on_file}")

                # Add interpretation
                st.markdown("---")
                st.write("**Risk Assessment:**")
                if loan_percent_income > 0.5:
                    st.error("🔴 Loan amount is >50% of annual income - High risk")
                elif loan_percent_income > 0.3:
                    st.warning("🟡 Loan amount is 30-50% of annual income - Moderate risk")
                else:
                    st.success("🟢 Loan amount is <30% of annual income - Low risk")

                if credit_score >= 720:
                    st.success("🟢 Excellent credit score")
                elif credit_score >= 650:
                    st.info("🟡 Good credit score")
                else:
                    st.warning("🔴 Below average credit score")

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error("Please ensure all model files are in the 'models_save' folder")
            import traceback

            st.code(traceback.format_exc())
    else:
        st.error("Model could not be loaded. Please check if the model file exists.")