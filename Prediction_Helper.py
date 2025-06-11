import pandas as pd
from joblib import load

model_young = load("artifacts/model_young.joblib")
model_rest = load(r"C:\Users\venka\OneDrive\Desktop\pythonpractice\statistics\App\artifacts\model_rest.joblib")



scaler_young = load(r"C:\Users\venka\OneDrive\Desktop\pythonpractice\statistics\App\artifacts\scaler_young.joblib")
scaler_rest = load(r"C:\Users\venka\OneDrive\Desktop\pythonpractice\statistics\App\artifacts\scaler_rest.joblib")
cols_to_scale = [
    'Age',
    'Number Of Dependants',
    'Income_Lakhs',
    'Insurance_Plan',
    'Genetical_Risk'
]

# Expected final columns (after one-hot encoding with drop_first=True)
expected_columns = [
    'Age', 'Number Of Dependants', 'Income_Lakhs', 'Insurance_Plan', 'Genetical_Risk',
    'normalized_risk_score',
    'Gender_Male',
    'Region_Northwest', 'Region_Southeast', 'Region_Southwest',
    'Marital_status_Unmarried',
    'BMI_Category_Obesity', 'BMI_Category_Overweight', 'BMI_Category_Underweight',
    'Smoking_Status_Occasional', 'Smoking_Status_Regular',
    'Employment_Status_Salaried', 'Employment_Status_Self-Employed'
]

# Function to calculate normalized risk score
def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }
    diseases = medical_history.lower().split(" & ")
    total_score = sum(risk_scores.get(d, 0) for d in diseases)
    max_score = 14  # max = heart disease + diabetes/high BP
    return (total_score - 0) / (max_score - 0)

# Function to handle scaling
def handle_scaling(df, age):
    df_scaled = df.copy()

    # Choose appropriate scaler based on age
    scaler_object = scaler_young if age <= 25 else scaler_rest
    scaler = scaler_object['scaler']
    scale_cols = scaler_object['cols_to_scale']

    # Add 'Income_Level' temporarily if required by scaler
    if 'Income_Level' in scale_cols and 'Income_Level' not in df_scaled.columns:
        df_scaled['Income_Level'] = 0

    # Perform scaling
    df_scaled[scale_cols] = scaler.transform(df_scaled[scale_cols])

    # Drop 'Income_Level' if it was added temporarily
    if 'Income_Level' in df_scaled.columns and 'Income_Level' not in cols_to_scale:
        df_scaled.drop('Income_Level', axis=1, inplace=True)

    return df_scaled





# Main preprocessing function
def preprocess_input(input_dict):
    # Start with a dataframe of zeros
    df = pd.DataFrame(0, index=[0], columns=expected_columns)

    # Direct numeric assignments
    df.at[0, 'Age'] = input_dict['Age']
    df.at[0, 'Number Of Dependants'] = input_dict['Number Of Dependants']
    df.at[0, 'Income_Lakhs'] = input_dict['Income in Lakhs']
    df.at[0, 'Insurance_Plan'] = {'Bronze': 1, 'Silver': 2, 'Gold': 3}.get(input_dict['Insurance Plan'], 1)
    df.at[0, 'Genetical_Risk'] = input_dict['Genetical Risk']
    df.at[0, 'normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])

    # Gender
    if input_dict['Gender'] == 'Male':
        df.at[0, 'Gender_Male'] = 1

    # Region
    region_col = f"Region_{input_dict['Region']}"
    if region_col in df.columns:
        df.at[0, region_col] = 1

    # Marital Status
    if input_dict['Marital Status'] == 'Unmarried':
        df.at[0, 'Marital_status_Unmarried'] = 1

    # BMI Category
    bmi_col = f"BMI_Category_{input_dict['BMI Category']}"
    if bmi_col in df.columns:
        df.at[0, bmi_col] = 1

    # Smoking Status
    smoke_col = f"Smoking_Status_{input_dict['Smoking Status']}"
    if smoke_col in df.columns:
        df.at[0, smoke_col] = 1

    # Employment Status
    emp_col = f"Employment_Status_{input_dict['Employment Status']}"
    if emp_col in df.columns:
        df.at[0, emp_col] = 1

    # Apply scaling
    df = handle_scaling(df, input_dict['Age'])




    return df

# Prediction function
def predict(input_dict):
    df_input = preprocess_input(input_dict)
    if input_dict['Age'] <= 25:
        pred = model_young.predict(df_input)
    else:
        pred = model_rest.predict(df_input)
    return int(pred[0])