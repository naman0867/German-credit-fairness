import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
COLUMNS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_commitment", "personal_status",
    "other_parties", "residence_since", "property_magnitude", "age",
    "other_payment_plans", "housing", "existing_credits", "job",
    "num_dependents", "own_telephone", "foreign_worker", "class",
]

@st.cache_data
def load_data():
    df = pd.read_csv(URL, sep=" ", header=None, names=COLUMNS)
    df["class"] = df["class"].map({1: 0, 2: 1})  # 0 = good, 1 = bad
    return df

@st.cache_resource
def train(df):
    X = pd.get_dummies(df.drop(columns="class"))
    y = df["class"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_tr, y_tr)
    return model, list(X.columns), accuracy_score(y_te, model.predict(X_te))

df = load_data()
model, cols, acc = train(df)

st.title("German Credit Risk Predictor")
st.write("Enter applicant details to predict whether the credit risk is good or bad.")
st.caption(f"Model: Random Forest | Test accuracy: {acc:.2%}")

inputs = {}
left, right = st.columns(2)
for i, col in enumerate(COLUMNS[:-1]):
    box = left if i % 2 == 0 else right
    label = col.replace("_", " ").title()
    if df[col].dtype == "object" or df[col].nunique() <= 5:
        options = sorted(df[col].unique().tolist())
        inputs[col] = box.selectbox(label, options)
    else:
        lo = int(df[col].min())
        hi = int(df[col].max())
        mid = int(df[col].median())
        inputs[col] = box.number_input(label, min_value=lo, max_value=hi, value=mid, step=1)

if st.button("Predict"):
    row = pd.get_dummies(pd.DataFrame([inputs])).reindex(columns=cols, fill_value=0)
    prob_bad = model.predict_proba(row)[0][1]
    if prob_bad >= 0.5:
        st.error(f"Bad credit risk (probability {prob_bad:.0%})")
    else:
        st.success(f"Good credit risk (probability {1 - prob_bad:.0%})")
