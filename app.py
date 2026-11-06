import streamlit as st
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

st.set_page_config(page_title="German Credit Risk Predictor", page_icon="💳", layout="wide")

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
COLUMNS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_commitment", "personal_status",
    "other_parties", "residence_since", "property_magnitude", "age",
    "other_payment_plans", "housing", "existing_credits", "job",
    "num_dependents", "own_telephone", "foreign_worker", "class",
]
FEATURES = COLUMNS[:-1]

NAMES = {
    "checking_status": "Checking account", "duration": "Loan duration (months)",
    "credit_history": "Credit history", "purpose": "Loan purpose",
    "credit_amount": "Credit amount (DM)", "savings_status": "Savings account",
    "employment": "Employment duration", "installment_commitment": "Installment rate (1-4)",
    "personal_status": "Personal status", "other_parties": "Other debtors / guarantors",
    "residence_since": "Years at residence (1-4)", "property_magnitude": "Property",
    "age": "Age", "other_payment_plans": "Other installment plans", "housing": "Housing",
    "existing_credits": "Existing credits at this bank", "job": "Job type",
    "num_dependents": "Number of dependents", "own_telephone": "Telephone",
    "foreign_worker": "Foreign worker",
}

LABELS = {
    "checking_status": {"A11": "Less than 0 DM", "A12": "0 to 200 DM",
                        "A13": "200 DM or more", "A14": "No checking account"},
    "credit_history": {"A30": "No credits taken / all paid back", "A31": "All credits at this bank paid back",
                       "A32": "Existing credits paid back on time", "A33": "Past delays in payment",
                       "A34": "Critical account / other credits elsewhere"},
    "purpose": {"A40": "New car", "A41": "Used car", "A42": "Furniture / equipment",
                "A43": "Radio / TV", "A44": "Domestic appliances", "A45": "Repairs",
                "A46": "Education", "A48": "Retraining", "A49": "Business", "A410": "Other"},
    "savings_status": {"A61": "Less than 100 DM", "A62": "100 to 500 DM", "A63": "500 to 1000 DM",
                       "A64": "1000 DM or more", "A65": "Unknown / no savings"},
    "employment": {"A71": "Unemployed", "A72": "Less than 1 year", "A73": "1 to 4 years",
                   "A74": "4 to 7 years", "A75": "7 years or more"},
    "personal_status": {"A91": "Male: divorced / separated", "A92": "Female: divorced / separated / married",
                        "A93": "Male: single", "A94": "Male: married / widowed"},
    "other_parties": {"A101": "None", "A102": "Co-applicant", "A103": "Guarantor"},
    "property_magnitude": {"A121": "Real estate", "A122": "Life insurance / savings agreement",
                           "A123": "Car or other property", "A124": "Unknown / no property"},
    "other_payment_plans": {"A141": "Bank", "A142": "Stores", "A143": "None"},
    "housing": {"A151": "Rent", "A152": "Own", "A153": "Free housing"},
    "job": {"A171": "Unemployed / unskilled non-resident", "A172": "Unskilled resident",
            "A173": "Skilled employee", "A174": "Management / highly qualified"},
    "own_telephone": {"A191": "None", "A192": "Yes"},
    "foreign_worker": {"A201": "Yes", "A202": "No"},
}


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
    return model, list(X.columns), X_te, y_te


try:
    df = load_data()
except Exception:
    st.error("Could not load the dataset from the UCI website. Please refresh in a minute.")
    st.stop()

model, cols, X_te, y_te = train(df)

# Default value for every feature: median for numeric ranges, most common for categories
BASE = {}
for c in FEATURES:
    if is_numeric_dtype(df[c]) and df[c].nunique() > 5:
        BASE[c] = int(df[c].median())
    else:
        BASE[c] = df[c].mode()[0]


def encode(rows):
    X = pd.get_dummies(pd.DataFrame(rows))
    return X.reindex(columns=cols, fill_value=0)


# ---------------- Sidebar: applicant form ----------------
st.sidebar.title("💳 Applicant details")
inputs = {}

with st.sidebar.expander("Loan details", expanded=True):
    inputs["credit_amount"] = st.slider(NAMES["credit_amount"], 250, 20000, 2300, step=50)
    inputs["duration"] = st.slider(NAMES["duration"], 4, 72, 18)
    inputs["purpose"] = st.selectbox(
        NAMES["purpose"], list(LABELS["purpose"]),
        index=list(LABELS["purpose"]).index(BASE["purpose"]),
        format_func=lambda c: LABELS["purpose"][c])
    inputs["installment_commitment"] = st.select_slider(
        NAMES["installment_commitment"], options=[1, 2, 3, 4], value=int(BASE["installment_commitment"]))

with st.sidebar.expander("Personal details", expanded=True):
    inputs["age"] = st.slider(NAMES["age"], 19, 75, 33)
    inputs["num_dependents"] = st.select_slider(
        NAMES["num_dependents"], options=[1, 2], value=int(BASE["num_dependents"]))
    inputs["residence_since"] = st.select_slider(
        NAMES["residence_since"], options=[1, 2, 3, 4], value=int(BASE["residence_since"]))

with st.sidebar.expander("Financial and other background"):
    for c in ["checking_status", "savings_status", "credit_history", "employment",
              "personal_status", "other_parties", "property_magnitude",
              "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker"]:
        options = list(LABELS[c])
        inputs[c] = st.selectbox(
            NAMES[c], options, index=options.index(BASE[c]),
            format_func=lambda code, c=c: LABELS[c][code])
    inputs["existing_credits"] = st.select_slider(
        NAMES["existing_credits"], options=[1, 2, 3, 4], value=int(BASE["existing_credits"]))

threshold = st.sidebar.slider(
    "Risk threshold", 0.20, 0.80, 0.50, 0.05,
    help="Applicants whose predicted risk is at or above this value are labelled 'Bad credit risk'. "
         "A lower threshold is stricter.")

# ---------------- Main page ----------------
st.title("German Credit Risk Predictor")
st.caption("Change anything in the sidebar and the prediction updates instantly.")

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📊 Model insights", "⚖️ Fairness check", "ℹ️ About"])

# ----- Tab 1: Prediction -----
with tab1:
    rows = [inputs] + [{**inputs, c: BASE[c]} for c in FEATURES]
    probs = model.predict_proba(encode(rows))[:, 1]
    p_bad = float(probs[0])
    is_bad = p_bad >= threshold

    left, right = st.columns([1, 1.3])
    with left:
        if is_bad:
            st.error("### ❌ Bad credit risk")
        else:
            st.success("### ✅ Good credit risk")
        m1, m2 = st.columns(2)
        m1.metric("Risk of default", f"{p_bad:.0%}")
        m2.metric("Chance of good credit", f"{1 - p_bad:.0%}")
        st.progress(min(max(p_bad, 0.0), 1.0), text=f"Risk score {p_bad:.0%} (threshold {threshold:.0%})")
        st.caption("The prediction comes from a Random Forest trained on 1,000 past applicants.")

    with right:
        st.subheader("What is driving this result?")
        st.caption("Each bar shows how much the risk changes compared with a typical applicant, "
                   "in percentage points. Positive means more risk.")
        effects = pd.Series((p_bad - probs[1:]) * 100, index=[NAMES[c] for c in FEATURES])
        top = effects.reindex(effects.abs().sort_values(ascending=False).index).head(8)
        st.bar_chart(top.rename("Effect on risk (points)"))
        up = effects[effects > 0.5].sort_values(ascending=False).head(3)
        down = effects[effects < -0.5].sort_values().head(3)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Raising risk**")
            for k, v in up.items():
                st.write(f"🔺 {k} (+{v:.1f})")
            if up.empty:
                st.write("Nothing notable")
        with c2:
            st.markdown("**Lowering risk**")
            for k, v in down.items():
                st.write(f"🔻 {k} ({v:.1f})")
            if down.empty:
                st.write("Nothing notable")

# ----- Tab 2: Model insights -----
with tab2:
    pred_test = model.predict(X_te)
    proba_test = model.predict_proba(X_te)[:, 1]
    a, b, c = st.columns(3)
    a.metric("Accuracy", f"{accuracy_score(y_te, pred_test):.1%}")
    b.metric("ROC-AUC", f"{roc_auc_score(y_te, proba_test):.3f}")
    c.metric("F1 score", f"{f1_score(y_te, pred_test):.3f}")
    st.caption("Measured on a held-out test set of 200 applicants (20% of the data).")

    st.subheader("Most important features")
    imp = pd.Series(model.feature_importances_, index=cols)
    grouped = {}
    for col_name, val in imp.items():
        orig = next(f for f in FEATURES if col_name == f or col_name.startswith(f + "_"))
        grouped[NAMES[orig]] = grouped.get(NAMES[orig], 0) + val
    imp_df = pd.Series(grouped).sort_values(ascending=False).head(10)
    st.bar_chart(imp_df.rename("Importance"))

    with st.expander("Look at the dataset"):
        st.write(f"{len(df)} applicants, {len(FEATURES)} features. "
                 f"{(df['class'] == 1).mean():.0%} are labelled bad credit risk.")
        st.dataframe(df.head(50), use_container_width=True)

# ----- Tab 3: Fairness check -----
with tab3:
    st.subheader("Does the model treat groups differently?")
    st.write("This compares how often the model approves applicants (predicts good credit) "
             "and how accurate it is, for different groups in the test set.")
    test = df.loc[X_te.index].copy()
    test["approved"] = model.predict(X_te) == 0
    test["correct"] = model.predict(X_te) == y_te
    test["Age group"] = pd.cut(test["age"], [0, 25, 40, 100], labels=["Under 25", "25 to 40", "Over 40"])
    test["Foreign worker"] = test["foreign_worker"].map(LABELS["foreign_worker"])
    test["Personal status"] = test["personal_status"].map(LABELS["personal_status"])

    for group in ["Age group", "Foreign worker", "Personal status"]:
        res = test.groupby(group, observed=True).agg(
            Applicants=("approved", "size"),
            ApprovalRate=("approved", "mean"),
            Accuracy=("correct", "mean"),
        )
        res["Approval rate (%)"] = (res["ApprovalRate"] * 100).round(1)
        res["Accuracy (%)"] = (res["Accuracy"] * 100).round(1)
        st.markdown(f"**By {group.lower()}**")
        cl, cr = st.columns([1, 1])
        cl.dataframe(res[["Applicants", "Approval rate (%)", "Accuracy (%)"]], use_container_width=True)
        cr.bar_chart(res["Approval rate (%)"])
    st.info("Large gaps between groups can signal bias. Groups here are small, so treat the numbers "
            "as indicative. See the notebook for the full Fairlearn analysis.")

# ----- Tab 4: About -----
with tab4:
    st.markdown("""
**Project:** Credit risk prediction with fairness analysis on the German Credit dataset (UCI Statlog).

**Model:** Random Forest classifier trained on 1,000 applicants with 20 features.

**How to use it:** Fill in the applicant details in the sidebar. The prediction, risk score and
explanation update automatically. Use the risk threshold slider to make the decision stricter or looser.

**Note:** This is a demonstration project, not a tool for real lending decisions.
""")
