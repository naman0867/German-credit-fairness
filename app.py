import math
import streamlit as st
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

st.set_page_config(page_title="FairLend | Smart Credit Decisions", page_icon="💳", layout="wide")

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
    "checking_status": "Checking account", "duration": "Loan term",
    "credit_history": "Credit history", "purpose": "Loan purpose",
    "credit_amount": "Loan amount", "savings_status": "Savings account",
    "employment": "Employment length", "installment_commitment": "Installment rate",
    "personal_status": "Personal status", "other_parties": "Other debtors / guarantors",
    "residence_since": "Years at residence", "property_magnitude": "Property owned",
    "age": "Age", "other_payment_plans": "Other installment plans", "housing": "Housing",
    "existing_credits": "Existing credits", "job": "Job type",
    "num_dependents": "Dependents", "own_telephone": "Telephone",
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

GREEN, AMBER, RED = "#12B76A", "#F79009", "#F04438"

# ---------------- Styling ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
footer {visibility: hidden;}
.block-container { padding-top: 1.2rem; max-width: 1250px; }
.hero { background: linear-gradient(120deg, #0B1F4B 0%, #1D4ED8 60%, #3B82F6 100%);
        border-radius: 20px; padding: 30px 36px; color: white; margin-bottom: 18px; }
.hero .brand { font-weight: 800; font-size: 20px; letter-spacing: .3px; opacity: .95; }
.hero h1 { color: white; font-size: 34px; margin: 10px 0 6px 0; font-weight: 800; padding: 0; }
.hero p { color: #DCE7FF; font-size: 16px; margin: 0 0 14px 0; }
.pill { display: inline-block; background: rgba(255,255,255,.16); border-radius: 999px;
        padding: 5px 14px; font-size: 13px; margin-right: 8px; margin-top: 4px; }
.card { background: white; border: 1px solid #E5E7EB; border-radius: 18px; padding: 22px;
        box-shadow: 0 4px 18px rgba(16,24,40,.06); margin-bottom: 16px; }
.dec { border-radius: 18px; padding: 20px 24px; margin-bottom: 14px; color: white; }
.dec h2 { color: white; margin: 0; font-size: 28px; font-weight: 800; padding: 0; }
.dec p { margin: 4px 0 0 0; opacity: .95; font-size: 15px; }
.kpis { display: flex; gap: 12px; margin: 6px 0 4px 0; }
.kpi { flex: 1; background: #F8FAFC; border: 1px solid #E5E7EB; border-radius: 14px; padding: 12px 14px; }
.kpi .l { font-size: 12px; color: #667085; font-weight: 600; text-transform: uppercase; letter-spacing: .4px; }
.kpi .v { font-size: 22px; font-weight: 800; color: #0B1F4B; margin-top: 2px; }
.factor { display: flex; align-items: center; margin: 9px 0; font-size: 14px; }
.factor .n { width: 46%; color: #344054; }
.factor .n small { color: #667085; display: block; font-size: 12px; }
.factor .b { flex: 1; background: #F2F4F7; border-radius: 8px; height: 12px; margin: 0 10px; overflow: hidden; }
.factor .b div { height: 12px; border-radius: 8px; }
.factor .t { width: 62px; text-align: right; font-weight: 700; font-size: 13px; }
.sec { font-weight: 700; font-size: 17px; color: #0B1F4B; margin-bottom: 8px; }
.tip { background: #EFF6FF; border: 1px solid #BFDBFE; color: #1E3A8A; border-radius: 12px;
       padding: 12px 14px; font-size: 14px; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)


# ---------------- Data and model ----------------
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

BASE = {}
for c in FEATURES:
    if is_numeric_dtype(df[c]) and df[c].nunique() > 5:
        BASE[c] = int(df[c].median())
    else:
        BASE[c] = df[c].mode()[0]


def encode(rows):
    return pd.get_dummies(pd.DataFrame(rows)).reindex(columns=cols, fill_value=0)


def emi(principal, apr, months):
    r = apr / 1200
    if r == 0:
        return principal / months
    return principal * r / (1 - (1 + r) ** -months)


def score_from(p):
    return int(round(850 - p * 550))


def score_band(s):
    if s >= 740: return "Excellent", GREEN
    if s >= 670: return "Good", "#84CC16"
    if s >= 580: return "Fair", AMBER
    return "Poor", RED


def gauge(score):
    def pt(f, r=80):
        a = math.pi * (1 - f)
        return 100 + r * math.cos(a), 100 - r * math.sin(a)
    segs = [(0, .25, RED), (.25, .5, AMBER), (.5, .75, "#84CC16"), (.75, 1, GREEN)]
    paths = ""
    for f0, f1, col in segs:
        x0, y0 = pt(f0)
        x1, y1 = pt(f1)
        paths += (f'<path d="M {x0:.1f} {y0:.1f} A 80 80 0 0 1 {x1:.1f} {y1:.1f}" '
                  f'stroke="{col}" stroke-width="14" fill="none"/>')
    f = (score - 300) / 550
    nx, ny = pt(f, 62)
    label, col = score_band(score)
    return (f'<div style="text-align:center"><svg viewBox="0 0 200 125" width="100%" style="max-width:340px">'
            f'{paths}<line x1="100" y1="100" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#0B1F4B" stroke-width="4" stroke-linecap="round"/>'
            f'<circle cx="100" cy="100" r="7" fill="#0B1F4B"/>'
            f'<text x="18" y="118" font-size="9" fill="#667085">300</text>'
            f'<text x="165" y="118" font-size="9" fill="#667085">850</text></svg>'
            f'<div style="font-size:44px;font-weight:800;color:#0B1F4B;line-height:1">{score}</div>'
            f'<div style="font-weight:700;color:{col};font-size:16px;margin-top:2px">{label} credit score</div></div>')


# ---------------- Header ----------------
st.markdown("""
<div class="hero">
<div class="brand">💳 FairLend</div>
<h1>Smart, fair credit decisions in seconds</h1>
<p>Tell us about your loan and get an instant, explainable decision powered by machine learning.</p>
<span class="pill">⚡ Instant decision</span><span class="pill">🔍 Fully explainable</span>
<span class="pill">⚖️ Bias-checked</span><span class="pill">🔒 No data stored</span>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1.15], gap="large")
inputs = {}

# ---------------- Left: application form ----------------
with left:
    with st.container(border=True):
        st.markdown('<div class="sec">1. Your loan request</div>', unsafe_allow_html=True)
        inputs["credit_amount"] = st.slider("Loan amount (DM)", 250, 20000, 2300, step=50)
        c1, c2 = st.columns(2)
        inputs["duration"] = c1.slider("Loan term (months)", 4, 72, 18)
        inputs["installment_commitment"] = c2.select_slider(
            "Installment rate (% of income, 1-4)", options=[1, 2, 3, 4],
            value=int(BASE["installment_commitment"]))
        inputs["purpose"] = st.selectbox(
            "Loan purpose", list(LABELS["purpose"]),
            index=list(LABELS["purpose"]).index(BASE["purpose"]),
            format_func=lambda c: LABELS["purpose"][c])

    with st.container(border=True):
        st.markdown('<div class="sec">2. About you</div>', unsafe_allow_html=True)
        inputs["age"] = st.slider("Age", 19, 75, 33)
        c1, c2 = st.columns(2)
        inputs["num_dependents"] = c1.select_slider(
            "Dependents", options=[1, 2], value=int(BASE["num_dependents"]))
        inputs["residence_since"] = c2.select_slider(
            "Years at residence", options=[1, 2, 3, 4], value=int(BASE["residence_since"]))
        for c in ["employment", "job", "housing", "personal_status"]:
            opts = list(LABELS[c])
            inputs[c] = st.selectbox(NAMES[c], opts, index=opts.index(BASE[c]),
                                     format_func=lambda code, c=c: LABELS[c][code])

    with st.container(border=True):
        st.markdown('<div class="sec">3. Financial background</div>', unsafe_allow_html=True)
        for c in ["checking_status", "savings_status", "credit_history", "property_magnitude",
                  "other_parties", "other_payment_plans", "own_telephone", "foreign_worker"]:
            opts = list(LABELS[c])
            inputs[c] = st.selectbox(NAMES[c], opts, index=opts.index(BASE[c]),
                                     format_func=lambda code, c=c: LABELS[c][code])
        inputs["existing_credits"] = st.select_slider(
            "Existing credits at this bank", options=[1, 2, 3, 4], value=int(BASE["existing_credits"]))

    with st.expander("⚙️ Lender policy settings"):
        threshold = st.slider("Maximum acceptable default risk", 0.30, 0.70, 0.50, 0.05,
                              help="Applicants at or above this risk are declined. "
                                   "Applicants within 15 points below it go to manual review.")

# ---------------- Prediction ----------------
rows = [inputs] + [{**inputs, c: BASE[c]} for c in FEATURES]
probs = model.predict_proba(encode(rows))[:, 1]
p_bad = float(probs[0])
score = score_from(p_bad)
review_lo = threshold - 0.15

if p_bad < review_lo:
    status, color, headline = "approved", GREEN, "✅ Congratulations, you are pre-approved"
    sub = "Your profile meets our lending criteria."
elif p_bad < threshold:
    status, color, headline = "review", AMBER, "🕒 Your application needs a manual review"
    sub = "You are close to our approval criteria. An analyst will take a second look."
else:
    status, color, headline = "declined", RED, "❌ We cannot approve this request right now"
    sub = "The estimated risk is above our lending limit. See how you can improve below."

# APR rises with risk
apr = round(8 + p_bad * 22, 1)
monthly = emi(inputs["credit_amount"], apr, inputs["duration"])
total_pay = monthly * inputs["duration"]

# ---------------- Right: decision ----------------
with right:
    st.markdown(f'<div class="dec" style="background:{color}"><h2>{headline}</h2><p>{sub}</p></div>',
                unsafe_allow_html=True)

    g1, g2 = st.columns([1.2, 1])
    with g1:
        st.markdown(f'<div class="card">{gauge(score)}</div>', unsafe_allow_html=True)
    with g2:
        st.markdown(
            f'<div class="card"><div class="sec">Risk summary</div>'
            f'<div class="kpi"><div class="l">Default risk</div><div class="v" style="color:{color}">{p_bad:.0%}</div></div>'
            f'<div style="height:8px"></div>'
            f'<div class="kpi"><div class="l">Repayment chance</div><div class="v">{1 - p_bad:.0%}</div></div>'
            f'<div style="height:8px"></div>'
            f'<div class="kpi"><div class="l">Policy limit</div><div class="v">{threshold:.0%}</div></div></div>',
            unsafe_allow_html=True)

    if status != "declined":
        label = "Your indicative offer" if status == "approved" else "Indicative offer if approved"
        st.markdown(
            f'<div class="card"><div class="sec">{label}</div><div class="kpis">'
            f'<div class="kpi"><div class="l">Loan amount</div><div class="v">{inputs["credit_amount"]:,} DM</div></div>'
            f'<div class="kpi"><div class="l">Term</div><div class="v">{inputs["duration"]} mo</div></div>'
            f'<div class="kpi"><div class="l">APR</div><div class="v">{apr}%</div></div>'
            f'<div class="kpi"><div class="l">Monthly</div><div class="v">{monthly:,.0f} DM</div></div>'
            f'</div><div style="font-size:13px;color:#667085;margin-top:8px">'
            f'Total repayment: {total_pay:,.0f} DM (interest {total_pay - inputs["credit_amount"]:,.0f} DM). '
            f'Interest rate is risk-based, so a lower-risk profile gets a cheaper loan.</div></div>',
            unsafe_allow_html=True)

    # Key factors
    effects = pd.Series((p_bad - probs[1:]) * 100, index=FEATURES)
    top = effects.reindex(effects.abs().sort_values(ascending=False).index).head(6)
    mx = max(float(top.abs().max()), 1.0)
    html = '<div class="card"><div class="sec">Why this decision?</div>'
    html += '<div style="font-size:13px;color:#667085;margin-bottom:6px">Change in default risk versus a typical applicant (percentage points).</div>'
    for c, v in top.items():
        val = LABELS.get(c, {}).get(inputs[c], inputs[c])
        if c == "credit_amount": val = f"{inputs[c]:,} DM"
        if c == "duration": val = f"{inputs[c]} months"
        width = max(abs(v) / mx * 100, 3)
        col = RED if v > 0 else GREEN
        sign = "+" if v > 0 else ""
        html += (f'<div class="factor"><div class="n">{NAMES[c]}<small>{val}</small></div>'
                 f'<div class="b"><div style="width:{width:.0f}%;background:{col}"></div></div>'
                 f'<div class="t" style="color:{col}">{sign}{v:.1f}</div></div>')
    html += '<div style="font-size:12px;color:#667085">Red raises risk, green lowers it.</div></div>'
    st.markdown(html, unsafe_allow_html=True)

    # What-if suggestion
    if status != "approved":
        smaller = {**inputs, "credit_amount": max(250, int(inputs["credit_amount"] * 0.7))}
        shorter = {**inputs, "duration": max(4, int(inputs["duration"] * 0.7))}
        p_small = float(model.predict_proba(encode([smaller]))[0, 1])
        p_short = float(model.predict_proba(encode([shorter]))[0, 1])
        st.markdown(
            f'<div class="tip">💡 <b>How to improve your chances</b><br>'
            f'• Borrow {smaller["credit_amount"]:,} DM instead (30% less): risk becomes <b>{p_small:.0%}</b><br>'
            f'• Choose a {shorter["duration"]}-month term instead: risk becomes <b>{p_short:.0%}</b><br>'
            f'• Add a co-applicant or guarantor, or build up your savings balance.</div>',
            unsafe_allow_html=True)

# ---------------- Bottom tabs ----------------
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📊 Model insights", "⚖️ Fairness check", "ℹ️ About"])

with tab1:
    pred_test = model.predict(X_te)
    proba_test = model.predict_proba(X_te)[:, 1]
    a, b, c = st.columns(3)
    a.metric("Accuracy", f"{accuracy_score(y_te, pred_test):.1%}")
    b.metric("ROC-AUC", f"{roc_auc_score(y_te, proba_test):.3f}")
    c.metric("F1 score", f"{f1_score(y_te, pred_test):.3f}")
    st.caption("Measured on a held-out test set of 200 applicants (20% of the data).")

    st.subheader("What the model looks at most")
    imp = pd.Series(model.feature_importances_, index=cols)
    ordered = sorted(FEATURES, key=len, reverse=True)
    grouped = {}
    for col_name, val in imp.items():
        orig = next(f for f in ordered if col_name == f or col_name.startswith(f + "_"))
        grouped[NAMES[orig]] = grouped.get(NAMES[orig], 0) + val
    st.bar_chart(pd.Series(grouped).sort_values(ascending=False).head(10).rename("Importance"))

    with st.expander("Look at the training data"):
        st.write(f"{len(df)} applicants, {len(FEATURES)} features. "
                 f"{(df['class'] == 1).mean():.0%} are labelled bad credit risk.")
        st.dataframe(df.head(50), use_container_width=True)

with tab2:
    st.subheader("Does the model treat groups differently?")
    st.write("Approval rate and accuracy for different groups in the test set. "
             "Large gaps between groups can signal bias.")
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
        cl, cr = st.columns(2)
        cl.dataframe(res[["Applicants", "Approval rate (%)", "Accuracy (%)"]], use_container_width=True)
        cr.bar_chart(res["Approval rate (%)"])
    st.info("Groups are small, so treat the numbers as indicative. "
            "See the notebook for the full Fairlearn analysis.")

with tab3:
    st.markdown("""
**FairLend** is a demonstration project: credit risk prediction with fairness analysis on the
German Credit dataset (UCI Statlog, 1,000 applicants, 20 features).

**Model:** Random Forest classifier. The credit score (300 to 850) and the APR are simple
transformations of the model's predicted default probability.

**Decision rule:** approved if risk is well below the policy limit, manual review if close to it,
declined if above it. You can change the limit under Lender policy settings.

**Note:** This is for demonstration only. It must not be used for real lending decisions.
Amounts are in Deutsche Marks (DM) because that is the currency in the original dataset.
""")
