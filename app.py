import math
import streamlit as st
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

st.set_page_config(page_title="FairLend | Instant Personal Loan", page_icon="💳", layout="centered")

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
COLUMNS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_commitment", "personal_status",
    "other_parties", "residence_since", "property_magnitude", "age",
    "other_payment_plans", "housing", "existing_credits", "job",
    "num_dependents", "own_telephone", "foreign_worker", "class",
]
FEATURES = COLUMNS[:-1]
RATE = 40  # demo conversion: 1 DM = 40 INR

NAMES = {
    "checking_status": "Checking account", "duration": "Loan tenure",
    "credit_history": "Credit history", "purpose": "Loan purpose",
    "credit_amount": "Loan amount", "savings_status": "Savings",
    "employment": "Employment length", "installment_commitment": "EMI burden",
    "personal_status": "Personal status", "other_parties": "Guarantor / co-applicant",
    "residence_since": "Years at residence", "property_magnitude": "Property owned",
    "age": "Age", "other_payment_plans": "Other EMIs", "housing": "Housing",
    "existing_credits": "Existing credits", "job": "Job type",
    "num_dependents": "Dependents", "own_telephone": "Telephone",
    "foreign_worker": "Foreign worker",
}

LABELS = {
    "checking_status": {"A11": "Balance below 0", "A12": "Low balance",
                        "A13": "Healthy balance", "A14": "No checking account"},
    "credit_history": {"A30": "No credits taken / all paid back", "A31": "All credits at this bank paid back",
                       "A32": "Existing credits paid on time", "A33": "Past delays in payment",
                       "A34": "Critical account / other credits elsewhere"},
    "purpose": {"A40": "New car", "A41": "Used car", "A42": "Furniture / equipment",
                "A43": "Radio / TV", "A44": "Domestic appliances", "A45": "Repairs",
                "A46": "Education", "A48": "Retraining", "A49": "Business", "A410": "Other"},
    "savings_status": {"A61": "Very low savings", "A62": "Low savings", "A63": "Moderate savings",
                       "A64": "High savings", "A65": "Unknown / no savings"},
    "employment": {"A71": "Unemployed", "A72": "Less than 1 year", "A73": "1 to 4 years",
                   "A74": "4 to 7 years", "A75": "7 years or more"},
    "personal_status": {"A91": "Male: divorced / separated", "A92": "Female: divorced / separated / married",
                        "A93": "Male: single", "A94": "Male: married / widowed"},
    "other_parties": {"A101": "None", "A102": "Co-applicant", "A103": "Guarantor"},
    "property_magnitude": {"A121": "Real estate", "A122": "Life insurance / savings plan",
                           "A123": "Car or other property", "A124": "Unknown / no property"},
    "other_payment_plans": {"A141": "Bank", "A142": "Stores", "A143": "None"},
    "housing": {"A151": "Rent", "A152": "Own", "A153": "Free housing"},
    "job": {"A171": "Unemployed / unskilled non-resident", "A172": "Unskilled resident",
            "A173": "Skilled employee", "A174": "Management / highly qualified"},
    "own_telephone": {"A191": "No", "A192": "Yes"},
    "foreign_worker": {"A201": "Yes", "A202": "No"},
}

NAVY, CYAN = "#002970", "#00BAF2"
GREEN, AMBER, RED = "#21C17A", "#FFAB00", "#FF5C5C"


def inr(x):
    x = int(round(x))
    s = str(abs(x))
    if len(s) > 3:
        last3, rest, groups = s[-3:], s[:-3], []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        s = ",".join(groups + [last3])
    return ("-" if x < 0 else "") + "₹" + s


# ---------------- Styling ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.stApp { background: #F2F6FC; }
footer { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 110px; max-width: 520px; }
[data-testid="stVerticalBlockBorderWrapper"] { background: white; border-radius: 20px;
  border: 1px solid #E6ECF5 !important; box-shadow: 0 4px 16px rgba(0,41,112,.06); }
.topbar { display:flex; align-items:center; justify-content:space-between; margin-bottom:14px; }
.logo { font-size:26px; font-weight:800; color:#002970; letter-spacing:-.5px; }
.logo span { color:#00BAF2; }
.avatar { width:38px; height:38px; border-radius:50%; background:#00BAF2; color:white;
  display:flex; align-items:center; justify-content:center; font-weight:700; }
.hero { background: linear-gradient(135deg,#002970 0%,#0B4FD6 55%,#00BAF2 100%); border-radius:24px;
  padding:24px; color:white; margin-bottom:14px; position:relative; overflow:hidden; }
.hero:after { content:""; position:absolute; right:-40px; top:-40px; width:160px; height:160px;
  border-radius:50%; background:rgba(255,255,255,.10); }
.hero .t { font-size:13px; opacity:.85; font-weight:500; }
.hero .a { font-size:44px; font-weight:800; margin:4px 0 2px 0; letter-spacing:-1px; }
.hero .s { font-size:13px; opacity:.9; }
.chips span { display:inline-block; background:rgba(255,255,255,.18); border-radius:999px;
  padding:4px 12px; font-size:12px; margin:10px 6px 0 0; }
.sec { font-weight:700; font-size:16px; color:#002970; margin-bottom:4px; }
.dec { border-radius:22px; padding:20px 22px; color:white; margin:6px 0 12px 0; }
.dec h2 { color:white; margin:0; padding:0; font-size:22px; font-weight:700; }
.dec p { margin:6px 0 0 0; font-size:14px; opacity:.95; }
.grid { display:flex; gap:10px; margin-top:10px; }
.kpi { flex:1; background:#F5F9FF; border:1px solid #E1EAF7; border-radius:14px; padding:10px 12px; }
.kpi .l { font-size:11px; color:#6B7A99; font-weight:600; text-transform:uppercase; letter-spacing:.4px; }
.kpi .v { font-size:17px; font-weight:700; color:#002970; margin-top:2px; }
.emi { text-align:center; padding:6px 0 2px 0; }
.emi .l { color:#6B7A99; font-size:13px; font-weight:500; }
.emi .v { color:#002970; font-size:38px; font-weight:800; letter-spacing:-1px; }
.row { display:flex; justify-content:space-between; font-size:14px; padding:7px 0;
  border-bottom:1px dashed #E1EAF7; color:#344054; }
.row b { color:#002970; }
.factor { display:flex; align-items:center; margin:10px 0; font-size:14px; }
.factor .n { width:44%; color:#344054; font-weight:500; }
.factor .n small { color:#6B7A99; display:block; font-size:12px; font-weight:400; }
.factor .b { flex:1; background:#EEF3FA; border-radius:8px; height:10px; margin:0 10px; overflow:hidden; }
.factor .b div { height:10px; border-radius:8px; }
.factor .t { width:56px; text-align:right; font-weight:700; font-size:13px; }
.tip { background:#E8F8FE; border:1px solid #B9ECFB; color:#003B73; border-radius:14px;
  padding:12px 14px; font-size:14px; margin-top:6px; }
.success { text-align:center; background:white; border-radius:22px; padding:28px 18px;
  border:1px solid #CDEFE0; box-shadow:0 8px 24px rgba(33,193,122,.18); }
.success .tick { width:76px; height:76px; border-radius:50%; background:#21C17A; color:white;
  font-size:42px; line-height:76px; margin:0 auto 12px auto; }
.success h3 { color:#002970; margin:0 0 4px 0; padding:0; font-size:22px; }
.success p { color:#6B7A99; margin:0; font-size:14px; }
.nav { position:fixed; left:0; right:0; bottom:0; background:white; border-top:1px solid #E6ECF5;
  display:flex; justify-content:center; gap:26px; padding:8px 0 10px 0; z-index:999; }
.nav div { text-align:center; font-size:11px; color:#6B7A99; width:64px; }
.nav div.on { color:#002970; font-weight:700; }
.nav div i { font-style:normal; font-size:20px; display:block; }
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
    return principal / months if r == 0 else principal * r / (1 - (1 + r) ** -months)


def score_from(p):
    return int(round(850 - p * 550))


def score_band(s):
    if s >= 740: return "Excellent", GREEN
    if s >= 670: return "Good", "#7ED321"
    if s >= 580: return "Fair", AMBER
    return "Poor", RED


def gauge(score):
    def pt(f, r=80):
        a = math.pi * (1 - f)
        return 100 + r * math.cos(a), 100 - r * math.sin(a)
    paths = ""
    for f0, f1, col in [(0, .25, RED), (.25, .5, AMBER), (.5, .75, "#7ED321"), (.75, 1, GREEN)]:
        x0, y0 = pt(f0)
        x1, y1 = pt(f1)
        paths += (f'<path d="M {x0:.1f} {y0:.1f} A 80 80 0 0 1 {x1:.1f} {y1:.1f}" '
                  f'stroke="{col}" stroke-width="14" fill="none"/>')
    nx, ny = pt((score - 300) / 550, 62)
    label, col = score_band(score)
    return (f'<div style="text-align:center"><svg viewBox="0 0 200 118" width="100%" style="max-width:260px">'
            f'{paths}<line x1="100" y1="100" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#002970" stroke-width="4" stroke-linecap="round"/>'
            f'<circle cx="100" cy="100" r="7" fill="#002970"/>'
            f'<text x="16" y="116" font-size="9" fill="#6B7A99">300</text>'
            f'<text x="166" y="116" font-size="9" fill="#6B7A99">850</text></svg>'
            f'<div style="font-size:40px;font-weight:800;color:#002970;line-height:1">{score}</div>'
            f'<div style="font-weight:600;color:{col};font-size:15px">{label} credit score</div></div>')


# ---------------- Header ----------------
st.markdown('<div class="topbar"><div class="logo">Fair<span>Lend</span></div><div class="avatar">N</div></div>',
            unsafe_allow_html=True)

inputs = {}

# ---------------- Loan selection ----------------
with st.container(border=True):
    st.markdown('<div class="sec">How much do you need?</div>', unsafe_allow_html=True)
    amount_inr = st.slider("Loan amount", 10000, 800000, 100000, step=5000,
                           format="₹%d", label_visibility="collapsed")
    inputs["credit_amount"] = max(250, int(amount_inr / RATE))
    st.markdown("**Choose tenure**")
    inputs["duration"] = st.radio("Tenure", [6, 12, 18, 24, 36, 48], index=2, horizontal=True,
                                  format_func=lambda m: f"{m} mo", label_visibility="collapsed")
    inputs["purpose"] = st.selectbox(
        "What is it for?", list(LABELS["purpose"]),
        index=list(LABELS["purpose"]).index(BASE["purpose"]),
        format_func=lambda c: LABELS["purpose"][c])

with st.expander("👤 Your details", expanded=False):
    inputs["age"] = st.slider("Age", 19, 75, 33)
    c1, c2 = st.columns(2)
    inputs["num_dependents"] = c1.select_slider("Dependents", options=[1, 2], value=int(BASE["num_dependents"]))
    inputs["residence_since"] = c2.select_slider("Years at residence", options=[1, 2, 3, 4],
                                                 value=int(BASE["residence_since"]))
    for c in ["employment", "job", "housing", "personal_status"]:
        opts = list(LABELS[c])
        inputs[c] = st.selectbox(NAMES[c], opts, index=opts.index(BASE[c]),
                                 format_func=lambda code, c=c: LABELS[c][code])

with st.expander("🏦 Financial background", expanded=False):
    inputs["installment_commitment"] = st.select_slider(
        "EMI burden (share of income, 1 low to 4 high)", options=[1, 2, 3, 4],
        value=int(BASE["installment_commitment"]))
    inputs["existing_credits"] = st.select_slider(
        "Existing credits", options=[1, 2, 3, 4], value=int(BASE["existing_credits"]))
    for c in ["checking_status", "savings_status", "credit_history", "property_magnitude",
              "other_parties", "other_payment_plans", "own_telephone", "foreign_worker"]:
        opts = list(LABELS[c])
        inputs[c] = st.selectbox(NAMES[c], opts, index=opts.index(BASE[c]),
                                 format_func=lambda code, c=c: LABELS[c][code])

with st.expander("⚙️ Lender policy"):
    threshold = st.slider("Maximum acceptable default risk", 0.30, 0.70, 0.50, 0.05,
                          help="At or above this risk the loan is declined. "
                               "Within 15 points below it goes to manual review.")

# ---------------- Prediction ----------------
rows = [inputs] + [{**inputs, c: BASE[c]} for c in FEATURES]
probs = model.predict_proba(encode(rows))[:, 1]
p_bad = float(probs[0])
score = score_from(p_bad)

if p_bad < threshold - 0.15:
    status, color = "approved", GREEN
    headline, sub = "🎉 You're pre-approved!", "Your profile meets our lending criteria. Money in minutes."
elif p_bad < threshold:
    status, color = "review", AMBER
    headline, sub = "🕒 Almost there", "You're close to our criteria. An analyst will review your application."
else:
    status, color = "declined", RED
    headline, sub = "😕 Not approved right now", "Estimated risk is above our limit. See how to improve below."

apr = round(10.5 + p_bad * 24, 1)
monthly = emi(amount_inr, apr, inputs["duration"])
total_pay = monthly * inputs["duration"]
fee = amount_inr * 0.02

# ---------------- Hero ----------------
if status == "declined":
    hero_title, hero_amt = "Loan request", inr(amount_inr)
    hero_sub = "Adjust amount or tenure to improve your chances"
else:
    hero_title = "Your instant personal loan" if status == "approved" else "Loan request under review"
    hero_amt, hero_sub = inr(amount_inr), f"{inputs['duration']} months at {apr}% p.a."
st.markdown(
    f'<div class="hero"><div class="t">{hero_title}</div><div class="a">{hero_amt}</div>'
    f'<div class="s">{hero_sub}</div>'
    f'<div class="chips"><span>⚡ Instant decision</span><span>🔍 Explainable</span>'
    f'<span>⚖️ Bias-checked</span></div></div>', unsafe_allow_html=True)

# ---------------- Decision ----------------
st.markdown(f'<div class="dec" style="background:{color}"><h2>{headline}</h2><p>{sub}</p></div>',
            unsafe_allow_html=True)

with st.container(border=True):
    st.markdown(gauge(score), unsafe_allow_html=True)
    st.markdown(
        f'<div class="grid"><div class="kpi"><div class="l">Default risk</div>'
        f'<div class="v" style="color:{color}">{p_bad:.0%}</div></div>'
        f'<div class="kpi"><div class="l">Repayment chance</div><div class="v">{1 - p_bad:.0%}</div></div>'
        f'<div class="kpi"><div class="l">Policy limit</div><div class="v">{threshold:.0%}</div></div></div>',
        unsafe_allow_html=True)

if status != "declined":
    with st.container(border=True):
        st.markdown(
            f'<div class="emi"><div class="l">Your monthly EMI</div><div class="v">{inr(monthly)}</div></div>'
            f'<div class="row"><span>Loan amount</span><b>{inr(amount_inr)}</b></div>'
            f'<div class="row"><span>Tenure</span><b>{inputs["duration"]} months</b></div>'
            f'<div class="row"><span>Interest rate (risk-based)</span><b>{apr}% p.a.</b></div>'
            f'<div class="row"><span>Processing fee (2%)</span><b>{inr(fee)}</b></div>'
            f'<div class="row"><span>Total interest</span><b>{inr(total_pay - amount_inr)}</b></div>'
            f'<div class="row" style="border:none"><span>Total repayment</span><b>{inr(total_pay)}</b></div>',
            unsafe_allow_html=True)
        if status == "approved":
            if st.button(f"Get {inr(amount_inr)} now", type="primary", use_container_width=True):
                st.balloons()
                st.markdown(
                    f'<div class="success"><div class="tick">✓</div><h3>Loan approved</h3>'
                    f'<p>{inr(amount_inr)} will be sent to your bank account.<br>'
                    f'First EMI of {inr(monthly)} is due next month.</p></div>',
                    unsafe_allow_html=True)
                st.caption("Demo only. No real money or data is involved.")
        else:
            st.button("Request manual review", use_container_width=True)

# ---------------- Why this decision ----------------
effects = pd.Series((p_bad - probs[1:]) * 100, index=FEATURES)
top = effects.reindex(effects.abs().sort_values(ascending=False).index).head(6)
mx = max(float(top.abs().max()), 1.0)
html = '<div class="sec">Why this decision?</div>'
html += '<div style="font-size:12px;color:#6B7A99;margin-bottom:4px">Change in default risk versus a typical applicant (points). Red raises risk, green lowers it.</div>'
for c, v in top.items():
    val = LABELS.get(c, {}).get(inputs[c], inputs[c])
    if c == "credit_amount": val = inr(amount_inr)
    if c == "duration": val = f"{inputs[c]} months"
    width = max(abs(v) / mx * 100, 3)
    col = RED if v > 0 else GREEN
    sign = "+" if v > 0 else ""
    html += (f'<div class="factor"><div class="n">{NAMES[c]}<small>{val}</small></div>'
             f'<div class="b"><div style="width:{width:.0f}%;background:{col}"></div></div>'
             f'<div class="t" style="color:{col}">{sign}{v:.1f}</div></div>')
with st.container(border=True):
    st.markdown(html, unsafe_allow_html=True)

if status != "approved":
    smaller = {**inputs, "credit_amount": max(250, int(inputs["credit_amount"] * 0.7))}
    shorter_m = max(4, int(inputs["duration"] * 0.7))
    shorter = {**inputs, "duration": shorter_m}
    p_small = float(model.predict_proba(encode([smaller]))[0, 1])
    p_short = float(model.predict_proba(encode([shorter]))[0, 1])
    st.markdown(
        f'<div class="tip">💡 <b>Ways to improve your chances</b><br>'
        f'• Borrow {inr(smaller["credit_amount"] * RATE)} (30% less): risk becomes <b>{p_small:.0%}</b><br>'
        f'• Choose a {shorter_m}-month tenure: risk becomes <b>{p_short:.0%}</b><br>'
        f'• Add a co-applicant or guarantor, or build up your savings.</div>',
        unsafe_allow_html=True)

# ---------------- Insights tabs ----------------
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📊 Model", "⚖️ Fairness", "ℹ️ About"])

with tab1:
    pred_test = model.predict(X_te)
    proba_test = model.predict_proba(X_te)[:, 1]
    a, b, c = st.columns(3)
    a.metric("Accuracy", f"{accuracy_score(y_te, pred_test):.1%}")
    b.metric("ROC-AUC", f"{roc_auc_score(y_te, proba_test):.3f}")
    c.metric("F1", f"{f1_score(y_te, pred_test):.3f}")
    st.caption("Measured on a held-out test set of 200 applicants (20% of the data).")
    st.markdown("**What the model looks at most**")
    imp = pd.Series(model.feature_importances_, index=cols)
    ordered = sorted(FEATURES, key=len, reverse=True)
    grouped = {}
    for col_name, val in imp.items():
        orig = next(f for f in ordered if col_name == f or col_name.startswith(f + "_"))
        grouped[NAMES[orig]] = grouped.get(NAMES[orig], 0) + val
    st.bar_chart(pd.Series(grouped).sort_values(ascending=False).head(10).rename("Importance"))
    with st.expander("Training data"):
        st.write(f"{len(df)} applicants, {len(FEATURES)} features. "
                 f"{(df['class'] == 1).mean():.0%} are labelled bad credit risk.")
        st.dataframe(df.head(50), use_container_width=True)

with tab2:
    st.markdown("**Does the model treat groups differently?**")
    st.caption("Approval rate and accuracy per group in the test set. Large gaps can signal bias.")
    test = df.loc[X_te.index].copy()
    test["approved"] = model.predict(X_te) == 0
    test["correct"] = model.predict(X_te) == y_te
    test["Age group"] = pd.cut(test["age"], [0, 25, 40, 100], labels=["Under 25", "25 to 40", "Over 40"])
    test["Foreign worker"] = test["foreign_worker"].map(LABELS["foreign_worker"])
    test["Personal status"] = test["personal_status"].map(LABELS["personal_status"])
    for group in ["Age group", "Foreign worker", "Personal status"]:
        res = test.groupby(group, observed=True).agg(
            Applicants=("approved", "size"), ApprovalRate=("approved", "mean"), Accuracy=("correct", "mean"))
        res["Approval rate (%)"] = (res["ApprovalRate"] * 100).round(1)
        res["Accuracy (%)"] = (res["Accuracy"] * 100).round(1)
        st.markdown(f"**By {group.lower()}**")
        st.dataframe(res[["Applicants", "Approval rate (%)", "Accuracy (%)"]], use_container_width=True)
        st.bar_chart(res["Approval rate (%)"])
    st.info("Groups are small, so treat the numbers as indicative. "
            "See the notebook for the full Fairlearn analysis.")

with tab3:
    st.markdown("""
**FairLend** is a demonstration project: credit risk prediction with fairness analysis on the
German Credit dataset (UCI Statlog, 1,000 applicants, 20 features).

**Model:** Random Forest classifier. The credit score (300 to 850), the interest rate and the
decision all come from the model's predicted default probability.

**Currency:** the dataset is in Deutsche Marks. Amounts are shown in rupees using a demo
conversion of 1 DM = ₹40.

**Note:** For demonstration only. It must not be used for real lending decisions.
""")

# ---------------- Decorative bottom navigation ----------------
st.markdown(
    '<div class="nav"><div><i>🏠</i>Home</div><div class="on"><i>💳</i>Loans</div>'
    '<div><i>📈</i>Score</div><div><i>👤</i>Profile</div></div>',
    unsafe_allow_html=True)
