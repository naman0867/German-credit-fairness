

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
st.markdown(
    '<div class="topbar"><div class="brand"><div class="brand-mark">F</div>'
    '<div><div class="logo">Fair<span>Lend</span></div><div class="brand-sub">CREDIT, MADE CLEAR</div></div></div>'
    '<div class="avatar">N</div></div>',
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
                st.caption("DEMO EXPERIENCE  •  No real loan is issued and no banking data is collected.")
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
