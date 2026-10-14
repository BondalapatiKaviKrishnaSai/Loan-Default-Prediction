"""Role-specific pages for the Loan Risk Assessment Desk."""
import json
import random
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import db
from config import (CATEGORICAL, CATEGORY_OPTIONS, DATA_PATH, NUMERIC, compute_emi, compute_ratios)
from ui import (GOLD, NAVY, decide, flash, html_table, kpis, load_models, load_report,
                render_result, style)

INT_COLS = {"age", "dependents", "open_credit_accounts", "recent_inquiries", "late_payments",
            "prior_default", "loan_term_months", "credit_score", "credit_utilization"}


# ---------------------------------------------------------------- helpers
def to_record(row: dict) -> dict:
    return {"ref": row["ref"], "name": row["applicant_name"], "purpose": row["purpose"],
            "mode": row["mode"], "p": row["probability"], "rf": row["rf_prob"], "gb": row["gb_prob"],
            "rec": row["recommendation"], "level": row["risk_level"],
            "cls": {"Low Risk": "low", "Medium Risk": "med", "High Risk": "high"}[row["risk_level"]],
            "applicant": json.loads(row["applicant_json"]), "status": row["status"],
            "reviewer": row["reviewer"], "reviewed_at": row["reviewed_at"],
            "review_comment": row["review_comment"]}


def table(rows):
    df = pd.DataFrame([{
        "Ref": r["ref"], "Time": r["created_at"], "Officer": r["officer_name"], "Branch": r["branch"],
        "Applicant": r["applicant_name"], "Purpose": r["purpose"], "Loan amount": r["loan_amount"],
        "Default prob.": round(r["probability"], 3), "Recommendation": r["recommendation"],
        "Status": r["status"], "Final decision": r["final_decision"] or "-"} for r in rows])
    return df


def pretty(df):
    df = df.copy()
    if "Loan amount" in df:
        df["Loan amount"] = df["Loan amount"].map("{:,.0f}".format)
    if "Default prob." in df:
        df["Default prob."] = df["Default prob."].map("{:.1%}".format)
    return df


def detail_picker(rows, key):
    if not rows:
        return
    labels = {f"{r['ref']} · {r['applicant_name']} · {r['recommendation']}": r for r in rows}
    choice = st.selectbox("Open an application", list(labels), key=key)
    render_result(to_record(labels[choice]), key=key)


# ---------------------------------------------------------------- loan officer / manager
def new_assessment(user):
    models = load_models()
    mode = st.radio("Scoring model", ["Both (consensus)", "Random Forest", "Gradient Boosting"],
                    horizontal=True, key="mode_radio")
    with st.form("assessment"):
        st.markdown("#### 1 · Loan request")
        l1, l2, l3, l4, l5 = st.columns(5)
        name = l1.text_input("Applicant name")
        purpose = l2.selectbox("Loan purpose", CATEGORY_OPTIONS["loan_purpose"])
        loan = l3.number_input("Loan amount", 1000, 5000000, 25000, 1000)
        term = l4.number_input("Term (months)", 6, 360, 36, 6)
        rate = l5.number_input("Interest rate (% p.a.)", 1.0, 40.0, 10.5, 0.25)
        k1, k2 = st.columns(2)
        collateral = k1.number_input("Collateral value (0 if none)", 0, 10000000, 0, 1000)
        cosigner = k2.selectbox("Co-signer", CATEGORY_OPTIONS["cosigner"])

        st.markdown("#### 2 · Applicant profile")
        p1, p2, p3, p4, p5 = st.columns(5)
        age = p1.number_input("Age", 18, 80, 35)
        gender = p2.selectbox("Gender", CATEGORY_OPTIONS["gender"])
        education = p3.selectbox("Education", CATEGORY_OPTIONS["education"])
        marital = p4.selectbox("Marital status", CATEGORY_OPTIONS["marital_status"])
        dependents = p5.number_input("Dependents", 0, 10, 1)

        st.markdown("#### 3 · Employment & income")
        e1, e2, e3, e4 = st.columns(4)
        emp_type = e1.selectbox("Employment type", CATEGORY_OPTIONS["employment_type"])
        emp_years = e2.number_input("Years in current employment", 0.0, 45.0, 6.0, 0.5)
        income = e3.number_input("Annual income", 10000, 5000000, 65000, 1000)
        home = e4.selectbox("Home ownership", CATEGORY_OPTIONS["home_ownership"])

        st.markdown("#### 4 · Credit profile")
        c1, c2, c3, c4, c5 = st.columns(5)
        credit = c1.number_input("Credit score (300-850)", 300, 850, 680)
        hist = c2.number_input("Credit history (years)", 0.0, 50.0, 6.0, 0.5)
        accounts = c3.number_input("Open credit accounts", 0, 30, 4)
        util = c4.number_input("Credit utilisation (%)", 0, 100, 35)
        inquiries = c5.number_input("Recent credit inquiries", 0, 20, 1)

        st.markdown("#### 5 · Finances & repayment history")
        f1, f2, f3, f4, f5 = st.columns(5)
        debt = f1.number_input("Total existing debt", 0, 10000000, 20000, 1000)
        mdebt = f2.number_input("Monthly debt payments", 0, 500000, 800, 50)
        savings = f3.number_input("Savings balance", 0, 10000000, 15000, 1000)
        late = f4.number_input("Late payments on record", 0, 30, 1)
        prior = f5.selectbox("Prior default history", ["No", "Yes"])
        submitted = st.form_submit_button("Assess Applicant")

    if submitted:
        emi = round(float(compute_emi(loan, rate, term)))
        dti, lti = compute_ratios(income, loan, emi, mdebt)
        a = {"age": age, "dependents": dependents, "employment_years": emp_years, "income": income,
             "credit_score": credit, "credit_history_years": hist, "open_credit_accounts": accounts,
             "credit_utilization": util, "recent_inquiries": inquiries, "late_payments": late,
             "prior_default": int(prior == "Yes"), "existing_debt": debt, "monthly_debt_payments": mdebt,
             "savings_balance": savings, "loan_amount": loan, "loan_term_months": term,
             "interest_rate": rate, "collateral_value": collateral, "emi": emi, "debt_to_income": dti,
             "loan_to_income": lti, "gender": gender, "education": education, "marital_status": marital,
             "employment_type": emp_type, "home_ownership": home, "loan_purpose": purpose,
             "cosigner": cosigner}
        row = pd.DataFrame([a])[NUMERIC + CATEGORICAL]
        probs = {n: float(m.predict_proba(row)[0, 1]) for n, m in models.items()}
        p = sum(probs.values()) / len(probs) if mode.startswith("Both") else probs[mode]
        rec, level, cls = decide(p)
        created = db.now()
        record = {"created_at": created, "officer": user["username"], "officer_name": user["full_name"],
                  "branch": user["branch"], "name": name.strip() or "Unnamed applicant",
                  "purpose": purpose, "loan_amount": loan, "p": p, "rf": probs["Random Forest"],
                  "gb": probs["Gradient Boosting"], "rec": rec, "level": level, "mode": mode,
                  "applicant": a, "status": "Cleared (low risk)" if cls == "low" else "Pending review",
                  "final_decision": "Approve" if cls == "low" else ""}
        aid = db.save_assessment(record)
        db.audit(user["username"], "ASSESSMENT_CREATED", f"id {aid}: {rec} ({p:.0%})")
        st.session_state["last_aid"] = aid

    aid = st.session_state.get("last_aid")
    if aid:
        mine = [r for r in db.get_assessments(officer=user["username"]) if r["id"] == aid]
        if mine:
            render_result(to_record(mine[0]), key="new")
            if mine[0]["status"] == "Pending review":
                st.info("This application has been sent to the Branch Manager's review queue.")


def my_applications(user):
    rows = db.get_assessments(officer=user["username"])
    if not rows:
        st.info("You haven't assessed any applications yet.")
        return
    c = st.columns(4)
    c[0].metric("Total", len(rows))
    c[1].metric("Pending review", sum(r["status"] == "Pending review" for r in rows))
    c[2].metric("Approved", sum(r["final_decision"] == "Approve" for r in rows))
    c[3].metric("Rejected", sum(r["final_decision"] == "Reject" for r in rows))
    html_table(pretty(table(rows)), 380)
    detail_picker(rows, "my_pick")


# ---------------------------------------------------------------- branch manager
def review_queue(user):
    rows = db.get_assessments(branch=user["branch"], status="Pending review")
    if not rows:
        st.success("🎉 Nothing waiting for review.")
        return
    st.markdown(f"**{len(rows)} application(s) waiting for your decision** · {user['branch']}")
    labels = {f"{r['ref']} · {r['applicant_name']} · {r['loan_amount']:,.0f} · {r['recommendation']}": r
              for r in rows}
    choice = st.selectbox("Select an application", list(labels), key="queue_pick")
    row = labels[choice]
    st.caption(f"Submitted by {row['officer_name']} on {row['created_at']}")
    render_result(to_record(row), key="queue")
    with st.form(f"review_{row['id']}"):
        st.markdown("#### Your decision")
        decision = st.radio("Decision", ["Approve", "Reject"], horizontal=True)
        comment = st.text_area("Comment (required)", placeholder="Reason for your decision")
        go_ = st.form_submit_button("Submit decision")
    if go_:
        if len(comment.strip()) < 5:
            st.error("Please add a short comment explaining your decision.")
        else:
            rec_says_approve = row["recommendation"] == "Recommend Approval"
            override = (decision == "Approve" and row["recommendation"] == "Recommend Decline") or \
                       (decision == "Reject" and rec_says_approve)
            db.review(row["id"], decision, comment.strip(), user["full_name"], override)
            st.success(f"Decision recorded: {decision}." + (" Flagged as a model override." if override else ""))
            st.rerun()


def dashboard(user, scope):
    rows = db.get_assessments(branch=user["branch"] if scope == "branch" else None)
    st.markdown(f"#### {'Branch dashboard · ' + user['branch'] if scope == 'branch' else 'Portfolio analytics · all branches'}")
    if not rows:
        st.info("No applications yet.")
        if user["role"] == "Administrator" and st.button("Load 40 sample applications", key=f"load_empty_{scope}"):
            run_demo_load(user)
        return
    df = table(rows)
    df["Date"] = pd.to_datetime(df["Time"]).dt.date
    k = st.columns(5)
    k[0].metric("Applications", len(df))
    k[1].metric("Pending review", int((df["Status"] == "Pending review").sum()))
    k[2].metric("Avg default prob.", f"{df['Default prob.'].mean():.0%}")
    k[3].metric("High-risk share", f"{(df['Recommendation'] == 'Recommend Decline').mean():.0%}")
    k[4].metric("Manager overrides", sum(r["override"] for r in rows))
    a, b = st.columns(2)
    with a:
        order = ["Recommend Approval", "Refer for Manual Review", "Recommend Decline"]
        fig = px.histogram(df, x="Recommendation", category_orders={"Recommendation": order},
                           color="Recommendation",
                           color_discrete_map={order[0]: "#2fb67c", order[1]: GOLD, order[2]: "#e5484d"},
                           title="Model recommendations")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig), width="stretch")
    with b:
        fig = px.histogram(df, x="Default prob.", nbins=20, title="Distribution of default probability")
        fig.update_traces(marker_color=NAVY)
        st.plotly_chart(style(fig), width="stretch")
    c, d = st.columns(2)
    with c:
        by = df.groupby("Officer").agg(Applications=("Ref", "count"), AvgRisk=("Default prob.", "mean")).reset_index()
        fig = px.bar(by, x="Officer", y="Applications", title="Applications by officer")
        fig.update_traces(marker_color=GOLD)
        st.plotly_chart(style(fig), width="stretch")
    with d:
        daily = df.groupby("Date").size().reset_index(name="Applications")
        fig = px.line(daily, x="Date", y="Applications", markers=True, title="Daily volume")
        fig.update_traces(line_color=NAVY)
        st.plotly_chart(style(fig), width="stretch")
    if scope == "all":
        by_branch = df.groupby("Branch").agg(Applications=("Ref", "count"), AvgRisk=("Default prob.", "mean")).reset_index()
        by_branch["AvgRisk"] = by_branch["AvgRisk"].map("{:.1%}".format)
        html_table(by_branch)


def all_applications(user, scope):
    rows = db.get_assessments(branch=user["branch"] if scope == "branch" else None)
    if not rows:
        st.info("No applications yet.")
        return
    f1, f2, f3 = st.columns(3)
    stat = f1.multiselect("Status", sorted({r["status"] for r in rows}))
    recm = f2.multiselect("Recommendation", sorted({r["recommendation"] for r in rows}))
    offs = f3.multiselect("Officer", sorted({r["officer_name"] for r in rows}))
    rows = [r for r in rows if (not stat or r["status"] in stat) and
            (not recm or r["recommendation"] in recm) and (not offs or r["officer_name"] in offs)]
    df = table(rows)
    html_table(pretty(df), 460)
    st.download_button("⬇ Download CSV", df.to_csv(index=False).encode(), "applications.csv", "text/csv")
    detail_picker(rows, f"all_pick_{scope}")


# ---------------------------------------------------------------- analyst / admin / shared
def model_insights():
    report = load_report()
    m = report["metrics"]
    st.markdown("#### How the models performed on held-out test data")
    df_m = pd.DataFrame({k: {x: v[x] for x in ("accuracy", "precision", "recall", "f1", "roc_auc")}
                         for k, v in m.items()}).T
    html_table(df_m.round(4).reset_index(names="Model"))
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        for nme, c in report["roc"].items():
            fig.add_scatter(x=c["fpr"], y=c["tpr"], name=f"{nme} (AUC {m[nme]['roc_auc']:.3f})")
        fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"), name="Chance")
        fig.update_layout(title="ROC Curves", xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(style(fig), width="stretch")
    with b:
        cm_name = st.selectbox("Confusion matrix for", list(m), key="cm_pick")
        fig = px.imshow(m[cm_name]["confusion_matrix"], text_auto=True, color_continuous_scale="Blues",
                        x=["Non-default", "Default"], y=["Non-default", "Default"],
                        labels=dict(x="Predicted", y="Actual"), title=f"{cm_name} - Confusion Matrix")
        st.plotly_chart(style(fig), width="stretch")
    fi_name = st.radio("Feature importance for", list(report["importance"]), horizontal=True, key="fi_pick")
    imp = pd.Series(report["importance"][fi_name]).head(12).sort_values()
    fig = px.bar(imp, orientation="h", title=f"Top drivers of default - {fi_name}",
                 labels={"value": "Importance", "index": ""})
    fig.update_traces(marker_color=GOLD)
    fig.update_layout(showlegend=False)
    st.plotly_chart(style(fig), width="stretch")


def admin_users(user):
    st.markdown("#### Staff accounts")
    html_table(pd.DataFrame(db.list_users()), 320)
    a, b = st.columns(2)
    with a:
        with st.form("new_user", clear_on_submit=True):
            st.markdown("**Create account**")
            uname = st.text_input("Username")
            fname = st.text_input("Full name")
            role = st.selectbox("Role", db.ROLES)
            branch = st.text_input("Branch", "Main Branch")
            pw = st.text_input("Temporary password (min 8 chars)", type="password")
            if st.form_submit_button("Create account"):
                if not uname.strip() or not fname.strip() or len(pw) < 8:
                    st.error("Username, full name and a password of at least 8 characters are required.")
                elif db.run("SELECT 1 AS x FROM users WHERE username=?", (uname.strip().lower(),), "one"):
                    st.error("That username already exists.")
                else:
                    db.create_user(uname, pw, role, fname.strip(), branch.strip(), who=user["username"])
                    st.success(f"Account '{uname.strip().lower()}' created.")
                    st.rerun()
    with b:
        users = [u["Username"] for u in db.list_users()]
        st.markdown("**Manage an account**")
        target = st.selectbox("Account", users, key="manage_user")
        c1, c2, c3 = st.columns(3)
        if c1.button("Activate / Deactivate"):
            cur = db.run("SELECT active FROM users WHERE username=?", (target,), "one")["active"]
            if target == user["username"]:
                st.error("You can't deactivate your own account.")
            else:
                db.set_active(target, not cur, user["username"])
                st.rerun()
        if c2.button("Unlock"):
            db.unlock(target, user["username"])
            st.rerun()
        with st.form("reset_pw", clear_on_submit=True):
            newpw = st.text_input("New password for the selected account", type="password")
            if st.form_submit_button("Reset password"):
                if len(newpw) < 8:
                    st.error("Password must be at least 8 characters.")
                else:
                    db.set_password(target, newpw, user["username"])
                    st.success("Password updated.")


def admin_policy(user):
    low, high = db.get_thresholds()
    st.markdown("#### Decision policy")
    st.caption("Default probability cut-offs that turn the model score into a recommendation.")
    with st.form("policy"):
        new_low = st.slider("Approve below", 0.05, 0.60, low, 0.01)
        new_high = st.slider("Decline at or above", 0.10, 0.95, high, 0.01)
        if st.form_submit_button("Save policy"):
            if new_low >= new_high:
                st.error("The approval cut-off must be lower than the decline cut-off.")
            else:
                db.set_setting("low_t", new_low)
                db.set_setting("high_t", new_high)
                db.audit(user["username"], "POLICY_CHANGED", f"approve<{new_low:.2f}, decline>={new_high:.2f}")
                st.success("Policy saved. New assessments will use these cut-offs.")
    st.markdown("#### Demo data")
    st.caption("Fills the system with sample applications (taken from the synthetic dataset) so the "
               "dashboards have something to show.")
    total = len(db.get_assessments())
    st.caption(f"Applications currently in the system: **{total}**")
    b1, b2 = st.columns(2)
    if b1.button("Load 40 sample applications", key="load_demo"):
        run_demo_load(user)
    sure = b2.checkbox("I understand this deletes every application", key="confirm_clear")
    if b2.button("Delete all applications", key="clear_all") and sure:
        db.run("DELETE FROM assessments")
        db.audit(user["username"], "ALL_APPLICATIONS_DELETED")
        flash("info", "All applications were deleted.")
        st.rerun()


def audit_log():
    st.markdown("#### Audit trail")
    df = pd.DataFrame(db.list_audit())
    if df.empty:
        st.info("No events yet.")
        return
    html_table(df, 520)
    st.download_button("⬇ Download CSV", df.to_csv(index=False).encode(), "audit_log.csv", "text/csv")


def account(user):
    st.markdown("#### My account")
    st.write(f"**Name:** {user['full_name']}  \n**Username:** {user['username']}  \n"
             f"**Role:** {user['role']}  \n**Branch:** {user['branch']}")
    with st.form("chg_pw", clear_on_submit=True):
        cur = st.text_input("Current password", type="password")
        new = st.text_input("New password (min 8 chars)", type="password")
        conf = st.text_input("Confirm new password", type="password")
        if st.form_submit_button("Change password"):
            row = db.run("SELECT salt, pw_hash FROM users WHERE username=?", (user["username"],), "one")
            if db.hash_pw(cur, row["salt"]) != row["pw_hash"]:
                st.error("Current password is incorrect.")
            elif len(new) < 8 or new != conf:
                st.error("New passwords must match and be at least 8 characters.")
            else:
                db.set_password(user["username"], new, user["username"])
                st.success("Password changed.")


# ---------------------------------------------------------------- demo data
def load_demo_data(n=40):
    models = load_models()
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:  # rebuild the dataset if the CSV is missing
        from generate_data import generate
        df = generate()
    df = df.dropna()
    for c in CATEGORICAL:
        df[c] = df[c].astype(str).str.strip().str.title()
    df = df.sample(n, random_state=random.randint(0, 10_000))
    officers = db.run("SELECT username, full_name, branch FROM users WHERE role='Loan Officer' AND active=1", fetch="all")
    mgr = db.run("SELECT full_name FROM users WHERE role='Branch Manager' AND active=1", fetch="one")
    if not officers:
        return 0
    count = 0
    for i, (_, r) in enumerate(df.iterrows(), 1):
        a = {c: (int(r[c]) if c in INT_COLS else float(r[c])) for c in NUMERIC}
        a.update({c: str(r[c]) for c in CATEGORICAL})
        probs = {k: float(m.predict_proba(pd.DataFrame([a])[NUMERIC + CATEGORICAL])[0, 1]) for k, m in models.items()}
        p = sum(probs.values()) / 2
        rec, level, cls = decide(p)
        o = random.choice(officers)
        created = (datetime.now() - timedelta(days=random.randint(0, 29), hours=random.randint(0, 8),
                                              minutes=random.randint(0, 59))).strftime("%Y-%m-%d %H:%M:%S")
        aid = db.save_assessment({
            "created_at": created, "officer": o["username"], "officer_name": o["full_name"],
            "branch": o["branch"], "name": f"Sample Applicant {i:02d}", "purpose": a["loan_purpose"],
            "loan_amount": a["loan_amount"], "p": p, "rf": probs["Random Forest"],
            "gb": probs["Gradient Boosting"], "rec": rec, "level": level, "mode": "Both (consensus)",
            "applicant": a, "status": "Cleared (low risk)" if cls == "low" else "Pending review",
            "final_decision": "Approve" if cls == "low" else ""})
        if cls != "low" and mgr and random.random() < 0.3:
            decision = "Approve" if cls == "med" and random.random() < 0.6 else "Reject"
            override = decision == "Approve" and rec == "Recommend Decline"
            db.review(aid, decision, "Reviewed against policy and supporting documents.", mgr["full_name"], override)
        count += 1
    return count


def run_demo_load(user):
    """Load sample applications, always giving the admin visible feedback."""
    try:
        with st.spinner("Scoring sample applicants with both models..."):
            n = load_demo_data()
        if n == 0:
            flash("error", "No active Loan Officer accounts exist, so there is nobody to attribute the samples to.")
        else:
            db.audit(user["username"], "DEMO_DATA_LOADED", f"{n} applications")
            flash("success", f"Loaded {n} sample applications. See them in Live Data and Portfolio Analytics.")
    except Exception as e:  # show the real reason instead of failing silently
        flash("error", f"Could not load sample data: {e}")
    st.rerun()


# ---------------------------------------------------------------- admin: live data
def flatten(rows):
    out = []
    for r in rows:
        a = json.loads(r["applicant_json"])
        row = {"Ref": r["ref"], "Time": r["created_at"], "Officer": r["officer_name"], "Branch": r["branch"],
               "Applicant": r["applicant_name"], "RF prob.": round(r["rf_prob"], 3),
               "GB prob.": round(r["gb_prob"], 3), "Final prob.": round(r["probability"], 3),
               "Recommendation": r["recommendation"], "Status": r["status"],
               "Final decision": r["final_decision"] or "-"}
        row.update(a)
        out.append(row)
    return pd.DataFrame(out)


def admin_data(user):
    st.markdown("### 🗂️ Live Data")
    st.caption("Everything the system is generating: applications entered by staff, their scores, and the "
               "synthetic dataset the models were trained on.")
    st.toggle("Auto-refresh every 5 seconds", value=True, key="data_live")
    live_feed()

    st.markdown("#### Training dataset (synthetic, generated by `generate_data.py`)")
    if not DATA_PATH.exists():
        st.warning("data/loan_data.csv not found. Run `python generate_data.py`.")
        return
    ds = pd.read_csv(DATA_PATH)
    kpis([("Rows", f"{len(ds):,}"), ("Columns", ds.shape[1]), ("Default rate", f"{ds['default'].mean():.1%}"),
          ("Missing values", f"{int(ds.isna().sum().sum()):,}")])
    html_table(ds.head(40).round(2), 360)
    c1, c2 = st.columns(2)
    feat = c1.selectbox("Explore a feature", [c for c in ds.columns if c != "default"],
                        index=list(ds.columns).index("credit_score"), key="ds_feat")
    if pd.api.types.is_numeric_dtype(ds[feat]):
        fig = px.histogram(ds, x=feat, color=ds["default"].map({0: "Repaid", 1: "Defaulted"}), nbins=40,
                           barmode="overlay", opacity=.65, labels={"color": ""},
                           color_discrete_map={"Repaid": "#2fb67c", "Defaulted": "#e5484d"},
                           title=f"{feat} — repaid vs defaulted")
    else:
        g = ds.groupby(feat)["default"].mean().reset_index()
        fig = px.bar(g, x=feat, y="default", title=f"Default rate by {feat}", labels={"default": "Default rate"})
        fig.update_traces(marker_color=GOLD)
    st.plotly_chart(style(fig), width="stretch", key="ds_chart")
    st.download_button("⬇ Download dataset CSV", ds.to_csv(index=False).encode(), "loan_data.csv", "text/csv")


@st.fragment(run_every="5s")
def live_feed():
    if not st.session_state.get("data_live", True):
        st.caption("Auto-refresh is paused.")
    rows = db.get_assessments()
    users = db.run("SELECT COUNT(*) AS n FROM users", fetch="one")["n"]
    today = datetime.now().strftime("%Y-%m-%d")
    kpis([("Applications", len(rows)), ("Today", sum(r["created_at"].startswith(today) for r in rows)),
          ("Pending review", sum(r["status"] == "Pending review" for r in rows)),
          ("Cleared (low risk)", sum(r["status"].startswith("Cleared") for r in rows)),
          ("Reviewed by managers", sum(bool(r["reviewer"]) for r in rows)),
          ("Staff accounts", users)])
    st.caption(f"Last refreshed {datetime.now():%H:%M:%S}")
    if not rows:
        st.info("No applications yet. Use **Policy & Demo Data → Load 40 sample applications**, or sign in "
                "as a Loan Officer and assess one.")
        return
    df = flatten(rows)
    st.markdown("**Application feed** (newest first, all 28 input parameters included)")
    show = ["Ref", "Time", "Officer", "Applicant", "loan_purpose", "loan_amount", "loan_term_months",
            "interest_rate", "income", "credit_score", "credit_utilization", "late_payments",
            "debt_to_income", "RF prob.", "GB prob.", "Recommendation", "Status"]
    html_table(df[show].head(25), 420)
    st.download_button("⬇ Download all applications with every parameter (CSV)",
                       df.to_csv(index=False).encode(), "applications_full.csv", "text/csv", key="feed_dl")
    with st.expander("Inspect every parameter of one application"):
        pick = st.selectbox("Application", df["Ref"].tolist(), key="feed_pick")
        r = df[df["Ref"] == pick].iloc[0].drop(["Ref"])
        html_table(pd.DataFrame({"Parameter": r.index, "Value": [str(x) for x in r.values]}), 480)
