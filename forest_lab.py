"""Admin page: what the Random Forest uses, how it works, and a live view of its votes."""
import html as _html
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import db
from config import CATEGORICAL, NUMERIC, compute_emi, compute_ratios
from ui import GOLD, html_table, kpis, load_models, load_report, style

DEFAULT = {"age": 35, "dependents": 1, "employment_years": 6.0, "income": 65000, "credit_score": 680,
           "credit_history_years": 6.0, "open_credit_accounts": 4, "credit_utilization": 35,
           "recent_inquiries": 1, "late_payments": 1, "prior_default": 0, "existing_debt": 20000,
           "monthly_debt_payments": 800, "savings_balance": 15000, "loan_amount": 25000,
           "loan_term_months": 36, "interest_rate": 10.5, "collateral_value": 0, "gender": "Male",
           "education": "Graduate", "marital_status": "Married", "employment_type": "Salaried",
           "home_ownership": "Rent", "loan_purpose": "Personal Loan", "cosigner": "No"}


# ------------------------------------------------------------------ model plumbing
def parts():
    pipe = load_models()["Random Forest"]
    return pipe.named_steps["prep"], pipe.named_steps["model"]


def transform(prep, applicant: dict) -> np.ndarray:
    X = prep.transform(pd.DataFrame([applicant])[NUMERIC + CATEGORICAL])
    return (X.toarray() if hasattr(X, "toarray") else np.asarray(X)).astype(np.float32)


def tree_probs(rf, X) -> np.ndarray:
    """Default probability from every individual tree (the forest averages these)."""
    return np.array([t.predict_proba(X)[0, 1] for t in rf.estimators_])


def finish(a: dict) -> dict:
    a = dict(a)
    a["emi"] = round(float(compute_emi(a["loan_amount"], a["interest_rate"], a["loan_term_months"])))
    a["debt_to_income"], a["loan_to_income"] = compute_ratios(
        a["income"], a["loan_amount"], a["emi"], a["monthly_debt_payments"])
    return a


def tree_path(prep, rf, idx, X, applicant):
    """Human-readable decision path of one tree for this applicant."""
    tree = rf.estimators_[idx]
    t = tree.tree_
    names = prep.get_feature_names_out()
    scaler = prep.named_transformers_["num"].named_steps["scale"]
    mean, scale = dict(zip(NUMERIC, scaler.mean_)), dict(zip(NUMERIC, scaler.scale_))
    steps, leaf = [], None
    for nid in tree.decision_path(X).indices:
        if t.children_left[nid] == t.children_right[nid]:
            v = t.value[nid][0]
            leaf = float(v[1] / v.sum())
            continue
        f, thr = t.feature[nid], t.threshold[nid]
        left = X[0, f] <= thr
        name = names[f]
        if name.startswith("num__"):
            col = name[5:]
            real = thr * scale[col] + mean[col]
            steps.append(f"<b>{col.replace('_', ' ')}</b> = {applicant[col]:,.2f} "
                         f"{'≤' if left else '>'} {real:,.2f}")
        else:
            base = name[5:]
            col = next(c for c in CATEGORICAL if base.startswith(c + "_"))
            level = base[len(col) + 1:]
            steps.append(f"<b>{col.replace('_', ' ')}</b> is “{level}”? {'No' if left else 'Yes'}")
    return steps, leaf


# ------------------------------------------------------------------ page
def page(user):
    prep, rf = parts()
    n_trees = len(rf.estimators_)
    n_feat = len(prep.get_feature_names_out())
    depths = [t.get_depth() for t in rf.estimators_]
    leaves = [t.get_n_leaves() for t in rf.estimators_]
    per_split = int(np.sqrt(n_feat)) if rf.max_features == "sqrt" else rf.max_features

    st.markdown("### 🌲 Random Forest Lab")
    st.caption("What the Random Forest is made of, how it reaches a decision, and what it is doing right now.")

    # ---- 1. what it uses
    st.markdown("#### 1 · What the model uses")
    kpis([("Trees in the forest", n_trees), ("Max depth allowed", rf.max_depth),
          ("Avg. actual depth", f"{np.mean(depths):.1f}"), ("Avg. leaves per tree", f"{np.mean(leaves):.0f}"),
          ("Input fields", len(NUMERIC) + len(CATEGORICAL)), ("Features after encoding", n_feat)])
    html_table(pd.DataFrame([
        ["Algorithm", "Random Forest classifier (bagging)", "Many decision trees, each trained on a different random sample"],
        ["Trees (n_estimators)", n_trees, "More trees = steadier, less noisy predictions"],
        ["Max depth", rf.max_depth, "Stops a tree from memorising individual applicants"],
        ["Min samples per leaf", rf.min_samples_leaf, "Every final decision rests on at least this many past loans"],
        ["Features tried per split", f"~{per_split} of {n_feat}", "Each split sees only a random subset, so trees differ"],
        ["Bootstrap sampling", "Yes", "Each tree trains on a random sample drawn with replacement"],
        ["Split quality", rf.criterion.title(), "How a tree measures whether a split separates defaulters well"],
        ["Class weight", str(rf.class_weight).title(), "Defaulters are rarer; this stops the model ignoring them"],
        ["Preprocessing", "Impute → scale → one-hot", "Median/most-frequent fill, standardise numbers, encode categories"],
        ["Final probability", "Average of all trees", "Decision cut-offs come from the policy set by the administrator"],
    ], columns=["Setting", "Value", "What it means"]))

    with st.expander("How a Random Forest works (in plain steps)", expanded=False):
        st.markdown(f"""
1. **Bootstrap.** Each of the {n_trees} trees gets its own random sample of past loans (drawn with replacement), so no two trees see exactly the same history.
2. **Grow a tree.** A tree asks a chain of yes/no questions such as *“credit score ≤ 641?”* or *“late payments > 2?”*. At every question it may only consider ~{per_split} randomly chosen features out of {n_feat}, which keeps the trees different from each other.
3. **Every tree votes.** For a new applicant, each tree follows its questions down to a leaf and reports the share of past borrowers in that leaf who defaulted.
4. **Average the votes.** The forest’s default probability is the average over all {n_trees} trees. The bank policy (approve / review / decline cut-offs) is applied to that number.

Many different, imperfect trees averaged together make fewer mistakes than any single tree. That is the idea behind the *“forest”*.""")

    # ---- 2. live vote
    st.markdown("#### 2 · Live forest vote")
    st.caption("Move any control and the whole forest re-votes instantly.")
    src = st.radio("Applicant source", ["What-if sliders", "Load a saved application"],
                   horizontal=True, key="lab_src")
    base = dict(DEFAULT)
    if src == "What-if sliders":
        c1, c2, c3 = st.columns(3)
        base["credit_score"] = c1.slider("Credit score", 300, 850, 680, key="lab_cs")
        base["late_payments"] = c1.slider("Late payments", 0, 12, 1, key="lab_late")
        base["credit_utilization"] = c1.slider("Credit utilisation %", 0, 100, 35, key="lab_util")
        base["recent_inquiries"] = c1.slider("Recent inquiries", 0, 10, 1, key="lab_inq")
        base["loan_amount"] = c2.slider("Loan amount", 1000, 500000, 25000, 1000, key="lab_loan")
        base["income"] = c2.slider("Annual income", 12000, 400000, 65000, 1000, key="lab_inc")
        base["loan_term_months"] = c2.select_slider("Term (months)", [12, 24, 36, 48, 60, 120, 180, 240], 36, key="lab_term")
        base["interest_rate"] = c2.slider("Interest rate %", 5.0, 24.0, 10.5, 0.25, key="lab_rate")
        base["monthly_debt_payments"] = c3.slider("Monthly debt payments", 0, 8000, 800, 50, key="lab_md")
        base["employment_years"] = c3.slider("Years employed", 0.0, 40.0, 6.0, 0.5, key="lab_emp")
        base["savings_balance"] = c3.slider("Savings", 0, 300000, 15000, 1000, key="lab_sav")
        base["prior_default"] = int(c3.checkbox("Prior default on record", key="lab_prior"))
        applicant = finish(base)
    else:
        rows = db.get_assessments()[:60]
        if not rows:
            st.info("No saved applications yet. Load sample data from **Policy & Demo Data** first.")
            applicant = finish(base)
        else:
            import json
            lab = {f"{r['ref']} · {r['applicant_name']} · {r['recommendation']}": r for r in rows}
            pick = st.selectbox("Application", list(lab), key="lab_pick")
            applicant = json.loads(lab[pick]["applicant_json"])

    X = transform(prep, applicant)
    probs = tree_probs(rf, X)
    p = float(probs.mean())
    yes = int((probs >= 0.5).sum())
    low, high = db.get_thresholds()
    verdict = "Decline" if p >= high else "Manual review" if p >= low else "Approve"
    kpis([("Forest default probability", f"{p:.1%}"), ("Trees voting “default”", f"{yes} / {n_trees}"),
          ("Trees voting “repay”", f"{n_trees - yes} / {n_trees}"), ("Policy outcome", verdict),
          ("Spread between trees", f"±{probs.std():.0%}")])

    cells = "".join(f'<span class="{"yes" if q >= .5 else "no"}" title="Tree {i + 1}: {q:.0%} default"></span>'
                    for i, q in enumerate(probs))
    st.markdown('<div class="legend"><b>Each square is one tree</b><span style="background:#e5484d"></span>votes default'
                '<span style="background:#2fb67c"></span>votes repay</div>'
                f'<div class="vgrid">{cells}</div>', unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        fig = go.Figure(go.Histogram(x=probs, nbinsx=20, marker_color=GOLD))
        fig.add_vline(x=p, line_dash="dash", line_color="#e5484d", annotation_text=f"average {p:.0%}")
        fig.update_layout(title="What the individual trees say", xaxis_title="Tree's default probability",
                          yaxis_title="Number of trees", xaxis_range=[0, 1])
        st.plotly_chart(style(fig), width="stretch", key="lab_hist")
    with b:
        run = np.cumsum(probs) / np.arange(1, n_trees + 1)
        fig = go.Figure(go.Scatter(x=list(range(1, n_trees + 1)), y=run, mode="lines", line_color=GOLD))
        fig.add_hline(y=p, line_dash="dot", line_color="#e5484d")
        fig.update_layout(title="Prediction settles as trees are added", xaxis_title="Trees included",
                          yaxis_title="Running average", yaxis_tickformat=".0%")
        st.plotly_chart(style(fig), width="stretch", key="lab_conv")

    # ---- 3. inside one tree
    st.markdown("#### 3 · Inside a single tree")
    idx = st.slider("Choose a tree", 1, n_trees, 1, key="lab_tree") - 1
    steps, leaf = tree_path(prep, rf, idx, X, applicant)
    st.caption(f"Tree {idx + 1} has depth {depths[idx]} and {leaves[idx]} leaves. "
               f"It reaches this applicant after {len(steps)} questions:")
    for i, s in enumerate(steps, 1):
        st.markdown(f'<div class="step">Q{i} · {s}</div>', unsafe_allow_html=True)
    if leaf is not None:
        st.markdown(f'<div class="step"><b>Leaf reached → this tree says {leaf:.0%} default '
                    f'({"votes default" if leaf >= .5 else "votes repay"})</b></div>', unsafe_allow_html=True)

    # ---- 4. importance
    st.markdown("#### 4 · What drives the forest overall")
    imp = pd.Series(load_report()["importance"]["Random Forest"]).head(12).sort_values()
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=GOLD))
    fig.update_layout(title="Feature importance (share of total split quality)", height=420)
    st.plotly_chart(style(fig), width="stretch", key="lab_imp")

    # ---- 5. live monitor
    st.markdown("#### 5 · Live monitor: forest votes on real applications")
    st.toggle("Auto-refresh every 5 seconds", value=True, key="lab_live")
    live_monitor()


@st.fragment(run_every="5s")
def live_monitor():
    if not st.session_state.get("lab_live", True):
        st.caption("Auto-refresh is paused.")
        return
    prep, rf = parts()
    import json
    rows = db.get_assessments()[:10]
    st.caption(f"Last refreshed {datetime.now():%H:%M:%S} · newest 10 applications scored in the system")
    if not rows:
        st.info("Waiting for applications. Assess one as a Loan Officer, or load sample data.")
        return
    out = []
    for r in rows:
        probs = tree_probs(rf, transform(prep, json.loads(r["applicant_json"])))
        out.append({"Time": r["created_at"], "Ref": r["ref"], "Applicant": r["applicant_name"],
                    "Trees → default": f"{int((probs >= .5).sum())} / {len(probs)}",
                    "Forest probability": f"{probs.mean():.1%}", "Recommendation": r["recommendation"],
                    "Status": r["status"]})
    html_table(pd.DataFrame(out))
