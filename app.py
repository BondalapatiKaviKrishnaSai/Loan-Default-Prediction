"""Loan Risk Assessment Desk - role-based Streamlit app (light / dark mode).

Portals: Loan Officer, Branch Manager, Credit Analyst, Administrator.
Run:  streamlit run app.py
"""
import streamlit as st

import db
import forest_lab
import views
from config import MODELS_DIR
from ui import ROLE_ICON, html_table, inject_css, show_flash

st.set_page_config(page_title="Loan Risk Assessment Desk", page_icon="🏦", layout="wide")
inject_css()
db.init_db()

PORTALS = {
    "Loan Officer": ("Assess new loan applicants one at a time and track the status of your applications.",
                     ["Detailed applicant form (28 parameters)", "Instant AI risk score and key factors",
                      "Personal application history"]),
    "Branch Manager": ("Review referred and high-risk applications, approve or reject them, and monitor your branch.",
                       ["Review queue with required comments", "Branch dashboard", "Model override tracking"]),
    "Credit Analyst": ("Analyse portfolio risk across branches and inspect model performance.",
                       ["Portfolio analytics", "All applications with export", "ROC, confusion matrices, feature importance"]),
    "Administrator": ("Manage staff, policy and data, and watch the models work in real time.",
                      ["Live data feed and dataset explorer", "Random Forest Lab (live votes)",
                       "User management, policy cut-offs, audit log"]),
}

# role -> ordered list of (label, function(user))
PAGES = {
    "Loan Officer": [("📝 New Assessment", views.new_assessment),
                     ("📂 My Applications", views.my_applications),
                     ("👤 My Account", views.account)],
    "Branch Manager": [("✅ Review Queue", views.review_queue),
                       ("📊 Branch Dashboard", lambda u: views.dashboard(u, "branch")),
                       ("📝 New Assessment", views.new_assessment),
                       ("📂 All Applications", lambda u: views.all_applications(u, "branch")),
                       ("📈 Model Insights", lambda u: views.model_insights()),
                       ("👤 My Account", views.account)],
    "Credit Analyst": [("📊 Portfolio Analytics", lambda u: views.dashboard(u, "all")),
                       ("📂 All Applications", lambda u: views.all_applications(u, "all")),
                       ("📈 Model Insights", lambda u: views.model_insights()),
                       ("👤 My Account", views.account)],
    "Administrator": [("🗂️ Live Data", views.admin_data),
                      ("🌲 Random Forest Lab", forest_lab.page),
                      ("👥 User Management", views.admin_users),
                      ("⚙️ Policy & Demo Data", views.admin_policy),
                      ("🧾 Audit Log", lambda u: views.audit_log()),
                      ("📊 Portfolio Analytics", lambda u: views.dashboard(u, "all")),
                      ("📈 Model Insights", lambda u: views.model_insights()),
                      ("👤 My Account", views.account)],
}


def login_page():
    top, toggle = st.columns([6, 1.2])
    with top:
        st.markdown("""<div class="banner"><h1>🏦 Loan Risk Assessment Desk</h1>
        <p>AI-assisted loan default risk assessment for bank staff</p>
        <span class="chip">Random Forest</span><span class="chip">Gradient Boosting</span>
        <span class="chip">Role-based access</span><span class="chip">Audit trail</span></div>""",
                    unsafe_allow_html=True)
    with toggle:
        st.write("")
        st.toggle("🌙 Dark mode", key="dark")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### Sign in")
        role = st.radio("Choose your portal", db.ROLES, horizontal=True, key="portal",
                        format_func=lambda r: f"{ROLE_ICON[r]} {r}")
        with st.form(f"login_{role}"):
            username = st.text_input("Username", key=f"user_{role}")
            password = st.text_input("Password", type="password", key=f"pass_{role}")
            ok = st.form_submit_button("Sign in", key=f"signin_{role}")
        if ok:
            user, err = db.authenticate(username, password, role)
            if err:
                st.error(err)
            else:
                st.session_state["user"] = user
                st.rerun()
    with right:
        desc, bullets = PORTALS[role]
        items = "".join(f"<li>{b}</li>" for b in bullets)
        st.markdown(f'<div class="card"><b>{ROLE_ICON[role]} {role} portal</b><br>{desc}'
                    f'<ul style="margin:.5rem 0 0 1rem">{items}</ul></div>', unsafe_allow_html=True)
        with st.expander("Demo accounts (project demo only)", expanded=True):
            import pandas as pd
            html_table(pd.DataFrame([{"Portal": u[2], "Username": u[0], "Password": u[1]} for u in db.DEMO_USERS]))
            st.caption("Change these passwords from the My Account tab before any real use.")


def main_app(user):
    left, tog, out = st.columns([6, 1.3, 1])
    with left:
        st.markdown(f"""<div class="banner"><h1>🏦 Loan Risk Assessment Desk</h1>
        <p><span class="badge">{ROLE_ICON[user['role']]} {user['role']}</span>
        {user['full_name']} · {user['branch']}</p></div>""", unsafe_allow_html=True)
    with tog:
        st.write("")
        st.toggle("🌙 Dark mode", key="dark")
    with out:
        st.write("")
        if st.button("Sign out", key="logout"):
            db.audit(user["username"], "LOGOUT")
            for k in list(st.session_state.keys()):
                if k != "dark":          # keep the theme choice
                    del st.session_state[k]
            st.rerun()
    show_flash()
    pages = PAGES[user["role"]]
    labels = [label for label, _ in pages]
    choice = st.radio("Navigate", labels, horizontal=True, label_visibility="collapsed",
                      key=f"nav_{user['role']}")
    st.write("")
    dict(pages)[choice](user)


if not (MODELS_DIR / "random_forest.joblib").exists():
    st.error("Models not found. Run `python generate_data.py` then `python train.py` first.")
    st.stop()

user = st.session_state.get("user")
if user:
    # re-validate on every run so deactivating an account takes effect immediately
    fresh = db.run("SELECT * FROM users WHERE username=? AND active=1", (user["username"],), "one")
    if not fresh:
        st.session_state.clear()
        st.rerun()
    main_app(fresh)
else:
    login_page()
