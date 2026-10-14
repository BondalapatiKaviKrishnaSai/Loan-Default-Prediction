"""Shared UI helpers: light/dark theme, model loading, risk rules, tables and the result card."""
import html as _html
import json
from string import Template

import joblib
import plotly.graph_objects as go
import streamlit as st

import db
from config import MODELS_DIR, REPORTS_DIR

NAVY, GOLD = "#1E2761", "#F2A93B"
MODEL_FILES = {"Random Forest": "random_forest.joblib", "Gradient Boosting": "gradient_boosting.joblib"}
ROLE_ICON = {"Loan Officer": "👤", "Branch Manager": "🏢", "Credit Analyst": "📈", "Administrator": "🛡️"}

LIGHT = dict(bg="#f3f6fd", surface="#ffffff", surface2="#f7f9ff", text="#1b2559", muted="#5b6591",
             border="#dfe5f5", input="#ffffff", th="#eaf0ff", ok="#e3f7ee", warn="#fff1d6", bad="#fde3e4",
             hover="#eef2ff", shadow="rgba(30,39,97,.10)", b1="#1E2761", b2="#33439c")
DARK = dict(bg="#0b1029", surface="#151c42", surface2="#1b2352", text="#eef1ff", muted="#a9b3e6",
            border="#2c3772", input="#0e1536", th="#1d2658", ok="rgba(47,182,124,.22)",
            warn="rgba(242,169,59,.24)", bad="rgba(229,72,77,.28)", hover="#222c63",
            shadow="rgba(0,0,0,.45)", b1="#1a2468", b2="#3a4cc0")

# Every widget is styled explicitly (with !important) so the app looks identical whatever the
# browser / Streamlit theme is, and so nothing is "invisible until hovered".
CSS = Template("""
<style>
:root{--bg:$bg;--surface:$surface;--surface2:$surface2;--text:$text;--muted:$muted;--border:$border;
--input:$input;--th:$th;--ok:$ok;--warn:$warn;--bad:$bad;--hover:$hover;--shadow:$shadow;}
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:var(--bg) !important;color:var(--text);}
header[data-testid="stHeader"]{background:transparent !important;}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none !important;}
.block-container{padding-top:1.2rem;max-width:1320px;}
/* ---- text ---- */
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5{color:var(--text) !important;}
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong,[data-testid="stMarkdownContainer"] em,
[data-testid="stMarkdownContainer"] td,[data-testid="stMarkdownContainer"] th{color:var(--text);}
label,label p,label span,[data-testid="stWidgetLabel"],[data-testid="stWidgetLabel"] *{color:var(--text) !important;}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] *{color:var(--muted) !important;}
[data-testid="stMetric"]{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:.6rem .9rem;}
[data-testid="stMetricLabel"] *{color:var(--muted) !important;}
[data-testid="stMetricValue"] *{color:var(--text) !important;}
/* ---- inputs (Streamlit uses data-testid hooks) ---- */
[data-testid="stTextInputRootElement"],[data-testid="stNumberInputContainer"],[data-testid="stTextAreaRootElement"],
[data-testid="stSelectbox"] [role="group"],[data-testid="stMultiSelect"] [role="group"],[data-baseweb="select"]>div,
[data-baseweb="input"],[data-baseweb="base-input"],[data-baseweb="textarea"]{
  background:var(--input) !important;border:1px solid var(--border) !important;border-radius:10px !important;}
input,textarea{color:var(--text) !important;-webkit-text-fill-color:var(--text) !important;
  background:transparent !important;caret-color:var(--text);}
input::placeholder,textarea::placeholder{color:var(--muted) !important;-webkit-text-fill-color:var(--muted) !important;opacity:.85;}
[data-testid="stSelectbox"] *,[data-testid="stMultiSelect"] *{color:var(--text) !important;}
[data-testid="stSelectbox"] svg,[data-testid="stMultiSelect"] svg,[data-testid="stTextInputRootElement"] svg{fill:var(--text) !important;}
[data-testid^="stNumberInputStep"],[data-testid^="stNumberInputStep"] *{color:var(--text) !important;fill:var(--text) !important;}
[data-testid^="stNumberInputStep"]{background:var(--surface2) !important;}
/* dropdown menus render in a portal outside the app */
[data-baseweb="popover"],[data-baseweb="popover"]>div,[data-baseweb="menu"],ul[role="listbox"],
[data-testid="stSelectboxVirtualDropdown"],div[data-testid="stSelectboxVirtualDropdown"] *{
  background:var(--surface) !important;color:var(--text) !important;}
li[role="option"],[role="option"]{background:var(--surface) !important;color:var(--text) !important;}
li[role="option"] *,[role="option"] *{color:var(--text) !important;}
li[role="option"]:hover,[role="option"]:hover,[role="option"][aria-selected="true"]{background:var(--hover) !important;}
[data-testid="stSlider"] *{color:var(--text);}
[data-testid="stTickBarMin"],[data-testid="stTickBarMax"]{color:var(--muted) !important;}
[data-testid="stCheckbox"] *,[data-testid="stToggle"] *{color:var(--text) !important;}
/* ---- radio used as navigation pills ---- */
[data-testid="stRadio"] [role="radiogroup"]{gap:.45rem;flex-wrap:wrap;}
label[data-testid="stRadioOption"]{background:var(--surface);border:1px solid var(--border);border-radius:999px;
  padding:.35rem .95rem;margin:0 !important;}
label[data-testid="stRadioOption"] *{color:var(--text) !important;}
label[data-testid="stRadioOption"]:hover{background:var(--hover);}
label[data-testid="stRadioOption"][data-selected="true"]{border-color:#F2A93B;background:var(--hover);}
/* ---- buttons ---- */
.stApp button[data-testid^="stBaseButton"]{background:#F2A93B !important;border:none !important;
  border-radius:10px !important;font-weight:700 !important;padding:.5rem 1.4rem !important;}
.stApp button[data-testid^="stBaseButton"],.stApp button[data-testid^="stBaseButton"] *{color:#1E2761 !important;}
.stApp button[data-testid^="stBaseButton"]:hover{filter:brightness(1.07);}
/* ---- containers ---- */
[data-testid="stForm"]{background:var(--surface);border:1px solid var(--border) !important;border-radius:14px;padding:1rem 1.2rem;}
[data-testid="stExpander"]{background:var(--surface);border:1px solid var(--border) !important;border-radius:12px;}
[data-testid="stExpander"] details,[data-testid="stExpander"] summary{background:var(--surface2) !important;}
[data-testid="stExpander"] summary,[data-testid="stExpander"] summary *{color:var(--text) !important;}
[data-testid="stAlert"]{background:var(--surface2) !important;border:1px solid var(--border);border-radius:12px;}
[data-testid="stAlert"] *{color:var(--text) !important;}
/* ---- custom components ---- */
.banner{background:linear-gradient(120deg,$b1,$b2);border-radius:16px;padding:1.1rem 1.6rem;margin-bottom:.9rem;}
.banner h1{margin:0;font-size:1.65rem;} .banner p{margin:.25rem 0 0 0;}
.banner,.banner h1,.banner p,.banner span{color:#ffffff !important;}
.banner p{color:#d7dcff !important;}
.chip{display:inline-block;border:1px solid rgba(255,255,255,.4);border-radius:999px;padding:.12rem .75rem;
  margin:.45rem .35rem 0 0;font-size:.82rem;color:#fff !important;}
.badge{display:inline-block;background:#F2A93B;font-weight:700;border-radius:20px;padding:.1rem .8rem;
  font-size:.85rem;margin-right:.5rem;color:#1E2761 !important;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:1rem 1.3rem;
  margin-bottom:.8rem;box-shadow:0 4px 14px var(--shadow);}
.card,.card *{color:var(--text);}
.card.low{border-left:8px solid #2fb67c;} .card.med{border-left:8px solid #F2A93B;} .card.high{border-left:8px solid #e5484d;}
.rec{font-size:1.65rem;font-weight:800;margin:.1rem 0;}
.flag{padding:.35rem .65rem;border-radius:8px;margin:.25rem 0;font-size:.95rem;color:var(--text) !important;}
.flag.ok{background:var(--ok);} .flag.warn{background:var(--warn);} .flag.bad{background:var(--bad);}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.7rem;margin-bottom:.9rem;}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:.7rem .9rem;}
.kpi .v{font-size:1.5rem;font-weight:800;color:var(--text);} .kpi .l{font-size:.8rem;color:var(--muted);}
.table-wrap{overflow:auto;border:1px solid var(--border);border-radius:12px;background:var(--surface);
  box-shadow:0 4px 14px var(--shadow);margin-bottom:.9rem;}
.dt{border-collapse:collapse;width:100%;font-size:.88rem;}
.dt th{position:sticky;top:0;background:var(--th);color:var(--text);text-align:left;padding:.55rem .8rem;
  white-space:nowrap;border-bottom:1px solid var(--border);}
.dt td{padding:.48rem .8rem;border-bottom:1px solid var(--border);color:var(--text);white-space:nowrap;}
.dt tr:hover td{background:var(--hover);}
.vgrid{display:flex;flex-wrap:wrap;gap:3px;margin:.4rem 0 .8rem 0;}
.vgrid span{width:15px;height:15px;border-radius:4px;display:inline-block;}
.vgrid .yes{background:#e5484d;} .vgrid .no{background:#2fb67c;}
.legend span{display:inline-block;width:12px;height:12px;border-radius:3px;margin:0 .3rem 0 .8rem;vertical-align:middle;}
.step{border-left:4px solid #F2A93B;background:var(--surface);border-radius:8px;padding:.45rem .8rem;margin:.3rem 0;color:var(--text);}
.step b{color:var(--text);}
</style>""")


def is_dark() -> bool:
    return bool(st.session_state.get("dark", False))


def inject_css():
    st.markdown(CSS.substitute(**(DARK if is_dark() else LIGHT)), unsafe_allow_html=True)


def flash(kind: str, msg: str):
    st.session_state["flash"] = (kind, msg)


def show_flash():
    f = st.session_state.pop("flash", None)
    if f:
        {"success": st.success, "error": st.error, "info": st.info, "warning": st.warning}[f[0]](f[1])


def html_table(df, max_height=None):
    """Theme-aware table (st.dataframe can't follow our light/dark theme)."""
    def fmt(v):
        if isinstance(v, float):
            return "-" if v != v else (f"{v:,.3f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:,.0f}")
        return v
    df = df.map(fmt)
    body = df.to_html(index=False, border=0, classes="dt", escape=True, na_rep="-").replace("\n", "")
    style_ = f' style="max-height:{max_height}px"' if max_height else ""
    st.markdown(f'<div class="table-wrap"{style_}>{body}</div>', unsafe_allow_html=True)


def kpis(items):
    cells = "".join(f'<div class="kpi"><div class="v">{_html.escape(str(v))}</div>'
                    f'<div class="l">{_html.escape(l)}</div></div>' for l, v in items)
    st.markdown(f'<div class="kpis">{cells}</div>', unsafe_allow_html=True)


@st.cache_resource
def load_models():
    return {n: joblib.load(MODELS_DIR / f) for n, f in MODEL_FILES.items()}


@st.cache_data
def load_report():
    return json.loads((REPORTS_DIR / "metrics.json").read_text())


def band(score: int) -> str:
    return ("Exceptional" if score >= 800 else "Very good" if score >= 740 else
            "Good" if score >= 670 else "Fair" if score >= 580 else "Poor")


def decide(p: float):
    low, high = db.get_thresholds()
    if p >= high:
        return "Recommend Decline", "High Risk", "high"
    if p >= low:
        return "Refer for Manual Review", "Medium Risk", "med"
    return "Recommend Approval", "Low Risk", "low"


def risk_factors(a: dict) -> list:
    """Plain-language checks an officer can read at a glance (policy-style rules)."""
    f = []
    cs = a["credit_score"]
    f.append(("ok" if cs >= 670 else "warn" if cs >= 580 else "bad", f"Credit score {cs:.0f} ({band(cs)})"))
    d = a["debt_to_income"]
    f.append(("ok" if d <= 0.36 else "warn" if d <= 0.43 else "bad",
              f"Debt-to-income {d:.2f} incl. new EMI (0.36 comfortable, 0.43 maximum)"))
    lti = a["loan_to_income"]
    f.append(("ok" if lti <= 1 else "warn" if lti <= 2 else "bad", f"Loan is {lti:.1f}x annual income"))
    u = a["credit_utilization"]
    f.append(("ok" if u <= 30 else "warn" if u <= 60 else "bad", f"Credit utilisation {u:.0f}%"))
    lp = int(a["late_payments"])
    f.append(("ok" if lp == 0 else "warn" if lp <= 2 else "bad",
              "No late payments on record" if lp == 0 else f"{lp} late payment(s) on record"))
    f.append(("bad", "Prior default on record") if a["prior_default"] else ("ok", "No prior default"))
    q = int(a["recent_inquiries"])
    f.append(("ok" if q <= 2 else "warn" if q <= 4 else "bad", f"{q} recent credit inquiries"))
    h = a["credit_history_years"]
    f.append(("ok" if h >= 5 else "warn" if h >= 2 else "bad", f"Credit history {h:.1f} yrs"))
    if a["employment_type"] == "Unemployed":
        f.append(("bad", "Applicant is currently unemployed"))
    elif a["employment_years"] < 1:
        f.append(("warn", f"Short employment history ({a['employment_years']:.1f} yr)"))
    else:
        f.append(("ok", f"{a['employment_type']}, {a['employment_years']:.1f} yrs employment"))
    cover = a["savings_balance"] / max(a["emi"], 1)
    f.append(("ok" if cover >= 6 else "warn" if cover >= 2 else "bad",
              f"Savings cover {cover:.0f} months of the new EMI"))
    if a["collateral_value"] > 0:
        f.append(("ok", f"Collateral {a['collateral_value']:,.0f} ({a['collateral_value']/a['loan_amount']:.0%} of loan)"))
    if a["cosigner"] == "Yes":
        f.append(("ok", "Co-signer on the loan"))
    return f


def style(fig):
    d = is_dark()
    fc = "#eef1ff" if d else "#1b2559"
    grid = "rgba(255,255,255,.14)" if d else "rgba(27,37,89,.12)"
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(255,255,255,.04)" if d else "rgba(255,255,255,.75)",
                      font_color=fc, title_font_color=fc, legend_font_color=fc,
                      margin=dict(l=10, r=10, t=45, b=10))
    fig.update_xaxes(gridcolor=grid, zerolinecolor=grid, linecolor=grid, tickfont_color=fc, title_font_color=fc)
    fig.update_yaxes(gridcolor=grid, zerolinecolor=grid, linecolor=grid, tickfont_color=fc, title_font_color=fc)
    return fig


def render_result(r: dict, key: str = "x"):
    """r: ref, name, purpose, mode, p, rf, gb, rec, level, cls, applicant (+ optional status/review)."""
    low, high = db.get_thresholds()
    d = is_dark()
    st.markdown(f"### Assessment result · {r['ref']}")
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown(f"""<div class="card {r['cls']}">
        <div>{_html.escape(r['name'])} · {_html.escape(r['purpose'])} · scored by {_html.escape(r['mode'])}</div>
        <div class="rec">{r['rec']}</div>
        <div><b>{r['level']}</b> &nbsp;·&nbsp; Estimated default probability <b>{r['p']:.0%}</b></div>
        </div>""", unsafe_allow_html=True)
        if r.get("status"):
            line = f"**Status:** {r['status']}"
            if r.get("reviewer"):
                line += f" · reviewed by {r['reviewer']} on {r['reviewed_at']}"
            st.markdown(line)
            if r.get("review_comment"):
                st.info(f"Manager comment: {r['review_comment']}")
        a = r["applicant"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Random Forest", f"{r['rf']:.0%}")
        m2.metric("Gradient Boosting", f"{r['gb']:.0%}")
        m3.metric("Monthly EMI", f"{a['emi']:,.0f}")
        m4.metric("Debt-to-income", f"{a['debt_to_income']:.2f}")
        if abs(r["rf"] - r["gb"]) > 0.15:
            st.warning("The two models disagree noticeably on this applicant. "
                       "Consider a manual review regardless of the recommendation.")
        st.markdown("**Key factors**")
        for lvl, text in risk_factors(a):
            icon = {"ok": "✅", "warn": "⚠️", "bad": "🚩"}[lvl]
            st.markdown(f'<div class="flag {lvl}">{icon} {_html.escape(text)}</div>', unsafe_allow_html=True)
    with right:
        g = go.Figure(go.Indicator(
            mode="gauge+number", value=r["p"] * 100, number={"suffix": "%"},
            title={"text": "Default probability"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": GOLD if d else NAVY},
                   "steps": [{"range": [0, low * 100], "color": "rgba(47,182,124,.35)"},
                             {"range": [low * 100, high * 100], "color": "rgba(242,169,59,.35)"},
                             {"range": [high * 100, 100], "color": "rgba(229,72,77,.35)"}]}))
        g.update_layout(height=300)
        st.plotly_chart(style(g), width="stretch", key=f"gauge_{key}")
    st.caption("Decision support only. The final lending decision rests with the authorised credit "
               "officer. These models were trained on synthetic demo data and must not be used for "
               "real lending decisions.")
