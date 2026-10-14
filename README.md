# Loan Default Prediction — Random Forest vs Gradient Boosting

ML project (KL University, Dept. of AIDS, A.Y. 2026-27) that predicts loan default,
compares two ensemble models, and serves them in a Streamlit web app.

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py        # models and data are already included
```
To rebuild the data and models: `python generate_data.py` then `python train.py`.

## Roles and demo logins
The first run creates `data/bank.db` (SQLite) with these accounts. Each role has its own login tab.

| Portal | Username | Password | What they can do |
|---|---|---|---|
| Loan Officer | `officer1`, `officer2` | `Officer@123` | Enter applicant details, see the AI assessment, track their own applications |
| Branch Manager | `manager1` | `Manager@123` | Review queue (approve/reject referred cases with a comment), branch dashboard, all branch applications |
| Credit Analyst | `analyst1` | `Analyst@123` | Portfolio analytics across branches, all applications, model performance |
| Administrator | `admin` | `Admin@123` | Live data feed + dataset explorer, Random Forest Lab (live tree votes), users, decision cut-offs, audit log, sample data |

Workflow: low-risk applications are cleared automatically; medium and high-risk ones go to the
Branch Manager's review queue. Overriding the model (e.g. approving a "Decline" recommendation)
is flagged and recorded. Every sign-in, assessment, decision and admin action is in the audit log.

Security notes: passwords are salted PBKDF2 hashes, accounts lock after 5 failed sign-ins, and
deactivating an account takes effect immediately. This is demo-grade; change the default passwords
and use SSO plus a managed database for anything real. Delete `data/bank.db` to reset all accounts and records.

## Dark mode
Use the **🌙 Dark mode** switch at the top right of any page (login included). The choice is kept when you sign out.

## Random Forest Lab (Administrator)
Shows the forest's settings, explains how it works, and re-runs all 250 trees live as you move sliders (each
square is one tree's vote), plus a single-tree decision path and an auto-refreshing monitor of real applications.

## Project layout
| File | Purpose |
|---|---|
| `app.py` | Login portals and role-based navigation |
| `views.py` | Pages for each role (assessment form, review queue, dashboards, admin tools, live data) |
| `forest_lab.py` | Admin Random Forest Lab: live per-tree votes and explanations |
| `.streamlit/config.toml` | Light base theme, hides the Streamlit menu (the app has its own dark mode) |
| `ui.py` | Styling, risk-factor rules, result card |
| `db.py` | SQLite: users, authentication, assessments, audit log, settings |
| `config.py` | Paths, feature lists, EMI / ratio formulas |
| `generate_data.py` | Synthetic applicant data (replace with a real CSV if you have one) |
| `train.py` | Cleaning, encoding, scaling, stratified split, train and evaluate RF and GB |

## Applicant parameters (28 inputs)
| Group | Fields |
|---|---|
| Loan request | purpose, amount, term (months), interest rate, collateral value, co-signer |
| Applicant | age, gender, education, marital status, dependents |
| Employment & income | employment type, years employed, annual income, home ownership |
| Credit profile | credit score, credit history (yrs), open accounts, utilisation %, recent inquiries |
| Finances & repayment | total existing debt, monthly debt payments, savings balance, late payments, prior default |
| Derived (auto) | monthly EMI, debt-to-income (incl. new EMI), loan-to-income |

## Pipeline (matches the report)
1. **Cleaning** - dedupe, normalise text, fix invalid values, median/mode imputation
2. **Encoding** - one-hot for categorical fields
3. **Scaling** - `StandardScaler` on numeric features
4. **Stratified 80/20 split** - preserves defaulter ratio
5. **Models** - `RandomForestClassifier` (bagging) and `GradientBoostingClassifier` (boosting)
6. **Metrics** - accuracy, precision, recall, F1, ROC-AUC, confusion matrices, ROC curves

## Decision rule in the app
Average (or selected-model) default probability: below 25% -> approve, 25-50% -> manual review,
50% or above -> decline. Change `LOW_T` / `HIGH_T` at the top of `app.py`.

## Using a real dataset
Save your CSV to `data/loan_data.csv` with a binary `default` column, and edit the
`NUMERIC` / `CATEGORICAL` lists in `config.py` to match your columns. Then run
`train.py`. In `app.py`, update the input form to the same fields.

## Results on the included synthetic data
Gradient Boosting ROC-AUC ~ 0.88, Random Forest ROC-AUC ~ 0.88 (see the Model Insights tab for exact figures).
Top drivers: credit score, debt-to-income ratio, late payments.
