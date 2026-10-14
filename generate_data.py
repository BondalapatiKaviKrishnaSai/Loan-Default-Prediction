"""Generate a synthetic loan-applicant dataset (data/loan_data.csv).

Replace the output with a real dataset (e.g. Kaggle "Loan Default Prediction") by
keeping the column names, or edit config.py.
"""
import numpy as np
import pandas as pd

from config import DATA_PATH, RANDOM_STATE, compute_emi


def generate(n: int = 25000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pick = lambda opts, p=None: rng.choice(opts, n, p=p)

    age = rng.integers(21, 66, n)
    gender = pick(["Male", "Female"])
    education = pick(["High School", "Graduate", "Postgraduate"], [0.35, 0.45, 0.20])
    marital = pick(["Single", "Married", "Divorced"], [0.35, 0.50, 0.15])
    emp_type = pick(["Salaried", "Self-Employed", "Unemployed"], [0.65, 0.30, 0.05])
    home = pick(["Rent", "Mortgage", "Own"], [0.40, 0.35, 0.25])
    purpose = pick(["Home Loan", "Personal Loan", "Auto Loan", "Education Loan",
                    "Business Loan", "Other"], [0.20, 0.30, 0.20, 0.10, 0.12, 0.08])
    cosigner = pick(["No", "Yes"], [0.88, 0.12])
    dependents = np.clip(rng.poisson(1.1, n), 0, 6)

    income = np.clip(rng.lognormal(10.9, 0.55, n), 12000, 400000).round(0)
    emp_years = np.clip((age - 21) * rng.uniform(0.1, 0.9, n), 0, 40).round(1)
    emp_years[emp_type == "Unemployed"] = 0

    credit_score = np.clip(rng.normal(680, 75, n) + 0.0001 * (income - 60000), 300, 850).round(0)
    hist_years = np.clip((age - 20) * rng.uniform(0.15, 0.8, n), 0.5, 40).round(1)
    accounts = np.clip(rng.poisson(4, n) + 1, 1, 15)
    utilization = np.clip(rng.normal(45 - (credit_score - 680) * 0.12, 20, n), 0, 100).round(0)
    inquiries = np.clip(rng.poisson(np.clip((720 - credit_score) / 120, 0.1, None)), 0, 10)
    late = np.clip(rng.poisson(np.clip((720 - credit_score) / 90, 0.05, None)), 0, 12)
    prior_default = (rng.random(n) < np.clip(0.03 + (650 - credit_score) / 800, 0.01, 0.4)).astype(int)

    monthly_debt = (income / 12 * rng.uniform(0.0, 0.40, n)).round(0)
    existing_debt = (monthly_debt * rng.uniform(12, 60, n)).round(0)
    savings = (income * rng.uniform(0.0, 1.2, n) * (1 + (credit_score - 600) / 600)).clip(0).round(0)

    loan_amount = np.clip(income * rng.uniform(0.2, 2.5, n), 2000, 600000).round(0)
    term = rng.choice([12, 24, 36, 48, 60, 120, 180, 240], n, p=[.08, .12, .25, .15, .15, .10, .08, .07])
    rate = np.clip(6 + (750 - credit_score) * 0.025 + rng.normal(0, 1.2, n), 5, 24).round(2)
    collateral = np.where(np.isin(purpose, ["Home Loan", "Auto Loan"]),
                          loan_amount * rng.uniform(0.6, 1.4, n), 0).round(0)

    emi = compute_emi(loan_amount, rate, term).round(0)
    dti = ((monthly_debt + emi) / (income / 12)).round(3)
    lti = (loan_amount / income).round(3)

    z = (-1.9
         - 0.016 * (credit_score - 680)
         + 2.4 * (dti - 0.5)
         + 0.36 * late
         + 1.3 * prior_default
         + 0.012 * (utilization - 45)
         + 0.10 * inquiries
         - 0.04 * hist_years
         - 0.05 * emp_years
         + 0.9 * (emp_type == "Unemployed") + 0.15 * (emp_type == "Self-Employed")
         + 0.10 * dependents
         - 0.45 * (home == "Own") + 0.15 * (home == "Rent")
         - 0.6 * (cosigner == "Yes")
         - 0.5 * (collateral > 0)
         - 0.35 * np.log1p(savings / income * 4)
         + 0.25 * np.clip(lti - 1.5, 0, None)
         + 0.10 * (education == "High School")
         + 0.10 * (purpose == "Business Loan")
         + rng.normal(0, 0.6, n))
    default = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    df = pd.DataFrame({
        "age": age, "gender": gender, "education": education, "marital_status": marital,
        "dependents": dependents, "employment_type": emp_type, "employment_years": emp_years,
        "income": income, "home_ownership": home, "credit_score": credit_score,
        "credit_history_years": hist_years, "open_credit_accounts": accounts,
        "credit_utilization": utilization, "recent_inquiries": inquiries,
        "late_payments": late, "prior_default": prior_default, "existing_debt": existing_debt,
        "monthly_debt_payments": monthly_debt, "savings_balance": savings,
        "loan_purpose": purpose, "loan_amount": loan_amount, "loan_term_months": term,
        "interest_rate": rate, "collateral_value": collateral, "cosigner": cosigner,
        "emi": emi, "debt_to_income": dti, "loan_to_income": lti, "default": default})

    # realistic messiness for the cleaning step
    for col, frac in [("income", 0.02), ("credit_score", 0.015), ("employment_years", 0.02),
                      ("savings_balance", 0.02), ("credit_utilization", 0.015)]:
        df.loc[rng.random(n) < frac, col] = np.nan
    df.loc[rng.random(n) < 0.01, "gender"] = " male "
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df):,} rows, {df.shape[1]-1} features -> {DATA_PATH} | default rate: {df['default'].mean():.1%}")
