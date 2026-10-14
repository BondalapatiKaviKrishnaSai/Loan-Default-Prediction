from pathlib import Path

BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "loan_data.csv"
MODELS_DIR = BASE / "models"
REPORTS_DIR = BASE / "reports"

TARGET = "default"
RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERIC = [
    # applicant
    "age", "dependents", "employment_years", "income",
    # credit profile
    "credit_score", "credit_history_years", "open_credit_accounts",
    "credit_utilization", "recent_inquiries", "late_payments", "prior_default",
    # finances
    "existing_debt", "monthly_debt_payments", "savings_balance",
    # loan
    "loan_amount", "loan_term_months", "interest_rate", "collateral_value",
    # derived ratios
    "emi", "debt_to_income", "loan_to_income",
]
CATEGORICAL = ["gender", "education", "marital_status", "employment_type",
               "home_ownership", "loan_purpose", "cosigner"]

CATEGORY_OPTIONS = {
    "gender": ["Male", "Female"],
    "education": ["High School", "Graduate", "Postgraduate"],
    "marital_status": ["Single", "Married", "Divorced"],
    "employment_type": ["Salaried", "Self-Employed", "Unemployed"],
    "home_ownership": ["Rent", "Mortgage", "Own"],
    "loan_purpose": ["Home Loan", "Personal Loan", "Auto Loan",
                     "Education Loan", "Business Loan", "Other"],
    "cosigner": ["No", "Yes"],
}


def compute_emi(amount, annual_rate_pct, months):
    """Monthly instalment (works with scalars or numpy arrays)."""
    r = annual_rate_pct / 1200.0
    return amount * r / (1 - (1 + r) ** (-months))


def compute_ratios(income, loan_amount, emi, monthly_debt_payments):
    """Debt-to-income (monthly obligations incl. new EMI / monthly income), loan-to-income."""
    dti = (monthly_debt_payments + emi) / (income / 12.0)
    return round(dti, 3), round(loan_amount / income, 3)
