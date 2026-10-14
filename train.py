"""Train + evaluate Random Forest and Gradient Boosting; save models and reports."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (CATEGORICAL, DATA_PATH, MODELS_DIR, NUMERIC, RANDOM_STATE,
                    REPORTS_DIR, TARGET, TEST_SIZE)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Step 1 - Data cleaning: duplicates, inconsistent text, impossible values."""
    df = df.drop_duplicates().copy()
    for c in CATEGORICAL:
        df[c] = df[c].astype("string").str.strip().str.title()
    df["credit_score"] = df["credit_score"].clip(300, 850)
    df.loc[df["income"] <= 0, "income"] = np.nan
    return df.dropna(subset=[TARGET])


def build_preprocessor() -> ColumnTransformer:
    """Steps 1-3: impute missing values, encode categoricals, scale numerics."""
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])


def make_models() -> dict:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=12, min_samples_leaf=10,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=250, learning_rate=0.08, max_depth=3,
            subsample=0.9, random_state=RANDOM_STATE),
    }


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    df = clean(pd.read_csv(DATA_PATH))
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET].astype(int)

    # Step 4 - Stratified train/test split preserves defaulter ratio
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f"Train {len(X_tr):,} | Test {len(X_te):,} | default rate {y.mean():.1%}")

    metrics, curves, importances = {}, {}, {}
    for name, clf in make_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", clf)])
        pipe.fit(X_tr, y_tr)

        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y_te, proba)

        metrics[name] = {
            "accuracy": accuracy_score(y_te, pred),
            "precision": precision_score(y_te, pred),
            "recall": recall_score(y_te, pred),
            "f1": f1_score(y_te, pred),
            "roc_auc": roc_auc_score(y_te, proba),
            "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        }
        curves[name] = {"fpr": fpr[::max(1, len(fpr) // 200)].tolist(),
                        "tpr": tpr[::max(1, len(tpr) // 200)].tolist()}

        # feature importance, aggregated back to original columns
        names = pipe.named_steps["prep"].get_feature_names_out()
        imp = pd.Series(clf.feature_importances_, index=names)
        agg = {}
        for n, v in imp.items():
            base = n.split("__", 1)[1]
            for cat in CATEGORICAL:
                if base.startswith(cat + "_"):
                    base = cat
            agg[base] = agg.get(base, 0) + float(v)
        importances[name] = dict(sorted(agg.items(), key=lambda kv: -kv[1]))

        joblib.dump(pipe, MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib", compress=3)
        print(f"\n{name}")
        for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            print(f"  {k:<10}{metrics[name][k]:.4f}")

    (REPORTS_DIR / "metrics.json").write_text(json.dumps(
        {"metrics": metrics, "roc": curves, "importance": importances}, indent=2))
    print("\nSaved models/ and reports/metrics.json")


if __name__ == "__main__":
    main()
