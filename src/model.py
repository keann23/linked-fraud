"""Phase 3: transaction-level LightGBM baseline with an honest time split.

Split (fixed for the whole project):
    train = days 1-120 (about 4 months), test = days 121-182 (about 2 months)
    Inside train, days 91-120 are held out only to pick the number of trees,
    then the model is refit on all of days 1-120.

Review budget (fixed for the whole project): the top 2% of test transactions
by score are "reviewed". Recall at budget = share of test fraud inside that 2%.
"""
from __future__ import annotations

import time

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import VELOCITY_PARQUET
from identity import IDENTITY_PARQUET
from load import BASE_PARQUET, PROCESSED

TRAIN_LAST_DAY = 120
VALID_FIRST_DAY = 91
BUDGET = 0.02
SEED = 42
PREDICTIONS_PARQUET = PROCESSED / "predictions.parquet"

# Never model inputs: row ids, raw time (would learn the calendar), the label,
# and the uid itself (a high-cardinality id that would memorize customers).
EXCLUDE = {"TransactionID", "TransactionDT", "day", "isFraud", "uid", "uid_email",
           "uid_card1", "has_identity"}

PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=256, min_child_samples=100,
    feature_fraction=0.4, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    max_cat_to_onehot=8, cat_smooth=20, verbose=-1, seed=SEED, num_threads=0,
)


def load_frame() -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    df = con.sql(f"""
        SELECT b.*,
               CAST(FLOOR((b.TransactionDT % 86400) / 3600) AS INTEGER) AS hour,
               ROUND(b.TransactionAmt - FLOOR(b.TransactionAmt), 3)    AS amt_cents,
               v.* EXCLUDE (TransactionID),
               i.uid
        FROM read_parquet('{BASE_PARQUET}') b
        JOIN read_parquet('{VELOCITY_PARQUET}') v USING (TransactionID)
        JOIN read_parquet('{IDENTITY_PARQUET}') i USING (TransactionID)
        ORDER BY TransactionID
    """).df()
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype("category")
    return df


VELOCITY_COLS = [
    "card1_txn_1h", "card1_txn_24h", "card1_txn_7d", "card1_secs_since_prev",
    "uid_txn_1h", "uid_txn_24h", "uid_txn_7d", "uid_txn_number", "uid_secs_since_prev",
    "uid_distinct_emails", "uid_distinct_devices", "uid_prior_avg_amt", "uid_amt_vs_avg",
]


def feature_cols(df: pd.DataFrame, velocity: bool = True, extra: list[str] | None = None) -> list[str]:
    cols = [c for c in df.columns if c not in EXCLUDE and c not in VELOCITY_COLS]
    if velocity:
        cols += VELOCITY_COLS
    cols = [c for c in cols if c not in (extra or [])] + list(extra or [])
    return cols


def split(df: pd.DataFrame):
    train = df[df.day <= TRAIN_LAST_DAY]
    test = df[df.day > TRAIN_LAST_DAY]
    return train, test


def fit(train: pd.DataFrame, cols: list[str], max_rounds: int = 3000) -> lgb.Booster:
    """Pick tree count on days 91-120, then refit on all of days 1-120."""
    fit_part = train[train.day < VALID_FIRST_DAY]
    val_part = train[train.day >= VALID_FIRST_DAY]
    dfit = lgb.Dataset(fit_part[cols], fit_part.isFraud)
    dval = lgb.Dataset(val_part[cols], val_part.isFraud, reference=dfit)
    probe = lgb.train({**PARAMS, "metric": "auc"}, dfit, max_rounds, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    n = probe.best_iteration
    full = lgb.train(PARAMS, lgb.Dataset(train[cols], train.isFraud), n)
    full.best_iteration_probe = n  # type: ignore[attr-defined]
    full.best_auc_probe = probe.best_score["valid_0"]["auc"]  # type: ignore[attr-defined]
    return full


def flag_top(scores: pd.Series | np.ndarray, budget: float = BUDGET) -> np.ndarray:
    """True for the top `budget` share of rows by score (ties broken by order)."""
    s = np.asarray(scores)
    k = int(round(budget * len(s)))
    idx = np.argsort(-s, kind="stable")[:k]
    out = np.zeros(len(s), dtype=bool)
    out[idx] = True
    return out


def evaluate(test: pd.DataFrame, scores, budget: float = BUDGET) -> dict:
    flagged = flag_top(scores, budget)
    y = test.isFraud.to_numpy().astype(bool)
    amt = test.TransactionAmt.to_numpy()
    return {
        "auc": round(roc_auc_score(y, scores), 4),
        "flagged": int(flagged.sum()),
        "fraud_total": int(y.sum()),
        "fraud_caught": int((flagged & y).sum()),
        "recall_at_budget": round((flagged & y).sum() / y.sum(), 4),
        "precision_at_budget": round((flagged & y).sum() / flagged.sum(), 4),
        "fraud_amt_total": round(float(amt[y].sum()), 0),
        "fraud_amt_caught": round(float(amt[flagged & y].sum()), 0),
    }


def run_baselines(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    train, test = split(df)
    results, models = {}, {}
    for name, vel in [("transaction features only", False), ("+ velocity features", True)]:
        t = time.time()
        cols = feature_cols(df, velocity=vel)
        m = fit(train, cols)
        scores = m.predict(test[cols])
        r = evaluate(test, scores)
        r.update(n_features=len(cols), trees=m.best_iteration_probe,
                 valid_auc=round(m.best_auc_probe, 4), minutes=round((time.time() - t) / 60, 1))
        results[name], models[name] = r, (m, cols, scores)
    return pd.DataFrame(results), models


def save_predictions(test: pd.DataFrame, scores, name: str = "baseline") -> None:
    out = pd.DataFrame({"TransactionID": test.TransactionID.values, f"score_{name}": scores,
                        f"flag_{name}": flag_top(scores)})
    if PREDICTIONS_PARQUET.exists():
        prev = pd.read_parquet(PREDICTIONS_PARQUET)
        prev = prev.drop(columns=[c for c in out.columns if c != "TransactionID" and c in prev], errors="ignore")
        out = prev.merge(out, on="TransactionID")
    out.to_parquet(PREDICTIONS_PARQUET, index=False)


if __name__ == "__main__":
    t = time.time()
    df = load_frame()
    print(f"frame {df.shape} in {time.time() - t:.0f}s")
    train, test = split(df)
    print(f"train {len(train):,} rows ({train.isFraud.mean():.2%} fraud), "
          f"test {len(test):,} rows ({test.isFraud.mean():.2%} fraud)")
    res, models = run_baselines(df)
    print(res.to_string())
    m, cols, scores = models["+ velocity features"]
    save_predictions(test, scores, "baseline")
    m.save_model(str(PROCESSED / "baseline.txt"))


# Plain-English names for the anonymized column families (per Vesta's data notes).
FAMILY = {"C": "count feature", "D": "time-delta feature", "V": "Vesta engineered", "M": "match flag",
          "id_": "identity/device field"}
READABLE = {
    "TransactionAmt": "transaction amount", "ProductCD": "product type", "card1": "card number (hashed)",
    "card2": "card attribute 2", "card5": "card attribute 5", "card6": "card type (debit/credit)",
    "addr1": "billing region", "addr2": "billing country", "dist1": "distance 1", "dist2": "distance 2",
    "P_emaildomain": "purchaser email domain", "R_emaildomain": "recipient email domain",
    "hour": "hour of day", "amt_cents": "cents of amount", "DeviceInfo": "device model",
    "card1_txn_1h": "card1 txns, last hour", "card1_txn_24h": "card1 txns, last 24h",
    "card1_txn_7d": "card1 txns, last 7 days", "card1_secs_since_prev": "card1 time since last txn",
    "uid_txn_1h": "customer txns, last hour", "uid_txn_24h": "customer txns, last 24h",
    "uid_txn_7d": "customer txns, last 7 days", "uid_txn_number": "customer's nth txn",
    "uid_secs_since_prev": "customer time since last txn", "uid_distinct_emails": "customer distinct emails",
    "uid_distinct_devices": "customer distinct devices", "uid_prior_avg_amt": "customer avg past amount",
    "uid_amt_vs_avg": "amount vs customer's average",
}


def readable(col: str) -> str:
    if col in READABLE:
        return f"{READABLE[col]}  ({col})" if col not in VELOCITY_COLS else READABLE[col]
    for prefix, fam in FAMILY.items():
        if col.startswith(prefix) and col[len(prefix):].isdigit():
            return f"{col}  ({fam})"
    return col


def shap_importance(model: lgb.Booster, X: pd.DataFrame, n: int = 20000) -> pd.Series:
    import shap
    sample = X.sample(min(n, len(X)), random_state=SEED)
    sv = shap.TreeExplainer(model).shap_values(sample)
    if isinstance(sv, list):
        sv = sv[1]
    return pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)


def shap_title(imp: pd.Series) -> tuple[str, str]:
    vel = imp.index.isin(VELOCITY_COLS)
    share = imp[vel].sum() / imp.sum()
    best = imp[vel].index[0]
    rank = list(imp.index).index(best) + 1
    return ("The model leans on amount, card and email signals",
            f"Top 15 of {len(imp)} features. The SQL velocity features add only {share:.1%} of total impact; "
            f"{READABLE.get(best, best).replace('card1 txns, last 7 days', 'card1 activity in the last 7 days')} ranks highest (#{rank}).")


def plot_shap(imp: pd.Series, title: str, subtitle: str, top: int = 15):
    import matplotlib.pyplot as plt
    import viz

    viz.use_style()
    d = imp.head(top)[::-1]
    colors = [viz.BLUE if c in VELOCITY_COLS else viz.MUTED for c in d.index]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh([readable(c) for c in d.index], d.values, color=colors, height=0.72)
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", labelsize=11.5, colors=viz.TEXT)
    ax.set_xlabel("Average impact on the fraud score (mean |SHAP|)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=viz.BLUE, label="Velocity feature (built in SQL)"),
                       Patch(color=viz.MUTED, label="Original transaction field")],
              loc="lower right", fontsize=11)
    viz.headline(fig, title, subtitle)
    viz.source_note(fig, "SHAP on 20,000 test-period transactions. Data: IEEE-CIS Fraud Detection (Vesta)")
    fig.tight_layout()
    return fig
