import logging
import warnings
from pathlib import Path
 
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from GPyOpt.methods import BayesianOptimization

# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def load_data():
    """Load the California Housing dataset and return X, y."""
    logger.info("Loading California Housing dataset...")
    dataset = fetch_california_housing()
    X, y = dataset.data, dataset.target
    logger.info(f"  Features : {dataset.feature_names}")
    logger.info(f"  X shape  : {X.shape}")
    logger.info(f"  y range  : [{y.min():.2f}, {y.max():.2f}]")
    return X[:1000], y[:1000], dataset.feature_names
 
 
def split_data(X, y, test_size=0.3, random_state=2024):
    """Split into train / test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Train shape: {X_train.shape}  |  Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test
 
 
def preprocess(X_train, X_test):
    """
    Fit a StandardScaler on training data and transform both splits.
    Returns scaled arrays and the fitted scaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Features standardised (mean=0, std=1).")
    return X_train_scaled, X_test_scaled, scaler
 
 
def evaluate(model, X_test, y_test, label="Model"):
    """Return and log MSE and R² for a fitted model."""
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    logger.info(f"[{label}]  MSE={mse:.4f}  |  R²={r2:.4f}  |  RMSE={mse**0.5:.4f}")
    return {"label": label, "mse": mse, "r2": r2, "rmse": mse ** 0.5}
 
 
def cross_validate(model, X_train, y_train, cv=3, label="Model"):
    """Run CV and return mean ± std of MSE."""
    scores = -cross_val_score(
        model, X_train, y_train,
        scoring="neg_mean_squared_error", cv=cv
    )
    logger.info(f"[{label}] CV MSE: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores
 
 
def train_baseline(X_train, y_train, X_test, y_test):
    """Train an XGBoost model with default parameters as a baseline."""
    logger.info("--- Baseline (default XGBoost) ---")
    model = xgb.XGBRegressor(random_state=2024, verbosity=0)
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test, label="Baseline")
    return model, metrics
 
 
#Random Search
 
PARAM_DIST = {
    "max_depth": [3, 5, 10, 15],
    "min_child_weight": [1, 5, 10],
    "subsample": [0.5, 0.7, 1.0],
    "colsample_bytree": [0.5, 0.7, 1.0],
    "n_estimators": [100, 200, 300, 400],
    "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
}
 
 
def random_search_tuning(X_train, y_train, X_test, y_test,
                         n_iter=2, cv=3, random_state=2024):
    """Run RandomizedSearchCV and return the best model + metrics."""
    logger.info("--- Random Search ---")
    base = xgb.XGBRegressor(random_state=random_state, verbosity=0)
    rs = RandomizedSearchCV(
        base,
        param_distributions=PARAM_DIST,
        n_iter=n_iter,
        scoring="neg_mean_squared_error",
        cv=cv,
        verbose=1,
        random_state=random_state,
        n_jobs=-1,
    )
    rs.fit(X_train, y_train)
    logger.info(f"Best params: {rs.best_params_}")
 
    model = xgb.XGBRegressor(**rs.best_params_, random_state=random_state, verbosity=0)
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test, label="Random Search")
    return model, rs.best_params_, metrics
 
 
# Bayesian Optimization
BAYES_BOUNDS = [
    {"name": "max_depth",         "type": "discrete",   "domain": (3, 5, 10, 15)},
    {"name": "min_child_weight",  "type": "discrete",   "domain": (1, 5, 10)},
    {"name": "subsample",         "type": "continuous", "domain": (0.5, 1.0)},
    {"name": "colsample_bytree",  "type": "continuous", "domain": (0.5, 1.0)},
    {"name": "n_estimators",      "type": "discrete",   "domain": (100, 200, 300, 400)},
    {"name": "learning_rate",     "type": "continuous", "domain": (0.01, 0.2)},
]
 
 
def _make_xgb_cv_scorer(X_train, y_train, cv=3):
    """Return a closure compatible with GPyOpt (minimises CV MSE)."""
    def scorer(parameters):
        p = parameters[0]
        model = xgb.XGBRegressor(
            max_depth=int(p[0]),
            min_child_weight=int(p[1]),
            subsample=float(p[2]),
            colsample_bytree=float(p[3]),
            n_estimators=int(p[4]),
            learning_rate=float(p[5]),
            verbosity=0,
        )
        return -cross_val_score(
            model, X_train, y_train,
            scoring="neg_mean_squared_error", cv=cv
        ).mean()
    return scorer
 
 
def bayesian_opt_tuning(X_train, y_train, X_test, y_test,
                        max_iter=2, cv=3, random_state=2024):
    """Run Bayesian Optimisation with GPyOpt and return best model + metrics."""
 
    logger.info("--- Bayesian Optimisation ---")
    scorer = _make_xgb_cv_scorer(X_train, y_train, cv=cv)
 
    optimizer = BayesianOptimization(
        f=scorer,
        domain=BAYES_BOUNDS,
        model_type="GP",
        acquisition_type="EI",
        max_iter=max_iter,
        verbosity=False,
    )
    optimizer.run_optimization()
 
    param_names = [d["name"] for d in BAYES_BOUNDS]
    best_params = dict(zip(param_names, optimizer.x_opt))
    for p in ("max_depth", "min_child_weight", "n_estimators"):
        best_params[p] = int(best_params[p])
 
    logger.info(f"Best params: {best_params}")
 
    model = xgb.XGBRegressor(**best_params, random_state=random_state, verbosity=0)
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test, label="Bayesian Opt")
    return model, best_params, metrics
 
 
def compare_models(results: list[dict], save_dir: Path | None = None):
    """
    Print a comparison table and plot MSE / R² bar charts.
 
    Parameters
    ----------
    results   : list of dicts with keys label, mse, r2, rmse
    save_dir  : optional directory to save the plot
    """
    print("\n" + "=" * 55)
    print(f"{'Model':<22} {'MSE':>8} {'RMSE':>8} {'R²':>8}")
    print("-" * 55)
    for r in results:
        print(f"{r['label']:<22} {r['mse']:>8.4f} {r['rmse']:>8.4f} {r['r2']:>8.4f}")
    print("=" * 55)
 
    labels = [r["label"] for r in results]
    mse_vals = [r["mse"] for r in results]
    r2_vals = [r["r2"] for r in results]
    x = np.arange(len(labels))
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
    axes[0].bar(x, mse_vals, color=["steelblue", "darkorange", "seagreen"][: len(x)])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15)
    axes[0].set_title("Mean Squared Error (lower is better)")
    axes[0].set_ylabel("MSE")
 
    axes[1].bar(x, r2_vals, color=["steelblue", "darkorange", "seagreen"][: len(x)])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15)
    axes[1].set_title("R² Score (higher is better)")
    axes[1].set_ylabel("R²")
    axes[1].set_ylim(0, 1)
 
    plt.tight_layout()
 
    if save_dir:
        path = Path(save_dir) / "model_comparison.png"
        plt.savefig(path, dpi=150)
        logger.info(f"Comparison chart saved → {path}")
    plt.show()
 
 
def plot_feature_importance(model, feature_names, label="Model",
                            save_dir: Path | None = None):
    """Bar chart of XGBoost feature importances."""
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
 
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(np.arange(len(importances)),
           importances[order],
           color="steelblue")
    ax.set_xticks(np.arange(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in order], rotation=40, ha="right")
    ax.set_title(f"Feature Importances — {label}")
    ax.set_ylabel("Importance")
    plt.tight_layout()
 
    if save_dir:
        path = Path(save_dir) / f"feature_importance_{label.replace(' ', '_')}.png"
        plt.savefig(path, dpi=150)
        logger.info(f"Feature importance chart saved → {path}")
    plt.show()
 
 

def save_best_model(model, path: str = "best_model.json"):
    """Persist the XGBoost model to disk."""
    model.save_model(path)
    logger.info(f"Best model saved → {path}")
 

def run_pipeline(
    test_size: float = 0.3,
    random_state: int = 2024,
    n_iter_random: int = 25,
    n_iter_bayes: int = 25,
    cv: int = 3,
    scale_features: bool = True,
    output_dir: str = "outputs",
):
    """End-to-end optimization pipeline."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
 
    # 1. Load
    X, y, feature_names = load_data()
 
    # 2. Split
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )
 
    # 3. Preprocess
    if scale_features:
        X_train, X_test, _ = preprocess(X_train, X_test)
 
    all_results = []
 
    # 4. Baseline
    baseline_model, baseline_metrics = train_baseline(X_train, y_train, X_test, y_test)
    all_results.append(baseline_metrics)
 
    # 5. Random Search
    rs_model, rs_params, rs_metrics = random_search_tuning(
        X_train, y_train, X_test, y_test,
        n_iter=n_iter_random, cv=cv, random_state=random_state
    )
    all_results.append(rs_metrics)
 
    # 6. Bayesian Optimisation
    bo_model, bo_params, bo_metrics = bayesian_opt_tuning(
        X_train, y_train, X_test, y_test,
        max_iter=n_iter_bayes, cv=cv, random_state=random_state
    )
    if bo_model is not None:
        all_results.append(bo_metrics)
 
    # 7. Compare
    compare_models(all_results, save_dir=out)
 
    # 8. Feature importance for best model
    best_model = bo_model if bo_model is not None else rs_model
    best_label = "Bayesian Opt" if bo_model is not None else "Random Search"
    plot_feature_importance(best_model, list(feature_names),
                            label=best_label, save_dir=out)
 
    # 9. Save best model
    save_best_model(best_model, path=str(out / "best_model.json"))
 
    return best_model, all_results
 
 
if __name__ == "__main__":
	config = {
	"test_size":      0.3,
	"random_state":   2024,
	"n_iter_random":  25,
	"n_iter_bayes":   25,
	"cv":             3,
	"scale_features": True,
	"output_dir":     "outputs",
	}
	run_pipeline(**config)
 
 