"""Train one XGBoost regressor per target with Optuna tuning."""
from pathlib import Path
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.1
N_TRIALS = 10

def load_df(
    features_path: Path | str = "data/processed/features.csv",
    targets_path: Path | str = "data/processed/targets.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(features_path, index_col=0)
    targets = pd.read_csv(targets_path, index_col=0)
    features.columns = features.columns.str.strip()
    targets.columns = targets.columns.str.strip()
    return features, targets


def tune_and_train(target_name: str, dtrain, dval, n_trials: int = 10):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 3.0, 12.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "lambda": trial.suggest_float("lambda", 1e-3, 2.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 0.5, log=True),
            "nthread": 4,
        }
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=800,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        score = float(booster.best_score) if booster.best_score is not None else np.inf
        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params.update(
        {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "nthread": 4,
        }
    )

    booster = xgb.train(
        params=best_params,
        dtrain=dtrain,
        num_boost_round=1200,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    return booster, best_params


if __name__ == "__main__":
    df_features, df_targets = load_df()

    df_features_enc = pd.get_dummies(
        df_features,
        columns=["cluster_id"],
        prefix="cluster",
        dtype=float,
    )

    X_train, X_val, y_train_all, y_val_all = train_test_split(
        df_features_enc, df_targets, test_size=TEST_SIZE, random_state=42
    )

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, float] = {}
    sample_preds_map: dict[str, np.ndarray] = {}
    sample_true_map: dict[str, np.ndarray] = {}

    for target_name in df_targets.columns:
        print(f"\n### Entraînement pour {target_name}")
        y_train = y_train_all[target_name]
        y_val = y_val_all[target_name]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        booster, best_params = tune_and_train(target_name, dtrain, dval, n_trials=N_TRIALS)

        best_ntree = getattr(booster, "best_ntree_limit", None)
        preds = (
            booster.predict(dval, ntree_limit=best_ntree)
            if best_ntree
            else booster.predict(dval)
        )
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        metrics[target_name] = rmse
        print(f"Validation RMSE ({target_name}): {rmse:.4f}")
        print("Best params:", best_params)

        # Sauvegarde du modèle et de best_ntree
        model_path = models_dir / f"xgb_{target_name}.json"
        booster.save_model(model_path)
        if best_ntree:
            (models_dir / f"xgb_{target_name}_best_ntree.txt").write_text(
                str(best_ntree)
            )

        # Sauvegarde de 10 prédictions de validation pour inspection
        sample_inputs = X_val.head(10)
        sample_targets = y_val.head(10)
        sample_dval = xgb.DMatrix(sample_inputs)
        sample_preds = (
            booster.predict(sample_dval, ntree_limit=best_ntree)
            if best_ntree
            else booster.predict(sample_dval)
        )
        sample_preds_map[target_name] = sample_preds
        sample_true_map[target_name] = sample_targets.values

    print("\nRésumé RMSE validation par cible :")
    for name, score in metrics.items():
        print(f"- {name}: {score:.4f}")

    print("\nExemple de prédictions (10 premiers samples de la validation) :")
    sample_df = pd.DataFrame(index=X_val.head(10).index)
    for name, preds in sample_preds_map.items():
        sample_df[f"pred_{name}"] = preds[: len(sample_df)]
        sample_df[f"true_{name}"] = sample_true_map[name][: len(sample_df)]
    print(sample_df)

    # Sauvegarde des logs
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"log_{timestamp}.txt"
    with log_path.open("w", encoding="utf-8") as f:
        f.write("Résumé RMSE validation par cible :\n")
        for name, score in metrics.items():
            f.write(f"- {name}: {score:.4f}\n")
        f.write("\nExemple de prédictions (10 premiers samples de la validation) :\n")
        f.write(sample_df.to_string())
    print(f"Logs sauvegardés dans {log_path}")
