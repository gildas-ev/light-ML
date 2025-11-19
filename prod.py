"""Interface Streamlit pour générer un preset XMP à partir d'une image et d'un cluster."""

from pathlib import Path
import tempfile
import uuid

import pandas as pd
import streamlit as st
import xgboost as xgb

from core.cluster import ROC_COMPONENTS
from core.roc import Roc
from tools import compute_features, save_xmp


def build_feature_row(
    image_path: Path,
    cluster_id: int,
    template_path: Path | str = "data/processed/features.csv",
) -> pd.DataFrame:
    """Calcule les features + one-hot du cluster, alignées sur les colonnes d'entraînement."""
    features = compute_features(image_path)
    row = pd.DataFrame([features])
    row["cluster_id"] = cluster_id

    row_enc = pd.get_dummies(row, columns=["cluster_id"], prefix="cluster", dtype=float)

    template_df = pd.read_csv(template_path, index_col=0)
    template_df.columns = template_df.columns.str.strip()
    template_enc = pd.get_dummies(
        template_df, columns=["cluster_id"], prefix="cluster", dtype=float
    )
    template_cols = template_enc.columns.tolist()

    for col in template_cols:
        if col not in row_enc.columns:
            row_enc[col] = 0.0
    row_enc = row_enc[template_cols]
    row_enc.index = [image_path.stem]
    return row_enc


def load_cluster_info(
    summary_path: Path | str = "cluster_summary.xlsx",
) -> tuple[dict[int, Roc], dict[int, str]]:
    """Retourne mapping cluster_id -> Roc moyen et description depuis l'Excel."""
    df = pd.read_excel(summary_path, index_col=0)
    df.index = df.index.str.strip()

    cluster_rocs: dict[int, Roc] = {}
    descriptions: dict[int, str] = {}
    for col in df.columns:
        col_name = str(col).strip()
        if not col_name.lower().startswith("cluster"):
            continue
        try:
            cluster_id = int(col_name.split()[-1])
        except ValueError:
            continue

        kwargs = {}
        for attr, label in ROC_COMPONENTS:
            row_label = f"{label}_mean"
            if row_label in df.index:
                kwargs[label] = df.loc[row_label, col]
        cluster_rocs[cluster_id] = Roc(**kwargs)

        if "description" in df.index:
            descriptions[cluster_id] = str(df.loc["description", col])
    return cluster_rocs, descriptions


def load_boosters(
    models_dir: Path | str = "models",
) -> dict[str, tuple[xgb.Booster, int | None]]:
    """Charge tous les boosters XGBoost présents dans le dossier."""
    models_dir = Path(models_dir)
    boosters: dict[str, tuple[xgb.Booster, int | None]] = {}
    for model_file in models_dir.glob("xgb_*.json"):
        target_name = model_file.stem.replace("xgb_", "")
        booster = xgb.Booster()
        booster.load_model(model_file)

        best_ntree_file = models_dir / f"{model_file.stem}_best_ntree.txt"
        best_ntree: int | None = None
        if best_ntree_file.exists():
            try:
                best_ntree = int(best_ntree_file.read_text().strip())
            except ValueError:
                best_ntree = None
        boosters[target_name] = (booster, best_ntree)
    return boosters


def predict_all_targets(
    features: pd.DataFrame, boosters: dict[str, tuple[xgb.Booster, int | None]]
) -> Roc:
    """Retourne un Roc où chaque attribut est la prédiction pour la cible correspondante."""
    dmat = xgb.DMatrix(features)
    preds: dict[str, float] = {}
    for target_name, (booster, best_ntree) in boosters.items():
        pred = (
            booster.predict(dmat, ntree_limit=best_ntree)
            if best_ntree
            else booster.predict(dmat)
        )
        preds[target_name] = float(pred[0])

    return Roc(
        Exposure2012=preds.get("Exposure2012", 0.0),
        Contrast2012=preds.get("Contrast2012", 0.0),
        Highlights2012=preds.get("Highlights2012", 0.0),
        Shadows2012=preds.get("Shadows2012", 0.0),
        Whites2012=preds.get("Whites2012", 0.0),
        Blacks2012=preds.get("Blacks2012", 0.0),
        Vibrance=preds.get("Vibrance", 0.0),
        Dehaze=preds.get("Dehaze", 0.0),
    )


def main() -> None:
    st.title("Générateur de preset XMP")

    # Upload d'image
    uploaded = st.file_uploader("Dépose une image", type=["jpg", "jpeg", "png"])

    # Chargement cluster infos
    cluster_means, cluster_desc = load_cluster_info()
    cluster_options = sorted(cluster_means.keys())
    selected_cluster = st.selectbox(
        "Cluster/Style à appliquer",
        cluster_options,
        format_func=lambda cid: f"Cluster {cid} – {cluster_desc.get(cid, '').strip()}",
    )

    # Répertoire de sortie
    default_out_dir = Path("presets")
    out_dir_str = st.text_input(
        "Dossier de sortie pour le preset XMP",
        value=str(default_out_dir),
    )
    out_dir = Path(out_dir_str)

    if st.button("Générer le preset"):
        if not uploaded:
            st.error("Merci de déposer une image.")
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.getvalue())
            temp_path = Path(tmp.name)

        try:
            features_df = build_feature_row(temp_path, selected_cluster)
            boosters = load_boosters()
            offset_roc = predict_all_targets(features_df, boosters)
            predicted_roc = cluster_means[selected_cluster] + offset_roc

            uid = uuid.uuid4().hex[:8].upper()
            preset_name = f"{temp_path.stem}_preset_{uid}"
            xmp = predicted_roc.to_xmp(name=preset_name)

            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{preset_name}.xmp"
            save_xmp(xmp, out_path)

            st.success(f"Preset généré et sauvegardé dans {out_path}")
            st.code(str(predicted_roc), language="text")
        finally:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
