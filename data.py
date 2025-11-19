from collections import defaultdict
from math import ceil
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from core.cluster import Cluster, ROC_COMPONENTS
from core.sample import Sample
from tools import describe_cluster, compute_features

DATASET_ROOT = "data/MMArt-PPR10k"
CLUSTER_HIST_DIR = "cluster_histograms"
CLUSTER_SUMMARY_FILE = "cluster_summary.xlsx"
PROCESSED_DATA_DIR = "data/processed"

ROC_HIST_METRICS: dict[str, str] = {
    "Highlights": "highlights2012",
    "Shadows": "shadows2012",
    "Whites": "whites2012",
    "Blacks": "blacks2012",
    "Exposure": "exposure2012",
    "Contrast": "contrast2012",
    "Vibrance": "vibrance",
    "Dehaze": "dehaze",
}

NUM_CLUSTERS = 4

def export_clusters_to_excel(
    clusters: list[Cluster],
    output_path: Path | str = CLUSTER_SUMMARY_FILE,
    top_words: int = 10,
) -> None:
    """Export cluster summaries (counts, descriptions, stats) to an Excel file."""

    rows: list[str] = ["nb_samples", "description"]
    for _, label in ROC_COMPONENTS:
        rows.append(f"{label}_mean")
    for _, label in ROC_COMPONENTS:
        rows.append(f"{label}_std")

    data: dict[str, dict[str, float | str]] = {}
    for cluster in clusters:
        column: dict[str, float | str] = {}
        column["nb_samples"] = len(cluster.samples)
        column["description"] = describe_cluster(
            [sample.query_short for sample in cluster.samples],
            top_words,
        )
        for attr, label in ROC_COMPONENTS:
            column[f"{label}_mean"] = getattr(cluster.avg_roc, attr)
            column[f"{label}_std"] = getattr(cluster.std_roc, attr)
        data[f"Cluster {cluster.id}"] = column

    df = pd.DataFrame.from_dict(data, orient="columns")
    df = df.reindex(rows)
    df.index.name = "Metric"

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output)
    print(f"Fichier Excel des clusters enregistré dans {output}")

def plot_roc_histograms(
    samples: list[Sample],
    save_path: Path | str | None = None,
    show: bool = True,
) -> None:
    """Plot or save histograms for selected ROC attributes."""
    if not samples:
        raise ValueError("Impossible de tracer des histogrammes sur un dataset vide.")

    metric_map = ROC_HIST_METRICS

    num_metrics = len(metric_map)
    cols = min(4, num_metrics)
    rows = ceil(num_metrics / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    if hasattr(axes, "flat"):
        axes_list = list(axes.flat)
    else:
        axes_list = [axes]

    for ax, (title, attr_name) in zip(axes_list, metric_map.items()):
        values = [getattr(sample.roc, attr_name) for sample in samples]
        ax.hist(values, bins=30, color="#4C72B0", edgecolor="black")
        ax.set_title(title)
        ax.set_xlabel("Valeur")
        ax.set_ylabel("Fréquence")

    for ax in axes_list[len(metric_map) :]:
        ax.axis("off")

    fig.suptitle("Histogrammes des réglages ROC")
    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        print(f"Histogrammes sauvegardés dans {path}")

    if show:
        plt.show()

    plt.close(fig)

def load_dataset() -> list[Sample]:
    """Read each sample folder and return Sample instances."""
    root = Path(DATASET_ROOT).resolve()
    split_dir = root / "global"

    samples: list[Sample] = []
    for folder in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        samples.append(Sample.from_folder(folder))
    return samples

def cluster_queries(
    samples: list[Sample],
    num_clusters: int = 10,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[Cluster]:
    """Cluster user short queries using sentence embeddings + KMeans."""

    queries: list[str] = []
    indexed_samples: list[Sample] = []
    for sample in samples:
        query = sample.query_short
        queries.append(query)
        indexed_samples.append(sample)

    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(queries, show_progress_bar=True)

    clustering_model = KMeans(
        n_clusters=num_clusters,
        n_init=10,
    )
    labels = clustering_model.fit_predict(embeddings)

    clusters = defaultdict(list)
    for label, sample in zip(labels, indexed_samples):
        clusters[label].append(sample)
        sample.cluster_id = label

    return [
        Cluster(cluster_id, cluster_samples)
        for cluster_id, cluster_samples in sorted(clusters.items())
    ]
    
def dataset_to_df(dataset: list[Sample], clusters: list[Cluster]) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows = []
    target_rows = []

    for sample in dataset:
        features = {
            "id": sample.id,
            "cluster_id": sample.cluster_id
        }
        features.update(compute_features(sample.processed_image_path()))

        target = {"id": sample.id}
        offset = sample.roc - clusters[sample.cluster_id].avg_roc
        target.update(offset.to_dict())
        
        feature_rows.append(features)
        target_rows.append(target)

    features_df = pd.DataFrame(feature_rows).set_index("id")
    targets_df = pd.DataFrame(target_rows).set_index("id")

    return features_df, targets_df

if __name__ == "__main__":
    dataset = load_dataset()
    print(f"Dataset chargée avec {len(dataset)} samples.")

    clusters = cluster_queries(dataset, num_clusters=NUM_CLUSTERS)
    print(f"{len(clusters)} clusters générés avec SentenceTransformer.")
    
    export_clusters_to_excel(clusters)

    if Path(CLUSTER_HIST_DIR).exists():
        shutil.rmtree(Path(CLUSTER_HIST_DIR))
    Path(CLUSTER_HIST_DIR).mkdir(parents=True, exist_ok=True)

    for cluster in clusters:
        print(f"\nCluster {cluster.id} ({len(cluster.samples)} samples)")
        print(describe_cluster([sample.query_short for sample in cluster.samples], 10))

        hist_path = Path(CLUSTER_HIST_DIR) / f"cluster_{cluster.id:03d}.png"
        plot_roc_histograms(
            cluster.samples,
            save_path=hist_path,
            show=False,
        )
        cluster.save_processed_examples(
            output_path=Path(CLUSTER_HIST_DIR) / f"cluster_{cluster.id:03d}_examples.png",
            max_examples=4,
        )

    df_features, df_targets = dataset_to_df(dataset, clusters)

    processed_dir = Path(PROCESSED_DATA_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)
    features_path = processed_dir / "features.csv"
    targets_path = processed_dir / "targets.csv"
    df_features.to_csv(features_path, index=True)
    df_targets.to_csv(targets_path, index=True)
