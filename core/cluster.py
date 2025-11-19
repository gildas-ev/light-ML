from math import sqrt
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from .roc import Roc
from .sample import Sample

ROC_COMPONENTS: list[tuple[str, str]] = [
    ("exposure2012", "Exposure2012"),
    ("contrast2012", "Contrast2012"),
    ("highlights2012", "Highlights2012"),
    ("shadows2012", "Shadows2012"),
    ("whites2012", "Whites2012"),
    ("blacks2012", "Blacks2012"),
    ("vibrance", "Vibrance"),
    ("dehaze", "Dehaze"),
]

class Cluster:
    """Group of samples sharing similar user queries."""

    def __init__(self, id: int, samples: list[Sample]) -> None:
        self.id = id
        self.samples = samples
        self._avg_roc = None
        self._std_roc = None

    @property
    def avg_roc(self) -> Roc:
        if self._avg_roc is None or self._std_roc is None:
            self._compute_roc_stats()
        return self._avg_roc

    @property
    def std_roc(self) -> Roc:
        if self._avg_roc is None or self._std_roc is None:
            self._compute_roc_stats()
        return self._std_roc

    def _compute_roc_stats(self) -> tuple[Roc, Roc]:
        if not self.samples:
            self._avg_roc, self._std_roc = Roc(), Roc()
            return self._avg_roc, self._std_roc

        count = len(self.samples)
        sums = {attr: 0.0 for attr, _ in ROC_COMPONENTS}
        sums_sq = {attr: 0.0 for attr, _ in ROC_COMPONENTS}

        for sample in self.samples:
            roc = sample.roc
            for attr, _ in ROC_COMPONENTS:
                value = getattr(roc, attr)
                sums[attr] += value
                sums_sq[attr] += value * value

        avg_kwargs: dict[str, float] = {}
        std_kwargs: dict[str, float] = {}
        for attr, ctor_name in ROC_COMPONENTS:
            mean = sums[attr] / count
            variance = max(sums_sq[attr] / count - mean * mean, 0.0)
            avg_kwargs[ctor_name] = mean
            std_kwargs[ctor_name] = sqrt(variance)

        self._avg_roc = Roc(**avg_kwargs)
        self._std_roc = Roc(**std_kwargs)
        return self._avg_roc, self._std_roc

    def save_processed_examples(
        self,
        output_path: Path | str,
        max_examples: int = 4,
    ) -> None:
        """Save a grid with up to 4 processed images from the cluster."""
        if not self.samples:
            raise ValueError("Cluster vide, impossible de générer un aperçu.")

        num_examples = min(max_examples, len(self.samples))
        selected_samples = self.samples[:num_examples]
        images = [mpimg.imread(sample.processed_image_path()) for sample in selected_samples]

        fig, axes = plt.subplots(1, num_examples, figsize=(num_examples * 4, 4))
        if num_examples == 1:
            axes = [axes]

        for ax, sample_obj, image in zip(axes, selected_samples, images):
            ax.imshow(image)
            ax.set_title(sample_obj.id)
            ax.axis("off")

        fig.suptitle(f"Cluster {self.id} – exemples traités")
        plt.tight_layout()

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
