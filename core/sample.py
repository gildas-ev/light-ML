from pathlib import Path

from .roc import Roc

_SHORT_QUERY_PATH = Path("en") / "user_want_short" / "user_prompt.txt"
_XMP_FILENAME = "config.xmp"
_BEFORE_FILENAME = "before.jpg"
_AFTER_FILENAME = "processed.jpg"

class Sample:
    """Container describing a single MMArt-PPR10k example."""

    def __init__(self, id: str, query_short: str, roc: Roc, folder: Path):
        self.id = id
        self.query_short = query_short
        self.roc = roc
        self._folder = folder
        self.cluster_id = None

    @classmethod
    def from_folder(cls, folder: str | Path) -> "Sample":
        """Build a Sample from the dataset folder structure."""
        folder_path = Path(folder).resolve()
        
        short_query_file = folder_path / _SHORT_QUERY_PATH
        query_short = short_query_file.read_text(encoding="utf-8").strip()
        xmp_path = folder_path / _XMP_FILENAME

        from tools import read_xmp_file  # local import to avoid circular dependency

        roc = Roc.from_xmp(read_xmp_file(xmp_path))

        return cls(
            id=folder_path.name,
            query_short=query_short,
            roc=roc,
            folder=folder_path,
        )

    def _image_path(self, filename: str) -> Path:
        path = self._folder / filename
        return path

    def show_before_after(self) -> None:
        """Display the before/after images side by side using matplotlib."""
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt

        before = mpimg.imread(self._image_path(_BEFORE_FILENAME))
        after = mpimg.imread(self._image_path(_AFTER_FILENAME))

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(before)
        axes[0].set_title("Before")
        axes[0].axis("off")

        axes[1].imshow(after)
        axes[1].set_title("After")
        axes[1].axis("off")

        fig.suptitle(f"Sample {self.id} – {self.query_short}")
        plt.tight_layout()
        plt.show()

    def processed_image_path(self) -> Path:
        """Return the path to the processed (after) image."""
        return self._image_path(_AFTER_FILENAME)

    def __str__(self) -> str:
        query_preview = self.query_short
        if len(query_preview) > 60:
            query_preview = f"{query_preview[:57]}..."
        return f"Sample(id='{self.id}', query_short='{query_preview}')"
