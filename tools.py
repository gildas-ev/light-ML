from pathlib import Path
from collections import defaultdict

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import nltk
nltk.download("stopwords")

from core.roc import Roc

CUSTOM_STOPWORDS_FILE = Path(__file__).with_name("custom_stopwords.txt")
stop_words = {word.casefold() for word in nltk.corpus.stopwords.words("english")}
stemmer = nltk.stem.PorterStemmer()
tokennizer = nltk.tokenize.RegexpTokenizer(r"\w+")

def load_custom_stopwords(path: Path = CUSTOM_STOPWORDS_FILE) -> set[str]:
    """Read custom stopwords (first token per line) if the file exists."""

    custom_words: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            first_token = stripped.split()[0]
            custom_words.add(first_token.casefold())
    return custom_words

stop_words.update(load_custom_stopwords())

def read_xmp_file(path: str) -> str:
    """Return the entire contents of an XMP file as a string."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as file:
        return file.read()

def save_xmp(xmp: str, path: str) -> None:
    with open(path, 'w') as f:
        f.write(xmp)

def text_processing(query: str) -> list[str]:
    words = tokennizer.tokenize(query)
    filtered = [word for word in words if word.casefold() not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in filtered]
    
    return stemmed_words

def describe_cluster(queries: list[str], nb: int) -> str:
    queries_processed = [text_processing(query) for query in queries]
    count = defaultdict(int)

    for query in queries_processed:
        for word in query:
            count[word] += 1
    
    sorted_count = {k: v for k, v in sorted(count.items(), key=lambda item: item[1], reverse=True)}
    return " ".join(list(sorted_count.keys())[:nb])

def compute_features(
    path: Path,
    flag: bool = False,
) -> dict[str, float]:
    img = cv.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Impossible de charger l'image: {path}")
    b,g,r = cv.split(img)
    luma = 0.2126*r+0.7152*g+0.0722*b
    bins = 16

    luma_mean = np.mean(luma)
    luma_median = np.median(luma)
    luma_std = np.std(luma)
    p1, p5, p95, p99 = np.percentile(luma, [1, 5, 95, 99])
    global_contrast = p95 - p5
    clipping_high = (luma >= 0.99*255).mean()*100
    clipping_low = (luma <= 0.01*255).mean()*100
    
    hist_r = cv.calcHist([img],[0],None,[bins],[0,256])
    hist_g = cv.calcHist([img],[1],None,[bins],[0,256])
    hist_b = cv.calcHist([img],[2],None,[bins],[0,256])

    # Normalise chaque histogramme en densité (somme = 1)
    for hist in (hist_r, hist_g, hist_b):
        total = hist.sum()
        if total > 0:
            hist /= total

    if flag:
        print(f"""Luma mean: {luma_mean} Median: {luma_median} Std: {luma_std}
P1: {p1} P5: {p5} P95: {p95} P99: {p99}
Global Contrast: {global_contrast}
Clipping High: {clipping_high} Clipping Low: {clipping_low}""")
    
        # Affiche l'image originale et les histogrammes
        plt.figure()
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Image originale")
        plt.show()

        plt.figure()
        plt.plot(hist_r, color='r')
        plt.plot(hist_g, color='g')
        plt.plot(hist_b, color='b')
        plt.xlim([0,bins])
        plt.title("Histogrammes RGB")
        plt.show()

    features: dict[str, float] = {
        "luma_mean": float(luma_mean),
        "luma_median": float(luma_median),
        "luma_std": float(luma_std),
        "p1": float(p1),
        "p5": float(p5),
        "p95": float(p95),
        "p99": float(p99),
        "global_contrast": float(global_contrast),
        "clipping_high": float(clipping_high),
        "clipping_low": float(clipping_low),
    }

    for idx, val in enumerate(hist_r.flatten(), start=1):
        features[f"r_bin{idx}"] = float(val)
    for idx, val in enumerate(hist_g.flatten(), start=1):
        features[f"g_bin{idx}"] = float(val)
    for idx, val in enumerate(hist_b.flatten(), start=1):
        features[f"b_bin{idx}"] = float(val)

    return features

if __name__=="__main__":
    img_path_rgb = "data/MMArt-PPR10k/global/1085_7/before.jpg"
    img_path_hdr = "data/MMArt-PPR10k/global/811_2/before.jpg"
    print(str(Path(img_path_rgb)))
    print(compute_features(Path(img_path_rgb), flag=True))
