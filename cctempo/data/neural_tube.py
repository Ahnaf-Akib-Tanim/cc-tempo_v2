"""In vitro ventral neural tube scRNA-seq time course (Maizels, Snell & Briscoe, Cell Systems 2024).

GEO GSE236512 (part of SuperSeries GSE236520, the RNA data used by Cislo et al., PNAS 2025).
mESC -> NMP -> ventral neural progenitors, SAG 500 nM from Day 3, sampled Days 3-8.
We use the splicing object (spliced + unspliced = total counts), the smallest of the three files.
"""
import gzip
import os
import shutil
import urllib.request

import numpy as np

URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE236nnn/GSE236512/suppl/GSE236512_processed_data_splicing.h5ad.gz"

# marker programmes (mouse); used only when the object has no usable annotation
MARKERS = {
    "NMP": ["T", "Tbxt", "Nkx1-2", "Cdx2", "Fgf8", "Wnt3a"],
    "p0/p1": ["Dbx1", "Dbx2", "Pax6", "Irx3"],
    "p2": ["Foxn4", "Vsx1", "Irx3", "Nkx6-1"],
    "pMN": ["Olig2", "Olig1"],
    "p3": ["Nkx2-2", "Nkx2-9"],
    "FP": ["Foxa2", "Shh", "Arx", "Ntn1", "Spon1"],
    "Neuron": ["Tubb3", "Elavl3", "Elavl4", "Stmn2", "Mnx1", "Isl1"],
}
TIME_KEYS = ["timepoint", "time", "day", "Day", "time_point", "Time", "tp"]
TYPE_KEYS = ["cell_type", "celltype", "annotation", "cell_annotation", "leiden_annotation", "fate", "state"]


def download(raw_dir):
    os.makedirs(raw_dir, exist_ok=True)
    gz = os.path.join(raw_dir, os.path.basename(URL))
    h5 = gz[:-3]
    if not os.path.exists(h5):
        if not os.path.exists(gz):
            print("[download] neural tube splicing h5ad (~1.3 GB)")
            with urllib.request.urlopen(URL) as r, open(gz + ".part", "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 22)
            os.replace(gz + ".part", gz)
        with gzip.open(gz, "rb") as fi, open(h5, "wb") as fo:
            shutil.copyfileobj(fi, fo, length=1 << 22)
    return h5


def inspect(adata):
    lines = [f"shape {adata.shape}", f"layers {list(adata.layers.keys())}", f"obs columns {list(adata.obs.columns)}"]
    for c in adata.obs.columns:
        if adata.obs[c].nunique() < 40:
            lines.append(f"  {c}: {adata.obs[c].value_counts().to_dict()}")
    return "\n".join(lines)


def find_key(adata, candidates):
    for k in candidates:
        if k in adata.obs.columns:
            return k
    return None


def total_counts_layer(adata):
    """X := spliced + unspliced (or X if those layers are absent)."""
    if "spliced" in adata.layers and "unspliced" in adata.layers:
        adata.X = adata.layers["spliced"] + adata.layers["unspliced"]
    elif "total" in adata.layers:
        adata.X = adata.layers["total"]
    return adata


def annotate_by_markers(adata, min_score=0.1):
    """Rule-based annotation on log-normalised data: highest marker-programme score wins."""
    import scanpy as sc
    names = []
    for name, genes in MARKERS.items():
        g = [x for x in genes if x in adata.var_names]
        if len(g) == 0:
            continue
        sc.tl.score_genes(adata, g, score_name=f"score_{name}", use_raw=False, random_state=0)
        names.append(name)
    S = np.column_stack([adata.obs[f"score_{n}"].values for n in names])
    lab = np.array(names)[S.argmax(1)]
    lab[S.max(1) < min_score] = "Other"
    adata.obs["celltype_markers"] = lab
    return "celltype_markers"
