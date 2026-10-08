"""Cell-cell communication prior from LIANA (replaces CellChat + Seurat of v1).

Pipeline
1. run one or several LIANA methods per time point at the cell-type level (consensus
   `rank_aggregate` by default; individual methods for the robustness study of Reviewer 1(c));
2. turn each method's magnitude score into a [0, 1] rank score (1 = strongest) so methods are
   comparable;
3. select the top-K LR pairs (all subunits must be measured) + forced exogenous pairs
   (cytokines / SAG), and store their subunit genes, a prior weight and a ligand family;
4. (v1 only) build static per-cell sending/receiving features like the original paper:
   type-level communication x per-cell module score of ligands / receptors.
"""
import numpy as np
import pandas as pd

from ..data.genesets import CYTOKINE_PAIRS, ligand_family

METHODS = ["rank_aggregate", "cellphonedb", "cellchat", "natmi", "connectome", "singlecellsignalr",
           "geometric_mean", "logfc"]


def _method_score(res: pd.DataFrame, meth) -> pd.Series:
    col = getattr(meth, "magnitude", None)
    asc = getattr(meth, "magnitude_ascending", False)
    if col is None or col not in res.columns:  # e.g. logfc only defines a specificity score
        col = getattr(meth, "specificity", None)
        asc = getattr(meth, "specificity_ascending", False)
    v = res[col].astype(float)
    r = v.rank(pct=True, ascending=True)
    return 1.0 - r if asc else r


def _subcluster(sub, groupby, n_clusters=4, seed=0):
    """Split a single-type time point (e.g. Day-2 progenitors) into k-means subclusters on PCA."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    Z = TruncatedSVD(n_components=min(20, sub.n_vars - 1), random_state=seed).fit_transform(sub.X)
    lab = KMeans(n_clusters=n_clusters, n_init=4, random_state=seed).fit_predict(Z)
    return np.array([f"{t}~k{c}" for t, c in zip(sub.obs[groupby].astype(str), lab)])


def run_liana(adata, groupby, time_key=None, methods=("rank_aggregate",), resource_name="mouseconsensus",
              expr_prop=0.1, min_cells=10, n_perms=100, max_cells_per_group=1500, seed=0, layer=None,
              use_raw=False, verbose=True, n_subclusters=4):
    """Return a long table: method, time, source, target, source_type, target_type, ligand_complex,
    receptor_complex, score. Time points with a single cell type are split into subclusters
    ('<type>~k<i>'); source_type/target_type give the original type."""
    import liana as li

    rng = np.random.default_rng(seed)
    times = [None] if time_key is None else sorted(adata.obs[time_key].unique())
    out = []
    for t in times:
        sub = adata if t is None else adata[adata.obs[time_key] == t]
        keep = []
        for _, idx in sub.obs.groupby(groupby, observed=True).indices.items():
            keep.extend(idx if len(idx) <= max_cells_per_group else rng.choice(idx, max_cells_per_group, replace=False))
        sub = sub[np.sort(np.asarray(keep))].copy()
        grp = sub.obs[groupby].astype(str).values
        if len(set(grp)) < 2:
            grp = _subcluster(sub, groupby, n_subclusters, seed)
            if verbose:
                print(f"[liana] time={t}: single cell type -> {n_subclusters} subclusters")
        sub.obs["_cct_group"] = pd.Categorical(grp)
        kw_group = "_cct_group"
        for m in methods:
            meth = getattr(li.mt, m)
            kw = dict(groupby=kw_group, resource_name=resource_name, expr_prop=expr_prop, min_cells=min_cells,
                      use_raw=use_raw, layer=layer, verbose=False, key_added="_liana_tmp", n_perms=n_perms, seed=seed)
            try:
                meth(sub, **kw)
            except TypeError:
                kw.pop("n_perms"), kw.pop("seed")
                meth(sub, **kw)
            res = sub.uns["_liana_tmp"].copy()
            res["score"] = _method_score(res, meth).values
            res["method"] = m
            res["time"] = t
            res["source_type"] = res["source"].astype(str).str.split("~").str[0]
            res["target_type"] = res["target"].astype(str).str.split("~").str[0]
            out.append(res[["method", "time", "source", "target", "source_type", "target_type", "ligand_complex",
                            "receptor_complex", "score"]])
            if verbose:
                print(f"[liana] time={t} method={m}: {len(res)} interactions")
    return pd.concat(out, ignore_index=True)


def select_pairs(table, genes_available, method="rank_aggregate", top_k=64, force=None, exclude_self=False):
    """Select LR pairs and build the `lr` spec stored in data.pt.

    force: list of cytokine names (keys of CYTOKINE_PAIRS) to add as exogenous channels.
    """
    avail = set(genes_available)
    df = table[table["method"] == method]
    if exclude_self:
        df = df[df["source"] != df["target"]]
    agg = df.groupby(["ligand_complex", "receptor_complex"])["score"].max().reset_index()
    agg = agg.sort_values("score", ascending=False)
    lig, rec, lg, rg, prior, fam, exo = [], [], [], [], [], [], []
    for _, r in agg.iterrows():
        lgs, rgs = r.ligand_complex.split("_"), r.receptor_complex.split("_")
        if not (set(lgs) <= avail and set(rgs) <= avail):
            continue
        lig.append(r.ligand_complex); rec.append(r.receptor_complex); lg.append(lgs); rg.append(rgs)
        prior.append(float(r.score)); fam.append(ligand_family(r.ligand_complex)); exo.append(None)
        if len(lig) >= top_k:
            break
    for name in force or []:
        lgs, rgs = CYTOKINE_PAIRS[name]
        if not (set(lgs) <= avail and set(rgs) <= avail):
            print(f"[select_pairs] WARNING exogenous {name} genes missing from dataset: {set(lgs + rgs) - avail}")
            continue
        lig.append("_".join(lgs)); rec.append("_".join(rgs)); lg.append(list(lgs)); rg.append(list(rgs))
        prior.append(1.0); fam.append(name.upper()); exo.append(name)
    return {"ligand": lig, "receptor": rec, "lig_genes": lg, "rec_genes": rg, "prior": prior, "family": fam,
            "exogenous": exo, "method": method}


def method_overlap(table, top_k=64):
    """Jaccard overlap of the top-K LR pairs between methods (robustness table)."""
    tops = {}
    for m, df in table.groupby("method"):
        agg = df.groupby(["ligand_complex", "receptor_complex"])["score"].max().sort_values(ascending=False)
        tops[m] = set(agg.index[:top_k])
    ms = sorted(tops)
    mat = pd.DataFrame(index=ms, columns=ms, dtype=float)
    for a in ms:
        for b in ms:
            mat.loc[a, b] = len(tops[a] & tops[b]) / max(1, len(tops[a] | tops[b]))
    return mat


def v1_features(adata, lr, table, groupby, time_key, method="rank_aggregate"):
    """Static v1 features: per ligand family F, sending_i = mean_targets S(type_i -> *, F) * lig_score_i(F),
    receiving_i = mean_sources S(* -> type_i, F) * rec_score_i(F). Returns (n_cells, 2F) and names."""
    import scanpy as sc

    df = table[table["method"] == method].copy()
    fam_of = {(l, r): f for l, r, f in zip(lr["ligand"], lr["receptor"], lr["family"]) if f is not None}
    df["family"] = [fam_of.get((l, r)) for l, r in zip(df.ligand_complex, df.receptor_complex)]
    df = df.dropna(subset=["family"])
    fams = sorted(set(f for f, e in zip(lr["family"], lr["exogenous"]) if e is None))
    lig_genes = {f: sorted({g for gg, ff in zip(lr["lig_genes"], lr["family"]) if ff == f for g in gg}) for f in fams}
    rec_genes = {f: sorted({g for gg, ff in zip(lr["rec_genes"], lr["family"]) if ff == f for g in gg}) for f in fams}
    ad = adata.copy()
    feats = np.zeros((ad.n_obs, 2 * len(fams)), dtype=np.float32)
    types = ad.obs[groupby].astype(str).values
    times = ad.obs[time_key].values
    for j, f in enumerate(fams):
        for kind, genes, off in (("lig", lig_genes[f], 0), ("rec", rec_genes[f], len(fams))):
            name = f"_v1_{kind}_{j}"
            try:
                sc.tl.score_genes(ad, genes, score_name=name, use_raw=False, random_state=0)
                cell_score = ad.obs[name].values
            except Exception:
                cell_score = np.zeros(ad.n_obs)
            sub = df[df.family == f]
            key = "source_type" if kind == "lig" else "target_type"
            for t in np.unique(times):
                st = sub[sub.time == t] if time_key is not None else sub
                type_score = st.groupby(key)["score"].mean().to_dict()
                m = times == t
                feats[m, off + j] = np.array([type_score.get(tt, 0.0) for tt in types[m]]) * cell_score[m]
    names = [f"send_{f}" for f in fams] + [f"recv_{f}" for f in fams]
    return feats, names
