"""Gene sets used for growth estimation, perturbations and exogenous ligands (mouse symbols)."""

# Tirosh et al. 2016 S / G2M genes (the list scanpy uses), converted to mouse capitalisation.
# Renamed genes are listed under both names; only genes present in a dataset are used.
_S = """MCM5 PCNA TYMS FEN1 MCM2 MCM4 RRM1 UNG GINS2 MCM6 CDCA7 DTL PRIM1 UHRF1 MLF1IP CENPU HELLS
RFC2 RPA2 NASP RAD51AP1 GMNN WDR76 SLBP CCNE2 UBR7 POLD3 MSH2 ATAD2 RAD51 RRM2 CDC45 CDC6 EXO1
TIPIN DSCC1 BLM CASP8AP2 USP1 CLSPN POLA1 CHAF1B BRIP1 E2F8"""
_G2M = """HMGB2 CDK1 NUSAP1 UBE2C BIRC5 TPX2 TOP2A NDC80 CKS2 NUF2 CKS1B MKI67 TMPO CENPF TACC3 FAM64A
PIMREG SMC4 CCNB2 CKAP2L CKAP2 AURKB BUB1 KIF11 ANP32E TUBB4B GTSE1 KIF20B HJURP CDCA3 HN1 JPT1
CDC20 TTK CDC25C KIF2C RANGAP1 NCAPD2 DLGAP5 CDCA2 CDCA8 ECT2 KIF23 HMMR AURKA PSRC1 ANLN LBR
CKAP5 CENPE CTCF NEK2 G2E3 GAS2L3 CBX5 CENPA"""


def _mouse(s):
    return [g.capitalize() for g in s.split()]


CELL_CYCLE = sorted(set(_mouse(_S) + _mouse(_G2M)))

# Core apoptosis machinery (KEGG mmu04210 members); used only for the death score.
APOPTOSIS = sorted(set("""Bax Bak1 Bad Bid Bcl2 Bcl2l1 Bcl2l11 Mcl1 Casp3 Casp6 Casp7 Casp8 Casp9 Casp12
Apaf1 Cycs Diablo Endog Aifm1 Xiap Birc2 Birc3 Cflar Fas Fasl Fadd Tradd Ripk1 Tnf Tnfrsf1a Tnfrsf10b
Trp53 Pmaip1 Bbc3 Gadd45a Gadd45b Gadd45g Atm Ddit3 Atf4 Capn1 Capn2 Ctsb Ctsd Ctsl Lmnb1 Lmna Parp1
Parp2 Dffa Dffb Nfkbia Nfkb1 Rela Ikbkb Chuk Map3k5 Mapk8 Jun Fos Akt1 Pik3ca Pik3r1""".split()))

# Lineage transcription factors (true TFs, unlike some 'TF' lists in v1, which were markers).
# VERIFY against the literature you cite before the final paper.
TFS = {
    "Monocyte": ["Irf8", "Klf4", "Mafb", "Nr4a1", "Zeb2", "Spi1"],
    "Neutrophil": ["Cebpe", "Gfi1", "Cebpa", "Cebpd"],
}

# The marker sets the v1 paper perturbed (kept for back-comparison with the old figures).
V1_LINEAGE_GENES = {
    "Monocyte": "F13a1 Ms4a6c Ly6c2 S100a4 Rassf4 Csf1r Hpse Ly86 Emb Papss2 Ctss Slpi Irf8 Nr4a1 Klf4".split(),
    "Neutrophil": "Ltf Ngp Lcn2 Cd177 Camp S100a9 Ifitm6 Itgb2l Pglyrp1 S100a8 Lrg1 Fcnb Gp1bb Lyz2 Syne1".split(),
}

# Exogenous ligands (cytokines / morphogen agonists): name -> (ligand subunits, receptor subunits)
CYTOKINE_PAIRS = {
    "M-csf": (["Csf1"], ["Csf1r"]),
    "G-csf": (["Csf3"], ["Csf3r"]),
    "Epo": (["Epo"], ["Epor"]),
    "SCF": (["Kitl"], ["Kit"]),
    "GM-csf": (["Csf2"], ["Csf2ra", "Csf2rb"]),
    "IL-3": (["Il3"], ["Il3ra", "Csf2rb"]),
    "IL-6": (["Il6"], ["Il6ra", "Il6st"]),
    "TPO": (["Thpo"], ["Mpl"]),
    "Flt3L": (["Flt3l"], ["Flt3"]),
    "IL-11": (["Il11"], ["Il11ra1", "Il6st"]),
    "IL-5": (["Il5"], ["Il5ra", "Csf2rb"]),
    # SAG is a small-molecule Smoothened agonist: modelled as an exogenous Hedgehog input received
    # through Smo. Endogenous Shh (floor plate) enters through LIANA's Shh -> Ptch1_Smo pair.
    "SAG": (["Shh"], ["Smo"]),
}

# Heuristic ligand-family names used to aggregate LR pairs into 'pathways' for reporting.
FAMILY_PREFIX = [
    ("Ccl", "CCL"), ("Cxcl", "CXCL"), ("Csf1", "CSF"), ("Il34", "CSF"), ("Csf2", "GM-CSF"), ("Csf3", "G-CSF"),
    ("Il", "IL"), ("Tnfsf", "TNF"), ("Tnf", "TNF"), ("Tgfb", "TGFb"), ("Bmp", "BMP"), ("Gdf", "BMP"),
    ("Wnt", "WNT"), ("Fgf", "FGF"), ("Pdgf", "PDGF"), ("Vegf", "VEGF"), ("Lama", "LAMININ"),
    ("Lamb", "LAMININ"), ("Lamc", "LAMININ"), ("Col", "COLLAGEN"), ("Fn1", "FN1"), ("Thbs", "THBS"),
    ("Spp1", "SPP1"), ("Icam", "ICAM"), ("Sema", "SEMAPHORIN"), ("Efn", "EPHRIN"), ("Jag", "NOTCH"),
    ("Dll", "NOTCH"), ("Shh", "HH"), ("Ihh", "HH"), ("Dhh", "HH"), ("Slit", "SLIT"), ("Ntn", "NETRIN"),
    ("App", "APP"), ("Mif", "MIF"), ("Grn", "GRN"), ("Sell", "SELL"), ("Selplg", "SELPLG"), ("Cd34", "CD34"),
    ("Kitl", "KIT"), ("Epo", "EPO"), ("Thpo", "TPO"), ("Flt3l", "FLT3"), ("Lgals", "GALECTIN"),
    ("Pros1", "PROS"), ("Gas6", "GAS"), ("Angpt", "ANGPT"), ("Cd", "CD"), ("H2-", "MHC"),
]


def ligand_family(ligand_complex: str) -> str:
    first = ligand_complex.split("_")[0]
    for pre, fam in FAMILY_PREFIX:
        if first.startswith(pre):
            return fam
    return first.upper()
