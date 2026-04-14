"""
phylo_umap_compare.py
─────────────────────
Visual comparison of IQ-TREE topology vs UMAP/dimensionality-reduction embedding.

Usage
-----
    python phylo_umap_compare.py \
        --tree    results.treefile \
        --umap    umap_coords.csv \
        --n-clades 8 \
        --out     comparison.pdf

UMAP CSV format (header optional):
    name, umap1, umap2
    seq1, -3.21,  1.45
    seq2,  2.87, -0.93

Dependencies
------------
    pip install biopython matplotlib scipy scikit-learn pandas numpy toytree
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# ─── Optional imports (graceful fallback) ─────────────────────────────────────
try:
    import toytree
    HAS_TOYTREE = True
except ImportError:
    HAS_TOYTREE = False

try:
    from Bio import Phylo
    from io import StringIO
    HAS_BIO = True
except ImportError:
    HAS_BIO = False

# ─── Color palette ────────────────────────────────────────────────────────────
PALETTE = [
    "#00d4ff", "#ff6b6b", "#ffd93d", "#6bcb77", "#ff922b",
    "#cc5de8", "#74c0fc", "#f06595", "#a9e34b", "#ff8787",
    "#4dabf7", "#63e6be", "#ffa94d", "#da77f2", "#66d9e8",
    "#f783ac", "#8ce99a", "#ffe066", "#a5d8ff", "#ffb2b2",
]

def get_color(clade_id):
    return PALETTE[int(clade_id) % len(PALETTE)]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & PARSE
# ═══════════════════════════════════════════════════════════════════════════════

def load_umap(path: str) -> pd.DataFrame:
    """Load UMAP coordinates. Accepts parquet (primary), CSV, or TSV."""
    p = Path(path)

    # ── Parquet (your primary format: embedding_param_sweep .parquet files) ────
    if p.suffix == ".parquet":
        df = pd.read_parquet(path)
        df.columns = [str(c).lower().strip() for c in df.columns]
        # rename common id column variants to 'name'
        for candidate in ["sequence_id", "seq_id", "id", "header", "accession", "sample_id"]:
            if candidate in df.columns and "name" not in df.columns:
                df = df.rename(columns={candidate: "name"})
                break
        # if index holds the names (common when saved with index=True)
        if "name" not in df.columns:
            df = df.reset_index().rename(columns={"index": "name"})
        # rename first two non-name columns to umap1/umap2
        coord_cols = [c for c in df.columns if c != "name"]
        if len(coord_cols) >= 2:
            df = df.rename(columns={coord_cols[0]: "umap1", coord_cols[1]: "umap2"})
        df["name"] = df["name"].astype(str).str.strip()
        df["umap1"] = pd.to_numeric(df["umap1"], errors="coerce")
        df["umap2"] = pd.to_numeric(df["umap2"], errors="coerce")
        df = df.dropna(subset=["umap1", "umap2"])
        print(f"  Loaded {len(df)} UMAP points from {path}")
        return df[["name", "umap1", "umap2"]].reset_index(drop=True)

    # ── npy ────────────────────────────────────────────────────────────────────
    if p.suffix == ".npy":
        raise SystemExit(
            "ERROR: .npy file detected.\n"
            "Convert it first:\n"
            "  import numpy as np, pandas as pd\n"
            "  coords = np.load('umap_coords.npy')\n"
            "  names  = open('names.txt').read().split()\n"
            "  pd.DataFrame({'name': names, 'umap1': coords[:,0], 'umap2': coords[:,1]})"
            ".to_csv('umap.csv', index=False)"
        )

    # ── CSV / TSV ──────────────────────────────────────────────────────────────
    raw = p.read_text()
    sep = "\t" if "\t" in raw.split("\n")[0] else ","
    df = pd.read_csv(path, sep=sep, header=None)
    try:
        float(df.iloc[0, 1])
    except (ValueError, TypeError):
        df = pd.read_csv(path, sep=sep)
        df.columns = ["name"] + [f"dim{i}" for i in range(1, len(df.columns))]
    else:
        df.columns = ["name"] + [f"dim{i}" for i in range(1, len(df.columns))]

    df = df.rename(columns={"dim1": "umap1", "dim2": "umap2"})
    df["name"] = df["name"].astype(str).str.strip()
    df["umap1"] = pd.to_numeric(df["umap1"], errors="coerce")
    df["umap2"] = pd.to_numeric(df["umap2"], errors="coerce")
    df = df.dropna(subset=["umap1", "umap2"])
    print(f"  Loaded {len(df)} UMAP points from {path}")
    return df.reset_index(drop=True)


def load_tree_toytree(path: str):
    import toytree
    tree = toytree.tree(path)
    print(f"  Loaded tree with {tree.ntips} tips via toytree")
    return tree


def load_tree_biopython(path: str):
    tree = Phylo.read(path, "newick")
    tips = tree.get_terminals()
    print(f"  Loaded tree with {len(tips)} tips via biopython")
    return tree


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CLADE ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

def assign_clades_toytree(tree, n_clades: int) -> dict:
    """Cut toytree into n_clades groups; returns {tip_name: clade_id}."""
    import toytree
    # Use branch lengths to define clusters via scipy linkage on patristic distances
    tips = tree.get_tip_labels()
    n = len(tips)
    dist = np.zeros((n, n))
    for i, t1 in enumerate(tips):
        for j, t2 in enumerate(tips):
            if i < j:
                d = tree.distance(t1, t2)
                dist[i, j] = dist[j, i] = d

    from scipy.cluster.hierarchy import linkage, fcluster
    Z = linkage(squareform(dist), method="average")
    labels = fcluster(Z, n_clades, criterion="maxclust")
    return {name: int(labels[i]) - 1 for i, name in enumerate(tips)}


def assign_clades_biopython(tree, n_clades: int) -> dict:
    """Cut biopython tree into n_clades groups via hierarchical clustering."""
    tips = tree.get_terminals()
    names = [t.name for t in tips]
    n = len(names)
    dist = np.zeros((n, n))
    for i, t1 in enumerate(tips):
        for j, t2 in enumerate(tips):
            if i < j:
                d = tree.distance(t1, t2)
                dist[i, j] = dist[j, i] = d

    from scipy.cluster.hierarchy import linkage, fcluster
    Z = linkage(squareform(dist), method="average")
    labels = fcluster(Z, n_clades, criterion="maxclust")
    return {name: int(labels[i]) - 1 for i, name in enumerate(names)}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_stats(umap_df: pd.DataFrame, clade_map: dict,
                  tree, use_toytree: bool) -> dict:
    """Compute Mantel r, ARI, NMI between tree distances and UMAP distances."""
    shared = [n for n in umap_df["name"] if n in clade_map]
    if len(shared) < 4:
        return {}

    sub = umap_df[umap_df["name"].isin(shared)].set_index("name").loc[shared]
    umap_coords = sub[["umap1", "umap2"]].values
    umap_dist = squareform(pdist(umap_coords))

    # patristic distances
    n = len(shared)
    tree_dist = np.zeros((n, n))
    if use_toytree:
        for i, t1 in enumerate(shared):
            for j, t2 in enumerate(shared):
                if i < j:
                    d = tree.distance(t1, t2)
                    tree_dist[i, j] = tree_dist[j, i] = d
    else:
        tip_objs = {t.name: t for t in tree.get_terminals()}
        for i, t1 in enumerate(shared):
            for j, t2 in enumerate(shared):
                if i < j:
                    d = tree.distance(tip_objs[t1], tip_objs[t2])
                    tree_dist[i, j] = tree_dist[j, i] = d

    # Mantel (vectorise upper triangle)
    idx = np.triu_indices(n, k=1)
    r, p = spearmanr(tree_dist[idx], umap_dist[idx])

    tree_labels  = [clade_map[n] for n in shared]
    # UMAP cluster labels via k-means
    from sklearn.cluster import KMeans
    n_clades = len(set(tree_labels))
    km = KMeans(n_clusters=n_clades, random_state=42, n_init=10)
    umap_labels = km.fit_predict(umap_coords)

    ari = adjusted_rand_score(tree_labels, umap_labels)
    nmi = normalized_mutual_info_score(tree_labels, umap_labels)

    return {"mantel_r": r, "mantel_p": p, "ARI": ari, "NMI": nmi,
            "n_shared": len(shared)}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

BG   = "#0a0e1a"
GRID = "#1a2a3a"
TEXT = "#c8d8f0"
DIM  = "#3a5070"


def style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=DIM, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)


def plot_umap(ax, umap_df: pd.DataFrame, clade_map: dict, title: str = ""):
    style_ax(ax)
    for _, row in umap_df.iterrows():
        clade = clade_map.get(row["name"], -1)
        color = get_color(clade) if clade >= 0 else "#333"
        ax.scatter(row["umap1"], row["umap2"], c=color, s=18,
                   alpha=0.85, linewidths=0, zorder=3)

    # draw ellipse per clade
    from matplotlib.patches import Ellipse
    clades = {}
    for _, row in umap_df.iterrows():
        c = clade_map.get(row["name"], -1)
        if c < 0: continue
        clades.setdefault(c, []).append([row["umap1"], row["umap2"]])

    for c, pts in clades.items():
        if len(pts) < 3: continue
        pts = np.array(pts)
        cx, cy = pts.mean(axis=0)
        rx = pts[:, 0].std() * 2 + 0.1
        ry = pts[:, 1].std() * 2 + 0.1
        ell = Ellipse((cx, cy), rx * 2, ry * 2,
                      edgecolor=get_color(c), facecolor="none",
                      linewidth=1, linestyle="--", alpha=0.5, zorder=2)
        ax.add_patch(ell)

    ax.set_xlabel("UMAP-1", color=DIM, fontsize=9)
    ax.set_ylabel("UMAP-2", color=DIM, fontsize=9)
    ax.set_title(title or "UMAP — colored by IQ-TREE clade", color=TEXT,
                 fontsize=10, pad=8)


def plot_tree_toytree(ax, tree, clade_map: dict):
    """Draw tree using toytree's coords, rendered onto matplotlib axes."""
    style_ax(ax)
    import toytree

    # Use toytree to get coordinates
    coords = tree.get_node_coordinates()
    tips   = tree.get_tip_labels()
    n_tips = len(tips)

    # map tip y-positions
    tip_y = {name: i for i, name in enumerate(tips)}

    def draw_node(node_idx):
        children = tree.get_children(node_idx)
        nx, ny = coords[node_idx]
        for child_idx in children:
            cx, cy = coords[child_idx]
            child_name = tree.get_node_labels()[child_idx]
            clade = clade_map.get(child_name, -1)
            # try to get tip under child
            child_tips = tree.get_tip_labels(idx=child_idx) \
                if hasattr(tree, "get_tip_labels") else []
            tip_clade = clade_map.get(child_tips[0], -1) if child_tips else -1
            color = get_color(tip_clade) if tip_clade >= 0 else DIM
            ax.plot([nx, nx, cx], [ny, cy, cy], color=color,
                    lw=0.9, alpha=0.8, solid_capstyle="round")
            draw_node(child_idx)

    # fallback: simple dendrogram via biopython if coords fail
    ax.set_title("IQ-TREE Topology — colored by clade", color=TEXT,
                 fontsize=10, pad=8)
    ax.set_xlabel("Branch length", color=DIM, fontsize=9)
    ax.set_yticks([])


def plot_tree_biopython(ax, tree, clade_map: dict):
    """Draw a horizontal cladogram using biopython + matplotlib."""
    style_ax(ax)
    tips = tree.get_terminals()
    n_tips = len(tips)

    # assign leaf y positions
    leaf_y = {}
    for i, t in enumerate(tips):
        leaf_y[t.name] = i

    # compute x (cumulative branch length from root)
    def get_x(clade, parent_x=0.0):
        x = parent_x + (clade.branch_length or 0.0)
        clade._x = x
        for child in clade.clades:
            get_x(child, x)

    get_x(tree.root)

    # assign y as mean of children
    def get_y(clade):
        if not clade.clades:
            clade._y = leaf_y[clade.name]
        else:
            for c in clade.clades: get_y(c)
            clade._y = np.mean([c._y for c in clade.clades])

    get_y(tree.root)

    # draw
    def draw(clade):
        for child in clade.clades:
            clade_id = clade_map.get(child.name, -1)
            if clade_id < 0 and not child.clades:
                clade_id = clade_map.get(child.name, -1)
            # find any tip under child to get color
            def first_tip(c):
                if not c.clades: return c.name
                return first_tip(c.clades[0])
            tip_name = first_tip(child)
            c_id = clade_map.get(tip_name, -1)
            color = get_color(c_id) if c_id >= 0 else DIM

            ax.plot([clade._x, clade._x, child._x],
                    [clade._y, child._y, child._y],
                    color=color, lw=0.9, alpha=0.85,
                    solid_capstyle="round")
            draw(child)

    draw(tree.root)

    # leaf dots
    for t in tips:
        c_id = clade_map.get(t.name, -1)
        color = get_color(c_id) if c_id >= 0 else DIM
        ax.scatter(t._x, leaf_y[t.name], color=color, s=12, zorder=4,
                   linewidths=0)

    ax.set_title("IQ-TREE Topology — colored by clade", color=TEXT,
                 fontsize=10, pad=8)
    ax.set_xlabel("Branch length", color=DIM, fontsize=9)
    ax.set_yticks([])
    ax.invert_yaxis()


def plot_legend(ax, clade_map: dict):
    ax.set_facecolor(BG)
    ax.axis("off")
    unique = sorted(set(clade_map.values()))
    handles = [mpatches.Patch(facecolor=get_color(c),
                              label=f"Clade {c + 1}",
                              edgecolor="#1a2a3a")
               for c in unique]
    ax.legend(handles=handles, loc="center", frameon=False,
              labelcolor=TEXT, fontsize=9,
              ncol=max(1, len(unique) // 10))
    ax.set_title("Clade legend", color=TEXT, fontsize=9, pad=4)


def plot_stats(ax, stats: dict):
    ax.set_facecolor(BG)
    ax.axis("off")
    if not stats:
        ax.text(0.5, 0.5, "Not enough shared sequences\nto compute statistics",
                ha="center", va="center", color=DIM, fontsize=9,
                transform=ax.transAxes)
        return

    lines = [
        ("Shared sequences",  f"{stats['n_shared']}"),
        ("Mantel r (Spearman)", f"{stats['mantel_r']:.3f}"),
        ("Mantel p-value",    f"{stats['mantel_p']:.4f}"),
        ("Adjusted Rand Index", f"{stats['ARI']:.3f}"),
        ("Norm. Mutual Info",  f"{stats['NMI']:.3f}"),
    ]

    descs = {
        "Mantel r (Spearman)":  "Correlation between patristic & UMAP distances\n(1 = perfect agreement)",
        "Adjusted Rand Index":  "Cluster label agreement (1 = identical, 0 = random)",
        "Norm. Mutual Info":    "Shared information between clusterings (0–1)",
    }

    y = 0.95
    ax.text(0.5, y, "Quantitative Comparison", ha="center", va="top",
            color=TEXT, fontsize=10, fontweight="bold",
            transform=ax.transAxes)
    y -= 0.12

    for label, val in lines:
        ax.text(0.05, y, label, color=DIM, fontsize=8.5,
                transform=ax.transAxes, va="top")
        ax.text(0.95, y, val, color="#00d4ff", fontsize=9,
                fontweight="bold", ha="right",
                transform=ax.transAxes, va="top")
        y -= 0.08
        if label in descs:
            ax.text(0.05, y, descs[label], color="#2a4a6a", fontsize=7,
                    transform=ax.transAxes, va="top", style="italic")
            y -= 0.09

    # interpretation hint
    if "mantel_r" in stats:
        r = stats["mantel_r"]
        msg = ("Strong agreement" if r > 0.7
               else "Moderate agreement" if r > 0.4
               else "Weak agreement — methods diverge")
        color = ("#6bcb77" if r > 0.7 else "#ffd93d" if r > 0.4 else "#ff6b6b")
        ax.text(0.5, 0.05, f"▶  {msg}", ha="center", va="bottom",
                color=color, fontsize=9, fontweight="bold",
                transform=ax.transAxes)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compare IQ-TREE topology with UMAP embedding visually."
    )
    parser.add_argument("--tree",      required=True, help="IQ-TREE .treefile (Newick)")
    parser.add_argument("--umap",      required=True, help="UMAP CSV: name,umap1,umap2")
    parser.add_argument("--n-clades",  type=int, default=8,
                        help="Number of clades to cut the tree into (default: 8)")
    parser.add_argument("--out",       default="phylo_umap_comparison.pdf",
                        help="Output file (.pdf, .png, .svg)")
    parser.add_argument("--no-stats",  action="store_true",
                        help="Skip quantitative statistics (faster for large trees)")
    args = parser.parse_args()

    print("\n── Loading files ──────────────────────────────────────────")
    umap_df = load_umap(args.umap)

    if HAS_TOYTREE:
        print("  Using toytree for tree rendering")
        tree = load_tree_toytree(args.tree)
        use_toytree = True
    elif HAS_BIO:
        print("  Using biopython for tree rendering (install toytree for better trees)")
        tree = load_tree_biopython(args.tree)
        use_toytree = False
    else:
        sys.exit("ERROR: Install at least one of: toytree, biopython\n"
                 "  pip install biopython")

    print(f"\n── Assigning {args.n_clades} clades ───────────────────────────────")
    if use_toytree:
        clade_map = assign_clades_toytree(tree, args.n_clades)
    else:
        clade_map = assign_clades_biopython(tree, args.n_clades)

    matched = sum(1 for n in umap_df["name"] if n in clade_map)
    print(f"  Matched {matched}/{len(umap_df)} UMAP sequences to tree tips")
    if matched == 0:
        print("  WARNING: No sequences matched! Check that names are identical in both files.")

    stats = {}
    if not args.no_stats:
        print("\n── Computing statistics ───────────────────────────────────")
        print("  (This may take a moment for large datasets; use --no-stats to skip)")
        stats = compute_stats(umap_df, clade_map, tree, use_toytree)
        if stats:
            print(f"  Mantel r = {stats['mantel_r']:.3f}  (p = {stats['mantel_p']:.4f})")
            print(f"  ARI      = {stats['ARI']:.3f}")
            print(f"  NMI      = {stats['NMI']:.3f}")

    print("\n── Plotting ────────────────────────────────────────────────")
    fig = plt.figure(figsize=(18, 10), facecolor=BG)
    fig.suptitle("Phylogenetic Tree  ×  UMAP Embedding  —  Visual Comparison",
                 color=TEXT, fontsize=13, fontweight="bold", y=0.98)

    # Layout: [tree | umap | (legend + stats)]
    gs = fig.add_gridspec(2, 3,
                          width_ratios=[2, 2, 1],
                          height_ratios=[3, 1],
                          hspace=0.35, wspace=0.25,
                          left=0.04, right=0.97,
                          top=0.93, bottom=0.05)

    ax_tree  = fig.add_subplot(gs[0, 0])
    ax_umap  = fig.add_subplot(gs[0, 1])
    ax_stats = fig.add_subplot(gs[0, 2])
    ax_leg   = fig.add_subplot(gs[1, :2])
    ax_note  = fig.add_subplot(gs[1, 2])

    # draw panels
    if use_toytree:
        plot_tree_toytree(ax_tree, tree, clade_map)
    else:
        plot_tree_biopython(ax_tree, tree, clade_map)

    plot_umap(ax_umap, umap_df, clade_map)
    plot_stats(ax_stats, stats)
    plot_legend(ax_leg, clade_map)

    # note panel
    ax_note.set_facecolor(BG)
    ax_note.axis("off")
    note = (
        "How to read this figure:\n\n"
        "Colors are shared across both panels.\n"
        "If same-color dots cluster together\n"
        "in UMAP → the embedding recovers\n"
        "phylogenetic structure.\n\n"
        "Dashed ellipses = clade extent\n"
        "in UMAP space.\n\n"
        f"Tree cut into {args.n_clades} clades\n"
        f"(--n-clades to change)."
    )
    ax_note.text(0.05, 0.95, note, va="top", color=DIM, fontsize=8,
                 transform=ax_note.transAxes, linespacing=1.6)

    plt.savefig(args.out, dpi=180, bbox_inches="tight",
                facecolor=BG)
    print(f"\n✓  Saved → {args.out}")
    print("────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()