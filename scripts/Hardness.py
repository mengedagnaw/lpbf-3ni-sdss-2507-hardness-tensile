#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import re
from difflib import get_close_matches

# -------------------------------------------------
# 0. Read data from Excel (one sheet per condition)
# -------------------------------------------------
hra_file = "HRA_SDSS_3% Ni .xlsx"     # adjust path/name if needed
hv_file  = "HV0.1_SDSS_3% Ni.xlsx"

hra_dict = pd.read_excel(hra_file, sheet_name=None)
hv_dict  = pd.read_excel(hv_file, sheet_name=None)

# --------------------------
# Helpers (robust Excel I/O)
# --------------------------
def _norm(s) -> str:
    """Normalize strings for matching column/sheet names."""
    s = "" if s is None else str(s)
    s = s.strip().replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", "", s)  # remove punctuation/symbols
    return s

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
    return df

def ensure_header_row(df: pd.DataFrame, keywords=("surface", "core"), max_scan_rows=15) -> pd.DataFrame:
    """
    If expected keywords are not found in the current column names,
    scan the first rows to find a header-like row and promote it.
    """
    df = df.copy()

    # quick check: do current columns contain the keywords?
    col_norm = [_norm(c) for c in df.columns]
    if any(keywords[0] in c for c in col_norm) and any(keywords[1] in c for c in col_norm):
        return normalize_columns(df)

    # scan first rows to find the best header candidate
    scan_n = min(max_scan_rows, len(df))
    best_i, best_hits = None, -1

    for i in range(scan_n):
        row = df.iloc[i].tolist()
        row_norm = [_norm(v) for v in row]
        hits = sum(any(k in cell for cell in row_norm) for k in keywords)
        if hits > best_hits:
            best_hits = hits
            best_i = i

    # promote row if it looks like a header (both keywords found)
    if best_i is not None and best_hits >= 2:
        new_cols = df.iloc[best_i].tolist()
        df2 = df.iloc[best_i + 1 :].copy()
        df2.columns = new_cols
        df2 = df2.reset_index(drop=True)
        return normalize_columns(df2)

    # otherwise, return normalized columns anyway
    return normalize_columns(df)

def find_col(df: pd.DataFrame, candidates, *, allow_fuzzy=True) -> str:
    """
    Find a dataframe column matching candidates robustly.
    """
    # map normalized column -> original column name
    norm_map = {_norm(c): c for c in df.columns}
    cand_norm = [_norm(x) for x in candidates]

    # exact normalized match
    for cn in cand_norm:
        if cn in norm_map:
            return norm_map[cn]

    # contains match (e.g., "surface hardness hra")
    for cn in cand_norm:
        for k, orig in norm_map.items():
            if cn in k:
                return orig

    # fuzzy match fallback
    if allow_fuzzy:
        keys = list(norm_map.keys())
        for cn in cand_norm:
            m = get_close_matches(cn, keys, n=1, cutoff=0.70)
            if m:
                return norm_map[m[0]]

    raise KeyError(f"Could not find a column matching {candidates}. Available columns: {list(df.columns)}")

def to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").dropna()

def safe_std(vals: pd.Series) -> float:
    # ddof=1 requires at least 2 points; otherwise return NaN
    return float(vals.std(ddof=1)) if len(vals) >= 2 else float("nan")

def build_sheet_lookup(d: dict) -> dict:
    """Map normalized sheet name -> original sheet name"""
    return {_norm(k): k for k in d.keys()}

def build_summary_df(hra_sheets: dict, hv_sheets: dict) -> pd.DataFrame:
    rows = []

    hv_lookup = build_sheet_lookup(hv_sheets)

    for sheet_name, hra_df_raw in hra_sheets.items():
        key = _norm(sheet_name)
        if key not in hv_lookup:
            raise KeyError(
                f"Sheet '{sheet_name}' exists in HRA file but not in HV file.\n"
                f"HV sheets: {list(hv_sheets.keys())}"
            )

        hv_sheet_name = hv_lookup[key]
        hv_df_raw = hv_sheets[hv_sheet_name]

        cond_label = sheet_name.split("_")[0]  # "SR400_1h" → "SR400"

        # Repair/normalize headers if needed
        hra_df = ensure_header_row(hra_df_raw, keywords=("surface", "core"))
        hv_df  = ensure_header_row(hv_df_raw,  keywords=("hv", "0"))  # loose, just to catch weird header rows

        # --- HRA columns (robust match) ---
        surface_col = find_col(hra_df, ["Surface hardness", "Surface Hardness", "Surface"])
        core_col    = find_col(hra_df, ["Core hardness", "Core Hardness", "Core"])

        surface = to_numeric_series(hra_df[surface_col])
        core    = to_numeric_series(hra_df[core_col])

        # --- HV column (robust match) ---
        # First try your original rule but cleaned:
        hv_candidates = []
        for c in hv_df.columns:
            c_str = str(c).strip().replace("\n", " ")
            if c_str.startswith("HV0.1") or _norm(c_str).startswith("hv01"):
                hv_candidates.append(c)

        # Fallback: find something containing hv and 0.1
        if not hv_candidates:
            for c in hv_df.columns:
                n = _norm(c)
                if ("hv" in n) and ("01" in n or "0 1" in n or "0.1" in str(c)):
                    hv_candidates.append(c)

        if not hv_candidates:
            raise ValueError(
                f"No HV0.1-like column found in sheet '{hv_sheet_name}'. "
                f"Available columns: {list(hv_df.columns)}"
            )

        hv_col = hv_candidates[0]
        hv_vals = to_numeric_series(hv_df[hv_col])

        rows.append({
            "Condition_raw": sheet_name,
            "Condition": cond_label,

            "HRA_Surface": float(surface.mean()) if len(surface) else float("nan"),
            "HRA_Surface_Err": safe_std(surface),

            "HRA_Cross": float(core.mean()) if len(core) else float("nan"),
            "HRA_Cross_Err": safe_std(core),

            "HV": float(hv_vals.mean()) if len(hv_vals) else float("nan"),
            "HV_Err": safe_std(hv_vals),
        })

    df = pd.DataFrame(rows)

    # Optional: enforce a sensible order if your sheet order is random
    order_pref = ["AS", "SR400", "SR450", "SR500", "SR550", "SA1100"]
    df["Condition"] = pd.Categorical(df["Condition"], categories=order_pref, ordered=True)
    df = df.sort_values("Condition").reset_index(drop=True)

    return df

df = build_summary_df(hra_dict, hv_dict)

# -------------------------------------------------
# 1. Global style
# -------------------------------------------------
mpl.rcParams['text.usetex'] = False
mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 20,
    'font.weight': 'bold',
    'axes.titlesize': 26,
    'axes.labelsize': 24,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'xtick.labelsize': 20,
    'ytick.labelsize': 22,
    'axes.linewidth': 2.0,
    'xtick.major.width': 2.0,
    'ytick.major.width': 2.0,
    'legend.fontsize': 20,
    'figure.dpi': 600,
    'grid.linestyle': '-',
    'grid.alpha': 0.3,
})

# -------------------------------------------------
# Palette / gradients
# -------------------------------------------------
START_COLOR_SURFACE = '#88CCEE'
END_COLOR_SURFACE   = '#44AADD'

START_COLOR_CROSS   = '#CC6677'
END_COLOR_CROSS     = '#AA4455'

START_COLOR_HV      = '#3cb371'
END_COLOR_HV        = '#0086A2'

cmap_surface = LinearSegmentedColormap.from_list("surface_grad", [START_COLOR_SURFACE, END_COLOR_SURFACE])
cmap_cross   = LinearSegmentedColormap.from_list("cross_grad",   [START_COLOR_CROSS,   END_COLOR_CROSS])
cmap_hv      = LinearSegmentedColormap.from_list("hv_grad",      [START_COLOR_HV,      END_COLOR_HV])

# -------------------------------------------------
# 2. Figure and axes
# -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(20, 9))
ax1, ax2 = axes
fig.patch.set_facecolor('white')



for ax in axes:
    ax.set_facecolor('#ffffff')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)

# -------------------------------------------------
# 3. HRA plot (left)
# -------------------------------------------------
x = np.arange(len(df["Condition"]))
bar_width = 0.35

for i, row in df.iterrows():
    frac = i / max(len(df) - 1, 1)
    color_s = cmap_surface(0.2 + 0.3 * frac)
    color_c = cmap_cross(0.2 + 0.3 * frac)

    ax1.bar(
        x[i] - bar_width/2, row["HRA_Surface"], bar_width,
        yerr=row["HRA_Surface_Err"], capsize=8, ecolor='black',
        color=color_s, edgecolor='black', linewidth=1.5, zorder=3
    )
    ax1.bar(
        x[i] + bar_width/2, row["HRA_Cross"], bar_width,
        yerr=row["HRA_Cross_Err"], capsize=8, ecolor='black',
        color=color_c, edgecolor='black', linewidth=1.5, zorder=3
    )

ax1.set_ylabel('HRA', rotation=90, labelpad=18)
ax1.set_xticks(x)
labels1 = ax1.set_xticklabels(df["Condition"], rotation=15, ha='right')
for lbl in labels1:
    lbl.set_fontweight('bold')

ax1.set_ylim(50, 80)
ax1.set_yticks(np.arange(50, 81, 5))
ax1.grid(axis='y', linewidth=1.0, alpha=0.4, zorder=0)
ax1.set_axisbelow(True)

ax1.legend(
    handles=[
        Patch(facecolor=cmap_surface(0.5), edgecolor='black', label='Surface'),
        Patch(facecolor=cmap_cross(0.5),   edgecolor='black', label='Core (Cross-section)')
    ],
    loc='upper left',
    ncol=1,
    frameon=False,
    edgecolor='black',
    prop={'weight': 'bold', 'size': 20},
    bbox_to_anchor=(0.32, 0.98)
)

# -------------------------------------------------
# 4. HV plot (right)
# -------------------------------------------------
for i, row in df.iterrows():
    frac = i / max(len(df) - 1, 1)
    color_h = cmap_hv(0.2 + 0.3 * frac)
    ax2.bar(
        x[i], row["HV"], bar_width * 1.2,
        yerr=row["HV_Err"], capsize=8, ecolor='black',
        color=color_h, edgecolor='black', linewidth=1.5, zorder=3
    )

ax2.set_ylabel(r'HV$_{0.1}$', rotation=90, labelpad=18)
ax2.set_xticks(x)
labels2 = ax2.set_xticklabels(df["Condition"], rotation=15, ha='right')
for lbl in labels2:
    lbl.set_fontweight('bold')

ax2.set_ylim(250, 550)
ax2.set_yticks(np.arange(250, 551, 50))
ax2.grid(axis='y', linewidth=1.0, alpha=0.4, zorder=0)
ax2.set_axisbelow(True)

# -------------------------------------------------
# 5. Layout & save
# -------------------------------------------------
plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.14, wspace=0.25)
plt.savefig("hardness_gradient_tolmuted.png", dpi=600, bbox_inches="tight")
plt.show()


# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import re
from difflib import get_close_matches

# -------------------------------------------------
# 0. Read data from Excel (one sheet per condition)
# -------------------------------------------------
hra_file = "HRA_SDSS_3% Ni .xlsx"     # adjust path/name if needed
hv_file  = "HV0.1_SDSS_3% Ni.xlsx"

hra_dict = pd.read_excel(hra_file, sheet_name=None)
hv_dict  = pd.read_excel(hv_file, sheet_name=None)

# --------------------------
# Helpers (robust Excel I/O)
# --------------------------
def _norm(s) -> str:
    """Normalize strings for matching column/sheet names."""
    s = "" if s is None else str(s)
    s = s.strip().replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
    return df

def ensure_header_row(df: pd.DataFrame, keywords=("surface", "core"), max_scan_rows=15) -> pd.DataFrame:
    """
    If expected keywords are not found in the current column names,
    scan the first rows to find a header-like row and promote it.
    """
    df = df.copy()

    col_norm = [_norm(c) for c in df.columns]
    if any(keywords[0] in c for c in col_norm) and any(keywords[1] in c for c in col_norm):
        return normalize_columns(df)

    scan_n = min(max_scan_rows, len(df))
    best_i, best_hits = None, -1

    for i in range(scan_n):
        row = df.iloc[i].tolist()
        row_norm = [_norm(v) for v in row]
        hits = sum(any(k in cell for cell in row_norm) for k in keywords)
        if hits > best_hits:
            best_hits = hits
            best_i = i

    if best_i is not None and best_hits >= 2:
        new_cols = df.iloc[best_i].tolist()
        df2 = df.iloc[best_i + 1:].copy()
        df2.columns = new_cols
        df2 = df2.reset_index(drop=True)
        return normalize_columns(df2)

    return normalize_columns(df)

def find_col(df: pd.DataFrame, candidates, *, allow_fuzzy=True) -> str:
    """
    Find a dataframe column matching candidates robustly.
    """
    norm_map = {_norm(c): c for c in df.columns}
    cand_norm = [_norm(x) for x in candidates]

    for cn in cand_norm:
        if cn in norm_map:
            return norm_map[cn]

    for cn in cand_norm:
        for k, orig in norm_map.items():
            if cn in k:
                return orig

    if allow_fuzzy:
        keys = list(norm_map.keys())
        for cn in cand_norm:
            m = get_close_matches(cn, keys, n=1, cutoff=0.70)
            if m:
                return norm_map[m[0]]

    raise KeyError(f"Could not find a column matching {candidates}. Available columns: {list(df.columns)}")

def to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").dropna()

def safe_std(vals: pd.Series) -> float:
    return float(vals.std(ddof=1)) if len(vals) >= 2 else float("nan")

def build_sheet_lookup(d: dict) -> dict:
    return {_norm(k): k for k in d.keys()}

def build_summary_df(hra_sheets: dict, hv_sheets: dict) -> pd.DataFrame:
    rows = []

    hv_lookup = build_sheet_lookup(hv_sheets)

    for sheet_name, hra_df_raw in hra_sheets.items():
        key = _norm(sheet_name)
        if key not in hv_lookup:
            raise KeyError(
                f"Sheet '{sheet_name}' exists in HRA file but not in HV file.\n"
                f"HV sheets: {list(hv_sheets.keys())}"
            )

        hv_sheet_name = hv_lookup[key]
        hv_df_raw = hv_sheets[hv_sheet_name]

        cond_label = sheet_name.split("_")[0]

        hra_df = ensure_header_row(hra_df_raw, keywords=("surface", "core"))
        hv_df  = ensure_header_row(hv_df_raw, keywords=("hv", "0"))

        surface_col = find_col(hra_df, ["Surface hardness", "Surface Hardness", "Surface"])
        core_col    = find_col(hra_df, ["Core hardness", "Core Hardness", "Core"])

        surface = to_numeric_series(hra_df[surface_col])
        core    = to_numeric_series(hra_df[core_col])

        hv_candidates = []
        for c in hv_df.columns:
            c_str = str(c).strip().replace("\n", " ")
            if c_str.startswith("HV0.1") or _norm(c_str).startswith("hv01"):
                hv_candidates.append(c)

        if not hv_candidates:
            for c in hv_df.columns:
                n = _norm(c)
                if ("hv" in n) and ("01" in n or "0 1" in n or "0.1" in str(c)):
                    hv_candidates.append(c)

        if not hv_candidates:
            raise ValueError(
                f"No HV0.1-like column found in sheet '{hv_sheet_name}'. "
                f"Available columns: {list(hv_df.columns)}"
            )

        hv_col = hv_candidates[0]
        hv_vals = to_numeric_series(hv_df[hv_col])

        rows.append({
            "Condition_raw": sheet_name,
            "Condition": cond_label,
            "HRA_Surface": float(surface.mean()) if len(surface) else float("nan"),
            "HRA_Surface_Err": safe_std(surface),
            "HRA_Cross": float(core.mean()) if len(core) else float("nan"),
            "HRA_Cross_Err": safe_std(core),
            "HV": float(hv_vals.mean()) if len(hv_vals) else float("nan"),
            "HV_Err": safe_std(hv_vals),
        })

    df = pd.DataFrame(rows)

    order_pref = ["AS", "SR400", "SR450", "SR500", "SR550", "SA1100"]
    df["Condition"] = pd.Categorical(df["Condition"], categories=order_pref, ordered=True)
    df = df.sort_values("Condition").reset_index(drop=True)

    return df

df = build_summary_df(hra_dict, hv_dict)

# -------------------------------------------------
# 1. Global style
# -------------------------------------------------
mpl.rcParams['text.usetex'] = False
mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 20,
    'font.weight': 'bold',
    'axes.titlesize': 26,
    'axes.labelsize': 24,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'xtick.labelsize': 20,
    'ytick.labelsize': 22,
    'axes.linewidth': 2.0,
    'xtick.major.width': 2.0,
    'ytick.major.width': 2.0,
    'legend.fontsize': 20,
    'figure.dpi': 600,
    'grid.linestyle': '-',
    'grid.alpha': 0.3,
})

# -------------------------------------------------
# Palette / gradients (more creative + distinct)
# -------------------------------------------------
START_COLOR_SURFACE = '#A7F3D0'   # seafoam
END_COLOR_SURFACE   = '#1D4ED8'   # royal blue

START_COLOR_CROSS   = '#F9A8D4'   # orchid pink
END_COLOR_CROSS     = '#9D174D'   # wine magenta

START_COLOR_HV      = '#FDE68A'   # soft gold
END_COLOR_HV        = '#047857'   # emerald green

cmap_surface = LinearSegmentedColormap.from_list(
    "surface_grad", [START_COLOR_SURFACE, END_COLOR_SURFACE]
)
cmap_cross = LinearSegmentedColormap.from_list(
    "cross_grad", [START_COLOR_CROSS, END_COLOR_CROSS]
)
cmap_hv = LinearSegmentedColormap.from_list(
    "hv_grad", [START_COLOR_HV, END_COLOR_HV]
)


cmap_surface = LinearSegmentedColormap.from_list("surface_grad", [START_COLOR_SURFACE, END_COLOR_SURFACE])
cmap_cross   = LinearSegmentedColormap.from_list("cross_grad",   [START_COLOR_CROSS,   END_COLOR_CROSS])
cmap_hv      = LinearSegmentedColormap.from_list("hv_grad",      [START_COLOR_HV,      END_COLOR_HV])

# -------------------------------------------------
# 2. Figure and axes
# -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(20, 9))
ax1, ax2 = axes
fig.patch.set_facecolor('white')

for ax in axes:
    ax.set_facecolor('#ffffff')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)

# -------------------------------------------------
# 3. HRA plot (left)
# -------------------------------------------------
x = np.arange(len(df["Condition"]))
bar_width = 0.35

for i, row in df.iterrows():
    frac = i / max(len(df) - 1, 1)
    color_s = cmap_surface(0.25 + 0.65 * frac)
    color_c = cmap_cross(0.25 + 0.65 * frac)

    ax1.bar(
        x[i] - bar_width/2, row["HRA_Surface"], bar_width,
        yerr=row["HRA_Surface_Err"], capsize=8, ecolor='black',
        color=color_s, edgecolor='black', linewidth=1.5, zorder=3
    )
    ax1.bar(
        x[i] + bar_width/2, row["HRA_Cross"], bar_width,
        yerr=row["HRA_Cross_Err"], capsize=8, ecolor='black',
        color=color_c, edgecolor='black', linewidth=1.5, zorder=3
    )

ax1.set_ylabel('HRA', rotation=90, labelpad=18)
ax1.set_xticks(x)
labels1 = ax1.set_xticklabels(df["Condition"], rotation=15, ha='right')
for lbl in labels1:
    lbl.set_fontweight('bold')

ax1.set_ylim(50, 80)
ax1.set_yticks(np.arange(50, 81, 5))
ax1.grid(axis='y', linewidth=1.0, alpha=0.4, zorder=0)
ax1.set_axisbelow(True)

ax1.legend(
    handles=[
        Patch(facecolor=cmap_surface(0.75), edgecolor='black', label='Surface'),
        Patch(facecolor=cmap_cross(0.75), edgecolor='black', label='Core (Cross-section)')
    ],
    loc='upper left',
    ncol=1,
    frameon=False,
    prop={'weight': 'bold', 'size': 20},
    bbox_to_anchor=(0.30, 0.98)
)

# -------------------------------------------------
# 4. HV plot (right)
# -------------------------------------------------
for i, row in df.iterrows():
    frac = i / max(len(df) - 1, 1)
    color_h = cmap_hv(0.25 + 0.65 * frac)
    ax2.bar(
        x[i], row["HV"], bar_width * 1.2,
        yerr=row["HV_Err"], capsize=8, ecolor='black',
        color=color_h, edgecolor='black', linewidth=1.5, zorder=3
    )

ax2.set_ylabel(r'HV$_{0.1}$', rotation=90, labelpad=18)
ax2.set_xticks(x)
labels2 = ax2.set_xticklabels(df["Condition"], rotation=15, ha='right')
for lbl in labels2:
    lbl.set_fontweight('bold')

ax2.set_ylim(250, 550)
ax2.set_yticks(np.arange(250, 551, 50))
ax2.grid(axis='y', linewidth=1.0, alpha=0.4, zorder=0)
ax2.set_axisbelow(True)

# -------------------------------------------------
# 5. Layout & save
# -------------------------------------------------
plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.14, wspace=0.25)
plt.savefig("hardness_distinct_palette.png", dpi=600, bbox_inches="tight")
plt.show()


# In[ ]:




