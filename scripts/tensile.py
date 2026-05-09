#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.ticker import FuncFormatter, AutoMinorLocator, MultipleLocator

# ============================================================
# Experimental tensile processing only (no FEM)
# White background, no panel letters, plain black labels,
# distinct color palette for each condition,
# x-axis percent ticks at 1% intervals
# ============================================================

# ---------------------------
# Setup output directories
# ---------------------------
os.makedirs("aligned_by_drop_results", exist_ok=True)
os.makedirs("comparison_plots", exist_ok=True)

# ---------------------------
# Input files
# ---------------------------
EXP_INPUT_SETS = [
    ("AS",     "AS.xlsx",            "AREA_3%Ni_AS _EN.csv"),
    ("SR400",  "SR400_1h .xlsx",     "AREA_3%Ni_SR400_EN.csv"),
    ("SR450",  "SR450_1h.xlsx",      "AREA_3%Ni_SR450 _EN.csv"),
    ("SR500",  "SR500_1h.xlsx",      "AREA_3%Ni_SR500 _EN.csv"),
    ("SR550",  "SR550_1h.xlsx",      "AREA_3%Ni_SR550_EN.csv"),
    ("SA1100", "SA1100_15min.xlsx",  "AREA_3%Ni_SA1100 _EN.csv"),
]

# ---------------------------
# Plot settings
# ---------------------------
DPI = 300
SMOOTH_MEAN_CURVE = True
SMOOTH_WINDOW_FRAC = 0.01
PANEL_BG = "white"

PALETTE_MAP = {
    "AS": {
        "replicate": "#8FB3D9",
        "mean": "#1E4E8C"
    },
    "SR400": {
        "replicate": "#E8B26D",
        "mean": "#B86B00"
    },
    "SR450": {
        "replicate": "#8FCB9B",
        "mean": "#2F7D4A"
    },
    "SR500": {
        "replicate": "#D99AA4",
        "mean": "#A63D57"
    },
    "SR550": {
        "replicate": "#B7A1D6",
        "mean": "#6C3FA1"
    },
    "SA1100": {
        "replicate": "#7FC7C2",
        "mean": "#007A78"
    },
}

LABEL_MAP = {
    "AS": "AS",
    "SR400": "SR400",
    "SR450": "SR450",
    "SR500": "SR500",
    "SR550": "SR550",
    "SA1100": "SA1100",
}

XMAX_MAP = {
    "AS": 3.5,
    "SR400": 5.2,
    "SR450": 5.2,
    "SR500": 5.2,
    "SR550": 5.2,
    "SA1100": 3.8,
}
YMAX_DEFAULT = 1450

# ---------------------------
# Helper functions
# ---------------------------
def normalize_text(s):
    return str(s).strip().lower().replace(" ", "")

def smooth(y, window_frac=0.01):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 5:
        return y

    win = max(5, int(window_frac * n))
    if win % 2 == 0:
        win += 1
    if win >= n:
        win = n - 1 if n % 2 == 0 else n
    if win < 5:
        return y

    return savgol_filter(y, window_length=win, polyorder=2, mode="interp")

def percent_formatter(x, pos):
    return f"{x:.0f}%"

def style_axis(ax):
    ax.set_facecolor(PANEL_BG)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    ax.tick_params(
        axis="both", which="major",
        direction="out", length=4, width=1.0,
        colors="black", labelsize=11
    )
    ax.tick_params(
        axis="both", which="minor",
        direction="out", length=2.5, width=0.8,
        colors="black"
    )

    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.8, alpha=0.65)
    ax.grid(True, which="minor", color="#e8e8e8", linewidth=0.5, alpha=0.45)

def add_condition_label(ax, label):
    ax.text(
        0.46, 0.16, label,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=13, fontweight="bold", color="black"
    )

def load_area_values(area_csv):
    values = []
    with open(area_csv, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            txt = line.strip().replace(",", ".")
            try:
                values.append(float(txt))
            except ValueError:
                continue
    return values

def detect_columns(df):
    strain_col = None
    load_col = None

    for col in df.columns:
        c = normalize_text(col)

        if strain_col is None and "strain" in c:
            strain_col = col

        if load_col is None:
            if "force" in c or "load" in c:
                load_col = col
            elif "stress" in c and "n" in c:
                load_col = col

    if strain_col is None:
        raise ValueError(f"Could not detect strain column in columns: {list(df.columns)}")
    if load_col is None:
        raise ValueError(f"Could not detect load column in columns: {list(df.columns)}")

    return strain_col, load_col

def clean_sample_curve(df, strain_col, load_col):
    out = df[[strain_col, load_col]].copy()
    out.columns = ["Strain_percent", "Load_N"]

    out["Strain_percent"] = pd.to_numeric(out["Strain_percent"], errors="coerce")
    out["Load_N"] = pd.to_numeric(out["Load_N"], errors="coerce")
    out = out.dropna(subset=["Strain_percent", "Load_N"])

    out = out[out["Strain_percent"] > 0]
    out = out[out["Load_N"] > 0]

    out = out.sort_values("Strain_percent").reset_index(drop=True)
    out = out.groupby("Strain_percent", as_index=False)["Load_N"].mean()

    return out

def plot_experimental_style(valid_curves, strain_grid, mean_stress, condition_key, display_label, out_path,
                            x_max=None, y_max=None):
    fig, ax = plt.subplots(figsize=(4.8, 3.8), dpi=DPI)

    palette = PALETTE_MAP.get(condition_key, {
        "replicate": "#A0A0A0",
        "mean": "#303030"
    })

    replicate_color = palette["replicate"]
    mean_color = palette["mean"]

    for s_arr, t_arr in valid_curves:
        ax.plot(
            s_arr * 100.0,
            t_arr,
            color=replicate_color,
            linewidth=1.0,
            alpha=0.75
        )

    ax.plot(
        strain_grid * 100.0,
        mean_stress,
        color=mean_color,
        linewidth=2.4,
        alpha=1.0
    )

    style_axis(ax)

    ax.set_xlabel(r'$\varepsilon$ (%)', fontsize=15, fontweight='bold')
    ax.set_ylabel(r'$\sigma$ (MPa)', fontsize=15, fontweight='bold')

    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))

    if x_max is None:
        x_max = np.nanmax(strain_grid * 100.0) * 1.03
    if y_max is None:
        y_max = max(YMAX_DEFAULT, np.nanmax(mean_stress) * 1.03)

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)

    add_condition_label(ax, display_label)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", transparent=False)
    plt.close()

# ---------------------------
# Main experimental preparation
# ---------------------------
def prepare_exp_data_from_xlsx(condition_name, xlsx_file, area_csv):
    try:
        if not os.path.exists(xlsx_file):
            print(f"Error: missing XLSX file: {xlsx_file}")
            return False, {}

        if not os.path.exists(area_csv):
            print(f"Error: missing area CSV file: {area_csv}")
            return False, {}

        xl = pd.ExcelFile(xlsx_file)
        sheets = xl.sheet_names
        area_values = load_area_values(area_csv)

        if len(area_values) < len(sheets):
            raise ValueError(
                f"{condition_name}: only {len(area_values)} area values found, "
                f"but {len(sheets)} sheets exist."
            )

        if len(area_values) != len(sheets):
            print(
                f"Warning: {condition_name} has {len(sheets)} sheets but {len(area_values)} area values. "
                f"Using the first {len(sheets)} areas."
            )

        valid_curves = []
        max_strains = []
        used_areas = []
        raw_export_rows = []

        for i, sheet in enumerate(sheets):
            df = pd.read_excel(xlsx_file, sheet_name=sheet)
            strain_col, load_col = detect_columns(df)
            sample_df = clean_sample_curve(df, strain_col, load_col)

            if len(sample_df) < 5:
                continue

            area_mm2 = area_values[i]
            used_areas.append(area_mm2)

            sample_df["Stress_MPa"] = sample_df["Load_N"] / area_mm2
            sample_df["Strain"] = sample_df["Strain_percent"] / 100.0

            sample_df = sample_df[["Strain", "Stress_MPa", "Load_N", "Strain_percent"]].copy()
            sample_df = sample_df.dropna()
            sample_df = sample_df[sample_df["Strain"] > 0]
            sample_df = sample_df[sample_df["Stress_MPa"] > 0]
            sample_df = sample_df.sort_values("Strain").drop_duplicates(subset=["Strain"], keep="first")

            if len(sample_df) < 5:
                continue

            max_strains.append(sample_df["Strain"].max())
            valid_curves.append((sample_df["Strain"].values, sample_df["Stress_MPa"].values))

            sample_df["Condition"] = condition_name
            sample_df["Sample"] = sheet
            sample_df["Area_mm2"] = area_mm2
            raw_export_rows.append(sample_df)

        if not valid_curves:
            raise ValueError(f"No valid samples found for {condition_name}")

        max_common_strain = min(max_strains)
        strain_grid = np.linspace(0.0, max_common_strain, 1000)

        interp_stresses = []
        for s_arr, t_arr in valid_curves:
            stress_interp = np.interp(strain_grid, s_arr, t_arr)
            interp_stresses.append(stress_interp)

        stress_matrix = np.vstack(interp_stresses)
        mean_stress = np.mean(stress_matrix, axis=0)
        std_stress = np.std(stress_matrix, axis=0)

        if SMOOTH_MEAN_CURVE:
            mean_stress = smooth(mean_stress, window_frac=SMOOTH_WINDOW_FRAC)

        df_processed = pd.DataFrame({
            "Strain": strain_grid,
            "FinalMean": mean_stress,
            "StdStress": std_stress
        })

        processed_csv = os.path.join("aligned_by_drop_results", f"{condition_name}_processed_aligned.csv")
        raw_csv = os.path.join("aligned_by_drop_results", f"{condition_name}_raw_converted.csv")

        df_processed.to_csv(processed_csv, index=False)
        pd.concat(raw_export_rows, ignore_index=True).to_csv(raw_csv, index=False)

        plot_path = os.path.join("comparison_plots", f"{condition_name}_experimental_mean_styled.png")

        plot_experimental_style(
            valid_curves=valid_curves,
            strain_grid=strain_grid,
            mean_stress=mean_stress,
            condition_key=condition_name,
            display_label=LABEL_MAP.get(condition_name, condition_name),
            out_path=plot_path,
            x_max=XMAX_MAP.get(condition_name, None),
            y_max=YMAX_DEFAULT
        )

        metadata = {
            "condition": condition_name,
            "processed_csv": processed_csv,
            "raw_csv": raw_csv,
            "plot_path": plot_path,
            "mean_area_mm2": float(np.mean(used_areas)),
            "areas_mm2": used_areas,
            "n_samples": len(valid_curves),
        }

        print(f"Prepared: {condition_name}")
        print(f"  Mean curve CSV: {processed_csv}")
        print(f"  Raw converted CSV: {raw_csv}")
        print(f"  Styled plot: {plot_path}")
        print(f"  Samples used: {len(valid_curves)}")
        print(f"  Mean area: {metadata['mean_area_mm2']:.4f} mm²")

        return True, metadata

    except Exception as e:
        print(f"Error while processing {condition_name}: {e}")
        return False, {}

# ---------------------------
# Run all conditions
# ---------------------------
exp_meta = {}

for condition_name, xlsx_file, area_csv in EXP_INPUT_SETS:
    ready, meta = prepare_exp_data_from_xlsx(condition_name, xlsx_file, area_csv)
    if ready:
        exp_meta[condition_name] = meta

# ---------------------------
# Combined comparison plot
# ---------------------------
if exp_meta:
    fig, ax = plt.subplots(figsize=(8.5, 5.8), dpi=DPI)
    ax.set_facecolor(PANEL_BG)

    for condition_name, _, _ in EXP_INPUT_SETS:
        if condition_name not in exp_meta:
            continue

        df = pd.read_csv(exp_meta[condition_name]["processed_csv"])
        palette = PALETTE_MAP.get(condition_name, {
            "replicate": "#A0A0A0",
            "mean": "#303030"
        })

        ax.plot(
            df["Strain"] * 100.0,
            df["FinalMean"],
            linewidth=2.2,
            color=palette["mean"],
            label=condition_name
        )

    style_axis(ax)
    ax.set_xlabel(r'$\varepsilon$ (%)', fontsize=15, fontweight='bold')
    ax.set_ylabel(r'$\sigma$ (MPa)', fontsize=15, fontweight='bold')

    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))

    ax.set_xlim(0, 5.5)
    ax.set_ylim(0, YMAX_DEFAULT)
    ax.legend(frameon=False, fontsize=11)
    ax.set_title("Experimental Mean Stress-Strain Curves (3 wt.% Ni)", fontsize=15, pad=10)

    plt.tight_layout()
    combined_plot = os.path.join("comparison_plots", "all_experimental_mean_curves_styled.png")
    plt.savefig(combined_plot, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"\nSaved combined plot: {combined_plot}")
    print("Done.")
else:
    print("No experimental datasets were processed.")


# In[3]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.ticker import FuncFormatter, AutoMinorLocator, MultipleLocator

# ============================================================
# Experimental tensile processing only (no FEM)
# White background, no panel letters, plain black labels,
# distinct color palette for each condition,
# major/minor ticks styled like the example figure
# ============================================================

# ---------------------------
# Setup output directories
# ---------------------------
os.makedirs("aligned_by_drop_results", exist_ok=True)
os.makedirs("comparison_plots", exist_ok=True)

# ---------------------------
# Input files
# ---------------------------
# Assumes this notebook/script is in the SAME folder as these files
EXP_INPUT_SETS = [
    ("AS",     "AS.xlsx",            "AREA_3%Ni_AS _EN.csv"),
    ("SR400",  "SR400_1h .xlsx",     "AREA_3%Ni_SR400_EN.csv"),
    ("SR450",  "SR450_1h.xlsx",      "AREA_3%Ni_SR450 _EN.csv"),
    ("SR500",  "SR500_1h.xlsx",      "AREA_3%Ni_SR500 _EN.csv"),
    ("SR550",  "SR550_1h.xlsx",      "AREA_3%Ni_SR550_EN.csv"),
    ("SA1100", "SA1100_15min.xlsx",  "AREA_3%Ni_SA1100 _EN.csv"),
]

# ---------------------------
# Plot settings
# ---------------------------
DPI = 300
SMOOTH_MEAN_CURVE = True
SMOOTH_WINDOW_FRAC = 0.01
PANEL_BG = "white"

# Distinct journal-style palette
PALETTE_MAP = {
    "AS": {
        "replicate": "#8FB3D9",
        "mean": "#1E4E8C"
    },
    "SR400": {
        "replicate": "#E8B26D",
        "mean": "#B86B00"
    },
    "SR450": {
        "replicate": "#8FCB9B",
        "mean": "#2F7D4A"
    },
    "SR500": {
        "replicate": "#D99AA4",
        "mean": "#A63D57"
    },
    "SR550": {
        "replicate": "#B7A1D6",
        "mean": "#6C3FA1"
    },
    "SA1100": {
        "replicate": "#7FC7C2",
        "mean": "#007A78"
    },
}

# Plain text labels
LABEL_MAP = {
    "AS": "AS",
    "SR400": "SR400",
    "SR450": "SR450",
    "SR500": "SR500",
    "SR550": "SR550",
    "SA1100": "SA1100",
}

# Optional fixed axis ranges
XMAX_MAP = {
    "AS": 3.5,
    "SR400": 5.2,
    "SR450": 5.2,
    "SR500": 5.2,
    "SR550": 5.2,
    "SA1100": 3.8,
}
YMAX_DEFAULT = 1450

# ---------------------------
# Helper functions
# ---------------------------
def normalize_text(s):
    return str(s).strip().lower().replace(" ", "")

def smooth(y, window_frac=0.01):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 5:
        return y

    win = max(5, int(window_frac * n))
    if win % 2 == 0:
        win += 1
    if win >= n:
        win = n - 1 if n % 2 == 0 else n
    if win < 5:
        return y

    return savgol_filter(y, window_length=win, polyorder=2, mode="interp")

def percent_formatter(x, pos):
    return f"{x:.0f}%"

def style_axis(ax):
    ax.set_facecolor(PANEL_BG)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    # Major ticks
    ax.tick_params(
        axis="both", which="major",
        direction="out", length=7, width=1.6,
        colors="black", labelsize=11
    )

    # Minor ticks
    ax.tick_params(
        axis="both", which="minor",
        direction="out", length=4, width=1.0,
        colors="black"
    )

    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.8, alpha=0.65)
    ax.grid(True, which="minor", color="#eeeeee", linewidth=0.5, alpha=0.35)

def add_condition_label(ax, label):
    ax.text(
        0.46, 0.16, label,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=13, fontweight="bold", color="black"
    )

def load_area_values(area_csv):
    """
    Reads area values from files like:
        S0
        mm²
        32.7756
        33.174
        ...
    Returns a list of floats in mm^2.
    """
    values = []
    with open(area_csv, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            txt = line.strip().replace(",", ".")
            try:
                values.append(float(txt))
            except ValueError:
                continue
    return values

def detect_columns(df):
    """
    Detect strain and load columns.
    In your files the load column may be labeled as Stress(N),
    but physically that is load in Newtons.
    """
    strain_col = None
    load_col = None

    for col in df.columns:
        c = normalize_text(col)

        if strain_col is None and "strain" in c:
            strain_col = col

        if load_col is None:
            if "force" in c or "load" in c:
                load_col = col
            elif "stress" in c and "n" in c:
                load_col = col

    if strain_col is None:
        raise ValueError(f"Could not detect strain column in columns: {list(df.columns)}")
    if load_col is None:
        raise ValueError(f"Could not detect load column in columns: {list(df.columns)}")

    return strain_col, load_col

def clean_sample_curve(df, strain_col, load_col):
    """
    Cleans one sample curve:
      - numeric conversion
      - drop NaNs
      - keep positive values
      - sort by strain
      - average duplicate strain values
    """
    out = df[[strain_col, load_col]].copy()
    out.columns = ["Strain_percent", "Load_N"]

    out["Strain_percent"] = pd.to_numeric(out["Strain_percent"], errors="coerce")
    out["Load_N"] = pd.to_numeric(out["Load_N"], errors="coerce")
    out = out.dropna(subset=["Strain_percent", "Load_N"])

    out = out[out["Strain_percent"] > 0]
    out = out[out["Load_N"] > 0]

    out = out.sort_values("Strain_percent").reset_index(drop=True)
    out = out.groupby("Strain_percent", as_index=False)["Load_N"].mean()

    return out

def plot_experimental_style(valid_curves, strain_grid, mean_stress, condition_key, display_label, out_path,
                            x_max=None, y_max=None):
    """
    valid_curves: list of tuples [(strain_unitless, stress_mpa), ...]
    strain_grid: unitless strain array for mean curve
    mean_stress: mean stress in MPa
    condition_key: key for palette map
    display_label: plain black text label
    """
    fig, ax = plt.subplots(figsize=(4.8, 3.8), dpi=DPI)

    palette = PALETTE_MAP.get(condition_key, {
        "replicate": "#A0A0A0",
        "mean": "#303030"
    })

    replicate_color = palette["replicate"]
    mean_color = palette["mean"]

    # Replicate curves
    for s_arr, t_arr in valid_curves:
        ax.plot(
            s_arr * 100.0,
            t_arr,
            color=replicate_color,
            linewidth=1.0,
            alpha=0.75
        )

    # Mean curve
    ax.plot(
        strain_grid * 100.0,
        mean_stress,
        color=mean_color,
        linewidth=2.4,
        alpha=1.0
    )

    style_axis(ax)

    ax.set_xlabel(r'$\varepsilon$ (%)', fontsize=15, fontweight='bold')
    ax.set_ylabel(r'$\sigma$ (MPa)', fontsize=15, fontweight='bold')

    # Tick styling like the example
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))   # 4 minor ticks between 1% majors
    ax.yaxis.set_major_locator(MultipleLocator(250))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))   # 1 minor tick between 250 MPa majors
    ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))

    if x_max is None:
        x_max = np.nanmax(strain_grid * 100.0) * 1.03
    if y_max is None:
        y_max = max(YMAX_DEFAULT, np.nanmax(mean_stress) * 1.03)

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)

    add_condition_label(ax, display_label)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", transparent=False)
    plt.close()

# ---------------------------
# Main experimental preparation
# ---------------------------
def prepare_exp_data_from_xlsx(condition_name, xlsx_file, area_csv):
    """
    Reads one XLSX + area CSV, converts:
        load (N) -> engineering stress (MPa)
        strain (%) -> unitless strain
    Then computes the mean experimental curve for that condition.
    """
    try:
        if not os.path.exists(xlsx_file):
            print(f"Error: missing XLSX file: {xlsx_file}")
            return False, {}

        if not os.path.exists(area_csv):
            print(f"Error: missing area CSV file: {area_csv}")
            return False, {}

        xl = pd.ExcelFile(xlsx_file)
        sheets = xl.sheet_names
        area_values = load_area_values(area_csv)

        if len(area_values) < len(sheets):
            raise ValueError(
                f"{condition_name}: only {len(area_values)} area values found, "
                f"but {len(sheets)} sheets exist."
            )

        if len(area_values) != len(sheets):
            print(
                f"Warning: {condition_name} has {len(sheets)} sheets but {len(area_values)} area values. "
                f"Using the first {len(sheets)} areas."
            )

        valid_curves = []
        max_strains = []
        used_areas = []
        raw_export_rows = []

        for i, sheet in enumerate(sheets):
            df = pd.read_excel(xlsx_file, sheet_name=sheet)
            strain_col, load_col = detect_columns(df)
            sample_df = clean_sample_curve(df, strain_col, load_col)

            if len(sample_df) < 5:
                continue

            area_mm2 = area_values[i]
            used_areas.append(area_mm2)

            # 1 N/mm² = 1 MPa
            sample_df["Stress_MPa"] = sample_df["Load_N"] / area_mm2
            sample_df["Strain"] = sample_df["Strain_percent"] / 100.0

            sample_df = sample_df[["Strain", "Stress_MPa", "Load_N", "Strain_percent"]].copy()
            sample_df = sample_df.dropna()
            sample_df = sample_df[sample_df["Strain"] > 0]
            sample_df = sample_df[sample_df["Stress_MPa"] > 0]
            sample_df = sample_df.sort_values("Strain").drop_duplicates(subset=["Strain"], keep="first")

            if len(sample_df) < 5:
                continue

            max_strains.append(sample_df["Strain"].max())
            valid_curves.append((sample_df["Strain"].values, sample_df["Stress_MPa"].values))

            sample_df["Condition"] = condition_name
            sample_df["Sample"] = sheet
            sample_df["Area_mm2"] = area_mm2
            raw_export_rows.append(sample_df)

        if not valid_curves:
            raise ValueError(f"No valid samples found for {condition_name}")

        max_common_strain = min(max_strains)
        strain_grid = np.linspace(0.0, max_common_strain, 1000)

        interp_stresses = []
        for s_arr, t_arr in valid_curves:
            stress_interp = np.interp(strain_grid, s_arr, t_arr)
            interp_stresses.append(stress_interp)

        stress_matrix = np.vstack(interp_stresses)
        mean_stress = np.mean(stress_matrix, axis=0)
        std_stress = np.std(stress_matrix, axis=0)

        if SMOOTH_MEAN_CURVE:
            mean_stress = smooth(mean_stress, window_frac=SMOOTH_WINDOW_FRAC)

        df_processed = pd.DataFrame({
            "Strain": strain_grid,
            "FinalMean": mean_stress,
            "StdStress": std_stress
        })

        processed_csv = os.path.join("aligned_by_drop_results", f"{condition_name}_processed_aligned.csv")
        raw_csv = os.path.join("aligned_by_drop_results", f"{condition_name}_raw_converted.csv")

        df_processed.to_csv(processed_csv, index=False)
        pd.concat(raw_export_rows, ignore_index=True).to_csv(raw_csv, index=False)

        plot_path = os.path.join("comparison_plots", f"{condition_name}_experimental_mean_styled.png")

        plot_experimental_style(
            valid_curves=valid_curves,
            strain_grid=strain_grid,
            mean_stress=mean_stress,
            condition_key=condition_name,
            display_label=LABEL_MAP.get(condition_name, condition_name),
            out_path=plot_path,
            x_max=XMAX_MAP.get(condition_name, None),
            y_max=YMAX_DEFAULT
        )

        metadata = {
            "condition": condition_name,
            "processed_csv": processed_csv,
            "raw_csv": raw_csv,
            "plot_path": plot_path,
            "mean_area_mm2": float(np.mean(used_areas)),
            "areas_mm2": used_areas,
            "n_samples": len(valid_curves),
        }

        print(f"Prepared: {condition_name}")
        print(f"  Mean curve CSV: {processed_csv}")
        print(f"  Raw converted CSV: {raw_csv}")
        print(f"  Styled plot: {plot_path}")
        print(f"  Samples used: {len(valid_curves)}")
        print(f"  Mean area: {metadata['mean_area_mm2']:.4f} mm²")

        return True, metadata

    except Exception as e:
        print(f"Error while processing {condition_name}: {e}")
        return False, {}

# ---------------------------
# Run all conditions
# ---------------------------
exp_meta = {}

for condition_name, xlsx_file, area_csv in EXP_INPUT_SETS:
    ready, meta = prepare_exp_data_from_xlsx(condition_name, xlsx_file, area_csv)
    if ready:
        exp_meta[condition_name] = meta

# ---------------------------
# Combined comparison plot
# ---------------------------
if exp_meta:
    fig, ax = plt.subplots(figsize=(8.5, 5.8), dpi=DPI)
    ax.set_facecolor(PANEL_BG)

    for condition_name, _, _ in EXP_INPUT_SETS:
        if condition_name not in exp_meta:
            continue

        df = pd.read_csv(exp_meta[condition_name]["processed_csv"])
        palette = PALETTE_MAP.get(condition_name, {
            "replicate": "#A0A0A0",
            "mean": "#303030"
        })

        ax.plot(
            df["Strain"] * 100.0,
            df["FinalMean"],
            linewidth=2.2,
            color=palette["mean"],
            label=condition_name
        )

    style_axis(ax)
    ax.set_xlabel(r'$\varepsilon$ (%)', fontsize=15, fontweight='bold')
    ax.set_ylabel(r'$\sigma$ (MPa)', fontsize=15, fontweight='bold')

    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(250))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))

    ax.set_xlim(0, 5.5)
    ax.set_ylim(0, YMAX_DEFAULT)
    ax.legend(frameon=False, fontsize=11)
    ax.set_title("Experimental Mean Stress-Strain Curves (3 wt.% Ni)", fontsize=15, pad=10)

    plt.tight_layout()
    combined_plot = os.path.join("comparison_plots", "all_experimental_mean_curves_styled.png")
    plt.savefig(combined_plot, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"\nSaved combined plot: {combined_plot}")
    print("Done.")
else:
    print("No experimental datasets were processed.")


# In[21]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.ticker import FuncFormatter, AutoMinorLocator, MultipleLocator

# ============================================================
# Experimental tensile processing only (no FEM)
# White background, no panel letters, plain black labels,
# distinct color palette for each condition,
# x-axis percent ticks at 1% intervals,
# minor ticks styled like the reference figure,
# automatic axis limits so curve ends stay visible
# ============================================================

# ---------------------------
# Setup output directories
# ---------------------------
os.makedirs("aligned_by_drop_results", exist_ok=True)
os.makedirs("comparison_plots", exist_ok=True)

# ---------------------------
# Input files
# ---------------------------
EXP_INPUT_SETS = [
    ("AS",     "AS.xlsx",            "AREA_3%Ni_AS _EN.csv"),
    ("SR400",  "SR400_1h .xlsx",     "AREA_3%Ni_SR400_EN.csv"),
    ("SR450",  "SR450_1h.xlsx",      "AREA_3%Ni_SR450 _EN.csv"),
    ("SR500",  "SR500_1h.xlsx",      "AREA_3%Ni_SR500 _EN.csv"),
    ("SR550",  "SR550_1h.xlsx",      "AREA_3%Ni_SR550_EN.csv"),
    ("SA1100", "SA1100_15min.xlsx",  "AREA_3%Ni_SA1100 _EN.csv"),
]

# ---------------------------
# Plot settings
# ---------------------------
DPI = 300
SMOOTH_MEAN_CURVE = True
SMOOTH_WINDOW_FRAC = 0.01
PANEL_BG = "white"

PALETTE_MAP = {
    "AS": {
        "replicate": "#8FB3D9",
        "mean": "#1E4E8C"
    },
    "SR400": {
        "replicate": "#E8B26D",
        "mean": "#B86B00"
    },
    "SR450": {
        "replicate": "#8FCB9B",
        "mean": "#2F7D4A"
    },
    "SR500": {
        "replicate": "#D99AA4",
        "mean": "#A63D57"
    },
    "SR550": {
        "replicate": "#B7A1D6",
        "mean": "#6C3FA1"
    },
    "SA1100": {
        "replicate": "#7FC7C2",
        "mean": "#007A78"
    },
}

LABEL_MAP = {
    "AS": "AS",
    "SR400": "SR400",
    "SR450": "SR450",
    "SR500": "SR500",
    "SR550": "SR550",
    "SA1100": "SA1100",
}

# optional baseline minimum y ceiling
YMAX_DEFAULT = 1450

# ---------------------------
# Helper functions
# ---------------------------
def normalize_text(s):
    return str(s).strip().lower().replace(" ", "")

def smooth(y, window_frac=0.01):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 5:
        return y

    win = max(5, int(window_frac * n))
    if win % 2 == 0:
        win += 1
    if win >= n:
        win = n - 1 if n % 2 == 0 else n
    if win < 5:
        return y

    return savgol_filter(y, window_length=win, polyorder=2, mode="interp")

def percent_formatter(x, pos):
    return f"{x:.0f}%"

def style_axis(ax):
    ax.set_facecolor(PANEL_BG)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    # Major ticks
    ax.tick_params(
        axis="both", which="major",
        direction="out", length=7, width=1.6,
        colors="black", labelsize=11
    )

    # Minor ticks
    ax.tick_params(
        axis="both", which="minor",
        direction="out", length=4, width=1.0,
        colors="black"
    )

    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.8, alpha=0.65)
    ax.grid(True, which="minor", color="#eeeeee", linewidth=0.5, alpha=0.35)

def add_condition_label(ax, label):
    ax.text(
        0.46, 0.16, label,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=13, fontweight="bold", color="black"
    )

def load_area_values(area_csv):
    values = []
    with open(area_csv, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            txt = line.strip().replace(",", ".")
            try:
                values.append(float(txt))
            except ValueError:
                continue
    return values

def detect_columns(df):
    strain_col = None
    load_col = None

    for col in df.columns:
        c = normalize_text(col)

        if strain_col is None and "strain" in c:
            strain_col = col

        if load_col is None:
            if "force" in c or "load" in c:
                load_col = col
            elif "stress" in c and "n" in c:
                load_col = col

    if strain_col is None:
        raise ValueError(f"Could not detect strain column in columns: {list(df.columns)}")
    if load_col is None:
        raise ValueError(f"Could not detect load column in columns: {list(df.columns)}")

    return strain_col, load_col

def clean_sample_curve(df, strain_col, load_col):
    out = df[[strain_col, load_col]].copy()
    out.columns = ["Strain_percent", "Load_N"]

    out["Strain_percent"] = pd.to_numeric(out["Strain_percent"], errors="coerce")
    out["Load_N"] = pd.to_numeric(out["Load_N"], errors="coerce")
    out = out.dropna(subset=["Strain_percent", "Load_N"])

    out = out[out["Strain_percent"] > 0]
    out = out[out["Load_N"] > 0]

    out = out.sort_values("Strain_percent").reset_index(drop=True)
    out = out.groupby("Strain_percent", as_index=False)["Load_N"].mean()

    return out

def plot_experimental_style(valid_curves, strain_grid, mean_stress, condition_key, display_label, out_path,
                            x_max=None, y_max=None):
    fig, ax = plt.subplots(figsize=(4.8, 3.8), dpi=DPI)

    palette = PALETTE_MAP.get(condition_key, {
        "replicate": "#A0A0A0",
        "mean": "#303030"
    })

    replicate_color = palette["replicate"]
    mean_color = palette["mean"]

    # Plot replicate curves
    for s_arr, t_arr in valid_curves:
        ax.plot(
            s_arr * 100.0,
            t_arr,
            color=replicate_color,
            linewidth=1.0,
            alpha=0.75
        )

    # Plot mean curve
    ax.plot(
        strain_grid * 100.0,
        mean_stress,
        color=mean_color,
        linewidth=2.4,
        alpha=1.0
    )

    style_axis(ax)

    ax.set_xlabel(r'$\varepsilon$ (%)', fontsize=15, fontweight='bold')
    ax.set_ylabel(r'$\sigma$ (MPa)', fontsize=15, fontweight='bold')

    # Tick layout to match the reference style
    ax.xaxis.set_major_locator(MultipleLocator(1.0))   # 0%, 1%, 2%, ...
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))    # 4 minor ticks between majors
    ax.yaxis.set_major_locator(MultipleLocator(250))   # 0, 250, 500, ...
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))    # 1 minor tick between majors
    ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))

    # Limits from all replicate curves so the ends are visible
    all_x = np.concatenate([s_arr * 100.0 for s_arr, _ in valid_curves])
    all_y = np.concatenate([t_arr for _, t_arr in valid_curves])

    x_data_max = np.nanmax(all_x)
    y_data_max = np.nanmax(all_y)

    if x_max is None:
        x_max = x_data_max * 1.04
    else:
        x_max = max(x_max, x_data_max * 1.02)

    if y_max is None:
        y_max = max(YMAX_DEFAULT, y_data_max * 1.04)
    else:
        y_max = max(y_max, y_data_max * 1.02)

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)

    add_condition_label(ax, display_label)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", transparent=False)
    plt.close()

# ---------------------------
# Main experimental preparation
# ---------------------------
def prepare_exp_data_from_xlsx(condition_name, xlsx_file, area_csv):
    try:
        if not os.path.exists(xlsx_file):
            print(f"Error: missing XLSX file: {xlsx_file}")
            return False, {}

        if not os.path.exists(area_csv):
            print(f"Error: missing area CSV file: {area_csv}")
            return False, {}

        xl = pd.ExcelFile(xlsx_file)
        sheets = xl.sheet_names
        area_values = load_area_values(area_csv)

        if len(area_values) < len(sheets):
            raise ValueError(
                f"{condition_name}: only {len(area_values)} area values found, "
                f"but {len(sheets)} sheets exist."
            )

        if len(area_values) != len(sheets):
            print(
                f"Warning: {condition_name} has {len(sheets)} sheets but {len(area_values)} area values. "
                f"Using the first {len(sheets)} areas."
            )

        valid_curves = []
        max_strains = []
        used_areas = []
        raw_export_rows = []

        for i, sheet in enumerate(sheets):
            df = pd.read_excel(xlsx_file, sheet_name=sheet)
            strain_col, load_col = detect_columns(df)
            sample_df = clean_sample_curve(df, strain_col, load_col)

            if len(sample_df) < 5:
                continue

            area_mm2 = area_values[i]
            used_areas.append(area_mm2)

            sample_df["Stress_MPa"] = sample_df["Load_N"] / area_mm2
            sample_df["Strain"] = sample_df["Strain_percent"] / 100.0

            sample_df = sample_df[["Strain", "Stress_MPa", "Load_N", "Strain_percent"]].copy()
            sample_df = sample_df.dropna()
            sample_df = sample_df[sample_df["Strain"] > 0]
            sample_df = sample_df[sample_df["Stress_MPa"] > 0]
            sample_df = sample_df.sort_values("Strain").drop_duplicates(subset=["Strain"], keep="first")

            if len(sample_df) < 5:
                continue

            max_strains.append(sample_df["Strain"].max())
            valid_curves.append((sample_df["Strain"].values, sample_df["Stress_MPa"].values))

            sample_df["Condition"] = condition_name
            sample_df["Sample"] = sheet
            sample_df["Area_mm2"] = area_mm2
            raw_export_rows.append(sample_df)

        if not valid_curves:
            raise ValueError(f"No valid samples found for {condition_name}")

        # Mean curve only over common strain range shared by all valid replicates
        max_common_strain = min(max_strains)
        strain_grid = np.linspace(0.0, max_common_strain, 1000)

        interp_stresses = []
        for s_arr, t_arr in valid_curves:
            stress_interp = np.interp(strain_grid, s_arr, t_arr)
            interp_stresses.append(stress_interp)

        stress_matrix = np.vstack(interp_stresses)
        mean_stress = np.mean(stress_matrix, axis=0)
        std_stress = np.std(stress_matrix, axis=0)

        if SMOOTH_MEAN_CURVE:
            mean_stress = smooth(mean_stress, window_frac=SMOOTH_WINDOW_FRAC)

        df_processed = pd.DataFrame({
            "Strain": strain_grid,
            "FinalMean": mean_stress,
            "StdStress": std_stress
        })

        processed_csv = os.path.join("aligned_by_drop_results", f"{condition_name}_processed_aligned.csv")
        raw_csv = os.path.join("aligned_by_drop_results", f"{condition_name}_raw_converted.csv")

        df_processed.to_csv(processed_csv, index=False)
        pd.concat(raw_export_rows, ignore_index=True).to_csv(raw_csv, index=False)

        plot_path = os.path.join("comparison_plots", f"{condition_name}_experimental_mean_styled.png")

        plot_experimental_style(
            valid_curves=valid_curves,
            strain_grid=strain_grid,
            mean_stress=mean_stress,
            condition_key=condition_name,
            display_label=LABEL_MAP.get(condition_name, condition_name),
            out_path=plot_path,
            x_max=None,
            y_max=None
        )

        metadata = {
            "condition": condition_name,
            "processed_csv": processed_csv,
            "raw_csv": raw_csv,
            "plot_path": plot_path,
            "mean_area_mm2": float(np.mean(used_areas)),
            "areas_mm2": used_areas,
            "n_samples": len(valid_curves),
        }

        print(f"Prepared: {condition_name}")
        print(f"  Mean curve CSV: {processed_csv}")
        print(f"  Raw converted CSV: {raw_csv}")
        print(f"  Styled plot: {plot_path}")
        print(f"  Samples used: {len(valid_curves)}")
        print(f"  Mean area: {metadata['mean_area_mm2']:.4f} mm²")

        return True, metadata

    except Exception as e:
        print(f"Error while processing {condition_name}: {e}")
        return False, {}

# ---------------------------
# Run all conditions
# ---------------------------
exp_meta = {}

for condition_name, xlsx_file, area_csv in EXP_INPUT_SETS:
    ready, meta = prepare_exp_data_from_xlsx(condition_name, xlsx_file, area_csv)
    if ready:
        exp_meta[condition_name] = meta

# ---------------------------
# Combined comparison plot
# ---------------------------
if exp_meta:
    fig, ax = plt.subplots(figsize=(8.5, 5.8), dpi=DPI)
    ax.set_facecolor(PANEL_BG)

    xmax_all = 0.0
    ymax_all = 0.0

    for condition_name, _, _ in EXP_INPUT_SETS:
        if condition_name not in exp_meta:
            continue

        df = pd.read_csv(exp_meta[condition_name]["processed_csv"])
        palette = PALETTE_MAP.get(condition_name, {
            "replicate": "#A0A0A0",
            "mean": "#303030"
        })

        xvals = df["Strain"].to_numpy() * 100.0
        yvals = df["FinalMean"].to_numpy()

        xmax_all = max(xmax_all, np.nanmax(xvals))
        ymax_all = max(ymax_all, np.nanmax(yvals))

        ax.plot(
            xvals,
            yvals,
            linewidth=2.2,
            color=palette["mean"],
            label=condition_name
        )

    style_axis(ax)
    ax.set_xlabel(r'$\varepsilon$ (%)', fontsize=15, fontweight='bold')
    ax.set_ylabel(r'$\sigma$ (MPa)', fontsize=15, fontweight='bold')

    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(250))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))

    ax.set_xlim(0, xmax_all * 1.05)
    ax.set_ylim(0, max(YMAX_DEFAULT, ymax_all * 1.08))
    ax.legend(frameon=False, fontsize=11)
    ax.set_title("Experimental Mean Stress-Strain Curves (3 wt.% Ni)", fontsize=15, pad=10)

    plt.tight_layout()
    combined_plot = os.path.join("comparison_plots", "all_experimental_mean_curves_styled.png")
    plt.savefig(combined_plot, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"\nSaved combined plot: {combined_plot}")
    print("Done.")
else:
    print("No experimental datasets were processed.")


# In[22]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.ticker import FuncFormatter, AutoMinorLocator, MultipleLocator

# ============================================================
# Experimental tensile processing only (no FEM)
# White background, no panel letters, plain black labels,
# distinct color palette for each condition,
# dynamic x-axis ticks for long-strain conditions,
# minor ticks styled like the reference figure,
# automatic axis limits so curve ends stay visible
# ============================================================

# ---------------------------
# Setup output directories
# ---------------------------
os.makedirs("aligned_by_drop_results", exist_ok=True)
os.makedirs("comparison_plots", exist_ok=True)

# ---------------------------
# Input files
# ---------------------------
EXP_INPUT_SETS = [
    ("AS",     "AS.xlsx",            "AREA_3%Ni_AS _EN.csv"),
    ("SR400",  "SR400_1h .xlsx",     "AREA_3%Ni_SR400_EN.csv"),
    ("SR450",  "SR450_1h.xlsx",      "AREA_3%Ni_SR450 _EN.csv"),
    ("SR500",  "SR500_1h.xlsx",      "AREA_3%Ni_SR500 _EN.csv"),
    ("SR550",  "SR550_1h.xlsx",      "AREA_3%Ni_SR550_EN.csv"),
    ("SA1100", "SA1100_15min.xlsx",  "AREA_3%Ni_SA1100 _EN.csv"),
]

# ---------------------------
# Plot settings
# ---------------------------
DPI = 300
SMOOTH_MEAN_CURVE = True
SMOOTH_WINDOW_FRAC = 0.01
PANEL_BG = "white"

PALETTE_MAP = {
    "AS": {
        "replicate": "#8FB3D9",
        "mean": "#1E4E8C"
    },
    "SR400": {
        "replicate": "#E8B26D",
        "mean": "#B86B00"
    },
    "SR450": {
        "replicate": "#8FCB9B",
        "mean": "#2F7D4A"
    },
    "SR500": {
        "replicate": "#D99AA4",
        "mean": "#A63D57"
    },
    "SR550": {
        "replicate": "#B7A1D6",
        "mean": "#6C3FA1"
    },
    "SA1100": {
        "replicate": "#7FC7C2",
        "mean": "#007A78"
    },
}

LABEL_MAP = {
    "AS": "AS",
    "SR400": "SR400",
    "SR450": "SR450",
    "SR500": "SR500",
    "SR550": "SR550",
    "SA1100": "SA1100",
}

YMAX_DEFAULT = 1450

# ---------------------------
# Helper functions
# ---------------------------
def normalize_text(s):
    return str(s).strip().lower().replace(" ", "")

def smooth(y, window_frac=0.01):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 5:
        return y

    win = max(5, int(window_frac * n))
    if win % 2 == 0:
        win += 1
    if win >= n:
        win = n - 1 if n % 2 == 0 else n
    if win < 5:
        return y

    return savgol_filter(y, window_length=win, polyorder=2, mode="interp")

def percent_formatter(x, pos):
    return f"{x:.0f}%"

def style_axis(ax):
    ax.set_facecolor(PANEL_BG)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    # Major ticks
    ax.tick_params(
        axis="both", which="major",
        direction="out", length=7, width=1.6,
        colors="black", labelsize=11
    )

    # Minor ticks
    ax.tick_params(
        axis="both", which="minor",
        direction="out", length=4, width=1.0,
        colors="black"
    )

    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.8, alpha=0.65)
    ax.grid(True, which="minor", color="#eeeeee", linewidth=0.5, alpha=0.35)

def apply_strain_tick_style(ax, x_max):
    """
    Dynamic major tick spacing based on total strain range.
    Short ranges keep 1% major ticks; long ranges use wider spacing.
    """
    if x_max <= 12:
        major_step = 1.0
    elif x_max <= 20:
        major_step = 2.0
    else:
        major_step = 5.0

    ax.xaxis.set_major_locator(MultipleLocator(major_step))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(250))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))

def add_condition_label(ax, label):
    ax.text(
        0.46, 0.16, label,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=13, fontweight="bold", color="black"
    )

def load_area_values(area_csv):
    values = []
    with open(area_csv, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            txt = line.strip().replace(",", ".")
            try:
                values.append(float(txt))
            except ValueError:
                continue
    return values

def detect_columns(df):
    strain_col = None
    load_col = None

    for col in df.columns:
        c = normalize_text(col)

        if strain_col is None and "strain" in c:
            strain_col = col

        if load_col is None:
            if "force" in c or "load" in c:
                load_col = col
            elif "stress" in c and "n" in c:
                load_col = col

    if strain_col is None:
        raise ValueError(f"Could not detect strain column in columns: {list(df.columns)}")
    if load_col is None:
        raise ValueError(f"Could not detect load column in columns: {list(df.columns)}")

    return strain_col, load_col

def clean_sample_curve(df, strain_col, load_col):
    out = df[[strain_col, load_col]].copy()
    out.columns = ["Strain_percent", "Load_N"]

    out["Strain_percent"] = pd.to_numeric(out["Strain_percent"], errors="coerce")
    out["Load_N"] = pd.to_numeric(out["Load_N"], errors="coerce")
    out = out.dropna(subset=["Strain_percent", "Load_N"])

    out = out[out["Strain_percent"] > 0]
    out = out[out["Load_N"] > 0]

    out = out.sort_values("Strain_percent").reset_index(drop=True)
    out = out.groupby("Strain_percent", as_index=False)["Load_N"].mean()

    return out

def plot_experimental_style(valid_curves, strain_grid, mean_stress, condition_key, display_label, out_path,
                            x_max=None, y_max=None):
    fig, ax = plt.subplots(figsize=(4.8, 3.8), dpi=DPI)

    palette = PALETTE_MAP.get(condition_key, {
        "replicate": "#A0A0A0",
        "mean": "#303030"
    })

    replicate_color = palette["replicate"]
    mean_color = palette["mean"]

    # Plot replicate curves
    for s_arr, t_arr in valid_curves:
        ax.plot(
            s_arr * 100.0,
            t_arr,
            color=replicate_color,
            linewidth=1.0,
            alpha=0.75
        )

    # Plot mean curve
    ax.plot(
        strain_grid * 100.0,
        mean_stress,
        color=mean_color,
        linewidth=2.4,
        alpha=1.0
    )

    style_axis(ax)

    ax.set_xlabel(r'$\varepsilon$ (%)', fontsize=15, fontweight='bold')
    ax.set_ylabel(r'$\sigma$ (MPa)', fontsize=15, fontweight='bold')

    # Limits from all replicate curves so the ends are visible
    all_x = np.concatenate([s_arr * 100.0 for s_arr, _ in valid_curves])
    all_y = np.concatenate([t_arr for _, t_arr in valid_curves])

    x_data_max = np.nanmax(all_x)
    y_data_max = np.nanmax(all_y)

    if x_max is None:
        x_max = x_data_max * 1.04
    else:
        x_max = max(x_max, x_data_max * 1.02)

    if y_max is None:
        y_max = max(YMAX_DEFAULT, y_data_max * 1.04)
    else:
        y_max = max(y_max, y_data_max * 1.02)

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)

    # Apply dynamic tick style after limits are known
    apply_strain_tick_style(ax, x_max)

    add_condition_label(ax, display_label)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", transparent=False)
    plt.close()

# ---------------------------
# Main experimental preparation
# ---------------------------
def prepare_exp_data_from_xlsx(condition_name, xlsx_file, area_csv):
    try:
        if not os.path.exists(xlsx_file):
            print(f"Error: missing XLSX file: {xlsx_file}")
            return False, {}

        if not os.path.exists(area_csv):
            print(f"Error: missing area CSV file: {area_csv}")
            return False, {}

        xl = pd.ExcelFile(xlsx_file)
        sheets = xl.sheet_names
        area_values = load_area_values(area_csv)

        if len(area_values) < len(sheets):
            raise ValueError(
                f"{condition_name}: only {len(area_values)} area values found, "
                f"but {len(sheets)} sheets exist."
            )

        if len(area_values) != len(sheets):
            print(
                f"Warning: {condition_name} has {len(sheets)} sheets but {len(area_values)} area values. "
                f"Using the first {len(sheets)} areas."
            )

        valid_curves = []
        max_strains = []
        used_areas = []
        raw_export_rows = []

        for i, sheet in enumerate(sheets):
            df = pd.read_excel(xlsx_file, sheet_name=sheet)
            strain_col, load_col = detect_columns(df)
            sample_df = clean_sample_curve(df, strain_col, load_col)

            if len(sample_df) < 5:
                continue

            area_mm2 = area_values[i]
            used_areas.append(area_mm2)

            sample_df["Stress_MPa"] = sample_df["Load_N"] / area_mm2
            sample_df["Strain"] = sample_df["Strain_percent"] / 100.0

            sample_df = sample_df[["Strain", "Stress_MPa", "Load_N", "Strain_percent"]].copy()
            sample_df = sample_df.dropna()
            sample_df = sample_df[sample_df["Strain"] > 0]
            sample_df = sample_df[sample_df["Stress_MPa"] > 0]
            sample_df = sample_df.sort_values("Strain").drop_duplicates(subset=["Strain"], keep="first")

            if len(sample_df) < 5:
                continue

            max_strains.append(sample_df["Strain"].max())
            valid_curves.append((sample_df["Strain"].values, sample_df["Stress_MPa"].values))

            sample_df["Condition"] = condition_name
            sample_df["Sample"] = sheet
            sample_df["Area_mm2"] = area_mm2
            raw_export_rows.append(sample_df)

        if not valid_curves:
            raise ValueError(f"No valid samples found for {condition_name}")

        # Mean curve only over common strain range shared by all valid replicates
        max_common_strain = min(max_strains)
        strain_grid = np.linspace(0.0, max_common_strain, 1000)

        interp_stresses = []
        for s_arr, t_arr in valid_curves:
            stress_interp = np.interp(strain_grid, s_arr, t_arr)
            interp_stresses.append(stress_interp)

        stress_matrix = np.vstack(interp_stresses)
        mean_stress = np.mean(stress_matrix, axis=0)
        std_stress = np.std(stress_matrix, axis=0)

        if SMOOTH_MEAN_CURVE:
            mean_stress = smooth(mean_stress, window_frac=SMOOTH_WINDOW_FRAC)

        df_processed = pd.DataFrame({
            "Strain": strain_grid,
            "FinalMean": mean_stress,
            "StdStress": std_stress
        })

        processed_csv = os.path.join("aligned_by_drop_results", f"{condition_name}_processed_aligned.csv")
        raw_csv = os.path.join("aligned_by_drop_results", f"{condition_name}_raw_converted.csv")

        df_processed.to_csv(processed_csv, index=False)
        pd.concat(raw_export_rows, ignore_index=True).to_csv(raw_csv, index=False)

        plot_path = os.path.join("comparison_plots", f"{condition_name}_experimental_mean_styled.png")

        plot_experimental_style(
            valid_curves=valid_curves,
            strain_grid=strain_grid,
            mean_stress=mean_stress,
            condition_key=condition_name,
            display_label=LABEL_MAP.get(condition_name, condition_name),
            out_path=plot_path,
            x_max=None,
            y_max=None
        )

        metadata = {
            "condition": condition_name,
            "processed_csv": processed_csv,
            "raw_csv": raw_csv,
            "plot_path": plot_path,
            "mean_area_mm2": float(np.mean(used_areas)),
            "areas_mm2": used_areas,
            "n_samples": len(valid_curves),
        }

        print(f"Prepared: {condition_name}")
        print(f"  Mean curve CSV: {processed_csv}")
        print(f"  Raw converted CSV: {raw_csv}")
        print(f"  Styled plot: {plot_path}")
        print(f"  Samples used: {len(valid_curves)}")
        print(f"  Mean area: {metadata['mean_area_mm2']:.4f} mm²")

        return True, metadata

    except Exception as e:
        print(f"Error while processing {condition_name}: {e}")
        return False, {}

# ---------------------------
# Run all conditions
# ---------------------------
exp_meta = {}

for condition_name, xlsx_file, area_csv in EXP_INPUT_SETS:
    ready, meta = prepare_exp_data_from_xlsx(condition_name, xlsx_file, area_csv)
    if ready:
        exp_meta[condition_name] = meta

# ---------------------------
# Combined comparison plot
# ---------------------------
if exp_meta:
    fig, ax = plt.subplots(figsize=(8.5, 5.8), dpi=DPI)
    ax.set_facecolor(PANEL_BG)

    xmax_all = 0.0
    ymax_all = 0.0

    for condition_name, _, _ in EXP_INPUT_SETS:
        if condition_name not in exp_meta:
            continue

        df = pd.read_csv(exp_meta[condition_name]["processed_csv"])
        palette = PALETTE_MAP.get(condition_name, {
            "replicate": "#A0A0A0",
            "mean": "#303030"
        })

        xvals = df["Strain"].to_numpy() * 100.0
        yvals = df["FinalMean"].to_numpy()

        xmax_all = max(xmax_all, np.nanmax(xvals))
        ymax_all = max(ymax_all, np.nanmax(yvals))

        ax.plot(
            xvals,
            yvals,
            linewidth=2.2,
            color=palette["mean"],
            label=condition_name
        )

    style_axis(ax)
    ax.set_xlabel(r'$\varepsilon$ (%)', fontsize=15, fontweight='bold')
    ax.set_ylabel(r'$\sigma$ (MPa)', fontsize=15, fontweight='bold')

    ax.set_xlim(0, xmax_all * 1.05)
    ax.set_ylim(0, max(YMAX_DEFAULT, ymax_all * 1.08))

    apply_strain_tick_style(ax, xmax_all * 1.05)

    ax.legend(frameon=False, fontsize=11)
    ax.set_title("Experimental Mean Stress-Strain Curves (3 wt.% Ni)", fontsize=15, pad=10)

    plt.tight_layout()
    combined_plot = os.path.join("comparison_plots", "all_experimental_mean_curves_styled.png")
    plt.savefig(combined_plot, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"\nSaved combined plot: {combined_plot}")
    print("Done.")
else:
    print("No experimental datasets were processed.")


# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.ticker import FuncFormatter, AutoMinorLocator, MultipleLocator

# ============================================================
# Experimental tensile processing only (no FEM)
# White background, no panel letters, plain black labels,
# distinct color palette for each condition,
# dynamic x-axis ticks for long-strain conditions,
# minor ticks styled like the reference figure,
# automatic axis limits so curve ends stay visible,
# bold axis tick values
# ============================================================

# ---------------------------
# Setup output directories
# ---------------------------
os.makedirs("aligned_by_drop_results", exist_ok=True)
os.makedirs("comparison_plots", exist_ok=True)

# ---------------------------
# Input files
# ---------------------------
EXP_INPUT_SETS = [
    ("AS",     "AS.xlsx",            "AREA_3%Ni_AS _EN.csv"),
    ("SR400",  "SR400_1h .xlsx",     "AREA_3%Ni_SR400_EN.csv"),
    ("SR450",  "SR450_1h.xlsx",      "AREA_3%Ni_SR450 _EN.csv"),
    ("SR500",  "SR500_1h.xlsx",      "AREA_3%Ni_SR500 _EN.csv"),
    ("SR550",  "SR550_1h.xlsx",      "AREA_3%Ni_SR550_EN.csv"),
    ("SA1100", "SA1100_15min.xlsx",  "AREA_3%Ni_SA1100 _EN.csv"),
]

# ---------------------------
# Plot settings
# ---------------------------
DPI = 300
SMOOTH_MEAN_CURVE = True
SMOOTH_WINDOW_FRAC = 0.01
PANEL_BG = "white"

PALETTE_MAP = {
    "AS": {
        "replicate": "#8FB3D9",
        "mean": "#1E4E8C"
    },
    "SR400": {
        "replicate": "#E8B26D",
        "mean": "#B86B00"
    },
    "SR450": {
        "replicate": "#8FCB9B",
        "mean": "#2F7D4A"
    },
    "SR500": {
        "replicate": "#D99AA4",
        "mean": "#A63D57"
    },
    "SR550": {
        "replicate": "#B7A1D6",
        "mean": "#6C3FA1"
    },
    "SA1100": {
        "replicate": "#7FC7C2",
        "mean": "#007A78"
    },
}

LABEL_MAP = {
    "AS": "AS",
    "SR400": "SR400",
    "SR450": "SR450",
    "SR500": "SR500",
    "SR550": "SR550",
    "SA1100": "SA1100",
}

YMAX_DEFAULT = 1450

# ---------------------------
# Helper functions
# ---------------------------
def normalize_text(s):
    return str(s).strip().lower().replace(" ", "")

def smooth(y, window_frac=0.01):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 5:
        return y

    win = max(5, int(window_frac * n))
    if win % 2 == 0:
        win += 1
    if win >= n:
        win = n - 1 if n % 2 == 0 else n
    if win < 5:
        return y

    return savgol_filter(y, window_length=win, polyorder=2, mode="interp")

def percent_formatter(x, pos):
    return f"{x:.0f}%"

def style_axis(ax):
    ax.set_facecolor(PANEL_BG)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    # Major ticks
    ax.tick_params(
        axis="both", which="major",
        direction="out", length=7, width=1.6,
        colors="black", labelsize=11
    )

    # Minor ticks
    ax.tick_params(
        axis="both", which="minor",
        direction="out", length=4, width=1.0,
        colors="black"
    )

    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.8, alpha=0.65)
    ax.grid(True, which="minor", color="#eeeeee", linewidth=0.5, alpha=0.35)

def apply_strain_tick_style(ax, x_max):
    """
    Dynamic major tick spacing based on total strain range.
    Short ranges keep 1% major ticks; long ranges use wider spacing.
    """
    if x_max <= 12:
        major_step = 1.0
    elif x_max <= 20:
        major_step = 2.0
    else:
        major_step = 5.0

    ax.xaxis.set_major_locator(MultipleLocator(major_step))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(250))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))

def make_ticklabels_bold(ax, size=12):
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")
        lbl.set_fontsize(size)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
        lbl.set_fontsize(size)

def add_condition_label(ax, label):
    ax.text(
        0.46, 0.16, label,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=13, fontweight="bold", color="black"
    )

def load_area_values(area_csv):
    values = []
    with open(area_csv, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            txt = line.strip().replace(",", ".")
            try:
                values.append(float(txt))
            except ValueError:
                continue
    return values

def detect_columns(df):
    strain_col = None
    load_col = None

    for col in df.columns:
        c = normalize_text(col)

        if strain_col is None and "strain" in c:
            strain_col = col

        if load_col is None:
            if "force" in c or "load" in c:
                load_col = col
            elif "stress" in c and "n" in c:
                load_col = col

    if strain_col is None:
        raise ValueError(f"Could not detect strain column in columns: {list(df.columns)}")
    if load_col is None:
        raise ValueError(f"Could not detect load column in columns: {list(df.columns)}")

    return strain_col, load_col

def clean_sample_curve(df, strain_col, load_col):
    out = df[[strain_col, load_col]].copy()
    out.columns = ["Strain_percent", "Load_N"]

    out["Strain_percent"] = pd.to_numeric(out["Strain_percent"], errors="coerce")
    out["Load_N"] = pd.to_numeric(out["Load_N"], errors="coerce")
    out = out.dropna(subset=["Strain_percent", "Load_N"])

    out = out[out["Strain_percent"] > 0]
    out = out[out["Load_N"] > 0]

    out = out.sort_values("Strain_percent").reset_index(drop=True)
    out = out.groupby("Strain_percent", as_index=False)["Load_N"].mean()

    return out

def plot_experimental_style(valid_curves, strain_grid, mean_stress, condition_key, display_label, out_path,
                            x_max=None, y_max=None):
    fig, ax = plt.subplots(figsize=(4.8, 3.8), dpi=DPI)

    palette = PALETTE_MAP.get(condition_key, {
        "replicate": "#A0A0A0",
        "mean": "#303030"
    })

    replicate_color = palette["replicate"]
    mean_color = palette["mean"]

    # Plot replicate curves
    for s_arr, t_arr in valid_curves:
        ax.plot(
            s_arr * 100.0,
            t_arr,
            color=replicate_color,
            linewidth=1.0,
            alpha=0.75
        )

    # Plot mean curve
    ax.plot(
        strain_grid * 100.0,
        mean_stress,
        color=mean_color,
        linewidth=2.4,
        alpha=1.0
    )

    style_axis(ax)

    ax.set_xlabel(r'$\varepsilon$ (%)', fontsize=15, fontweight='bold')
    ax.set_ylabel(r'$\sigma$ (MPa)', fontsize=15, fontweight='bold')

    # Limits from all replicate curves so the ends are visible
    all_x = np.concatenate([s_arr * 100.0 for s_arr, _ in valid_curves])
    all_y = np.concatenate([t_arr for _, t_arr in valid_curves])

    x_data_max = np.nanmax(all_x)
    y_data_max = np.nanmax(all_y)

    if x_max is None:
        x_max = x_data_max * 1.04
    else:
        x_max = max(x_max, x_data_max * 1.02)

    if y_max is None:
        y_max = max(YMAX_DEFAULT, y_data_max * 1.04)
    else:
        y_max = max(y_max, y_data_max * 1.02)

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)

    # Apply dynamic tick style after limits are known
    apply_strain_tick_style(ax, x_max)
    make_ticklabels_bold(ax, size=12)

    add_condition_label(ax, display_label)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", transparent=False)
    plt.close()

# ---------------------------
# Main experimental preparation
# ---------------------------
def prepare_exp_data_from_xlsx(condition_name, xlsx_file, area_csv):
    try:
        if not os.path.exists(xlsx_file):
            print(f"Error: missing XLSX file: {xlsx_file}")
            return False, {}

        if not os.path.exists(area_csv):
            print(f"Error: missing area CSV file: {area_csv}")
            return False, {}

        xl = pd.ExcelFile(xlsx_file)
        sheets = xl.sheet_names
        area_values = load_area_values(area_csv)

        if len(area_values) < len(sheets):
            raise ValueError(
                f"{condition_name}: only {len(area_values)} area values found, "
                f"but {len(sheets)} sheets exist."
            )

        if len(area_values) != len(sheets):
            print(
                f"Warning: {condition_name} has {len(sheets)} sheets but {len(area_values)} area values. "
                f"Using the first {len(sheets)} areas."
            )

        valid_curves = []
        max_strains = []
        used_areas = []
        raw_export_rows = []

        for i, sheet in enumerate(sheets):
            df = pd.read_excel(xlsx_file, sheet_name=sheet)
            strain_col, load_col = detect_columns(df)
            sample_df = clean_sample_curve(df, strain_col, load_col)

            if len(sample_df) < 5:
                continue

            area_mm2 = area_values[i]
            used_areas.append(area_mm2)

            sample_df["Stress_MPa"] = sample_df["Load_N"] / area_mm2
            sample_df["Strain"] = sample_df["Strain_percent"] / 100.0

            sample_df = sample_df[["Strain", "Stress_MPa", "Load_N", "Strain_percent"]].copy()
            sample_df = sample_df.dropna()
            sample_df = sample_df[sample_df["Strain"] > 0]
            sample_df = sample_df[sample_df["Stress_MPa"] > 0]
            sample_df = sample_df.sort_values("Strain").drop_duplicates(subset=["Strain"], keep="first")

            if len(sample_df) < 5:
                continue

            max_strains.append(sample_df["Strain"].max())
            valid_curves.append((sample_df["Strain"].values, sample_df["Stress_MPa"].values))

            sample_df["Condition"] = condition_name
            sample_df["Sample"] = sheet
            sample_df["Area_mm2"] = area_mm2
            raw_export_rows.append(sample_df)

        if not valid_curves:
            raise ValueError(f"No valid samples found for {condition_name}")

        # Mean curve only over common strain range shared by all valid replicates
        max_common_strain = min(max_strains)
        strain_grid = np.linspace(0.0, max_common_strain, 1000)

        interp_stresses = []
        for s_arr, t_arr in valid_curves:
            stress_interp = np.interp(strain_grid, s_arr, t_arr)
            interp_stresses.append(stress_interp)

        stress_matrix = np.vstack(interp_stresses)
        mean_stress = np.mean(stress_matrix, axis=0)
        std_stress = np.std(stress_matrix, axis=0)

        if SMOOTH_MEAN_CURVE:
            mean_stress = smooth(mean_stress, window_frac=SMOOTH_WINDOW_FRAC)

        df_processed = pd.DataFrame({
            "Strain": strain_grid,
            "FinalMean": mean_stress,
            "StdStress": std_stress
        })

        processed_csv = os.path.join("aligned_by_drop_results", f"{condition_name}_processed_aligned.csv")
        raw_csv = os.path.join("aligned_by_drop_results", f"{condition_name}_raw_converted.csv")

        df_processed.to_csv(processed_csv, index=False)
        pd.concat(raw_export_rows, ignore_index=True).to_csv(raw_csv, index=False)

        plot_path = os.path.join("comparison_plots", f"{condition_name}_experimental_mean_styled.png")

        plot_experimental_style(
            valid_curves=valid_curves,
            strain_grid=strain_grid,
            mean_stress=mean_stress,
            condition_key=condition_name,
            display_label=LABEL_MAP.get(condition_name, condition_name),
            out_path=plot_path,
            x_max=None,
            y_max=None
        )

        metadata = {
            "condition": condition_name,
            "processed_csv": processed_csv,
            "raw_csv": raw_csv,
            "plot_path": plot_path,
            "mean_area_mm2": float(np.mean(used_areas)),
            "areas_mm2": used_areas,
            "n_samples": len(valid_curves),
        }

        print(f"Prepared: {condition_name}")
        print(f"  Mean curve CSV: {processed_csv}")
        print(f"  Raw converted CSV: {raw_csv}")
        print(f"  Styled plot: {plot_path}")
        print(f"  Samples used: {len(valid_curves)}")
        print(f"  Mean area: {metadata['mean_area_mm2']:.4f} mm²")

        return True, metadata

    except Exception as e:
        print(f"Error while processing {condition_name}: {e}")
        return False, {}

# ---------------------------
# Run all conditions
# ---------------------------
exp_meta = {}

for condition_name, xlsx_file, area_csv in EXP_INPUT_SETS:
    ready, meta = prepare_exp_data_from_xlsx(condition_name, xlsx_file, area_csv)
    if ready:
        exp_meta[condition_name] = meta

# ---------------------------
# Combined comparison plot
# ---------------------------
if exp_meta:
    fig, ax = plt.subplots(figsize=(8.5, 5.8), dpi=DPI)
    ax.set_facecolor(PANEL_BG)

    xmax_all = 0.0
    ymax_all = 0.0

    for condition_name, _, _ in EXP_INPUT_SETS:
        if condition_name not in exp_meta:
            continue

        df = pd.read_csv(exp_meta[condition_name]["processed_csv"])
        palette = PALETTE_MAP.get(condition_name, {
            "replicate": "#A0A0A0",
            "mean": "#303030"
        })

        xvals = df["Strain"].to_numpy() * 100.0
        yvals = df["FinalMean"].to_numpy()

        xmax_all = max(xmax_all, np.nanmax(xvals))
        ymax_all = max(ymax_all, np.nanmax(yvals))

        ax.plot(
            xvals,
            yvals,
            linewidth=2.2,
            color=palette["mean"],
            label=condition_name
        )

    style_axis(ax)
    ax.set_xlabel(r'$\varepsilon$ (%)', fontsize=15, fontweight='bold')
    ax.set_ylabel(r'$\sigma$ (MPa)', fontsize=15, fontweight='bold')

    ax.set_xlim(0, xmax_all * 1.05)
    ax.set_ylim(0, max(YMAX_DEFAULT, ymax_all * 1.08))

    apply_strain_tick_style(ax, xmax_all * 1.05)
    make_ticklabels_bold(ax, size=12)

    ax.legend(frameon=False, fontsize=11)
    ax.set_title("Experimental Mean Stress-Strain Curves (3 wt.% Ni)", fontsize=15, pad=10)

    plt.tight_layout()
    combined_plot = os.path.join("comparison_plots", "all_experimental_mean_curves_styled.png")
    plt.savefig(combined_plot, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"\nSaved combined plot: {combined_plot}")
    print("Done.")
else:
    print("No experimental datasets were processed.")


# In[ ]:




