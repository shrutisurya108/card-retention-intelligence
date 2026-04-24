"""
eda.py
------
Phase 2 — Exploratory Data Analysis

Responsibilities:
  1. Load ingested data from SQLite (customers table)
  2. Generate 10 publication-quality plots saved to outputs/figures/
  3. Print statistical summaries to log
  4. All plots are reused by the Streamlit dashboard in Phase 5

Design decisions:
  - All figures saved as PNG (300 DPI) for crisp dashboard rendering
  - Consistent colour palette: churned = #E74C3C (red), retained = #2ECC71 (green)
  - Every plot function is independent — can be called individually or all at once
  - Matplotlib backend set to 'Agg' (non-interactive) for server/CI compatibility
"""

import matplotlib
matplotlib.use("Agg")  # Must be set before importing pyplot — works on all OS/servers

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import create_engine

from src.logger import get_logger

log = get_logger("eda")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
DB_PATH     = ROOT_DIR / "database" / "churn.db"
DB_URL      = f"sqlite:///{DB_PATH}"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"

# ── Colour constants ──────────────────────────────────────────────────────────
CLR_CHURN   = "#E74C3C"
CLR_RETAIN  = "#2ECC71"
CLR_PALETTE = [CLR_RETAIN, CLR_CHURN]
CLR_HEATMAP = "coolwarm"
PLOT_DPI    = 300

sns.set_theme(style="whitegrid", font_scale=1.1)


# ── Data loader ───────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load the customers table from SQLite into a DataFrame."""
    log.info("Loading data from SQLite for EDA...")
    engine = create_engine(DB_URL, echo=False)
    df = pd.read_sql("SELECT * FROM customers", con=engine)
    engine.dispose()
    log.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


# ── Save helper ───────────────────────────────────────────────────────────────

def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save a matplotlib figure to outputs/figures/ at 300 DPI."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved plot → {path.relative_to(ROOT_DIR)}")


# ── Plot 1: Churn class distribution ─────────────────────────────────────────

def plot_churn_distribution(df: pd.DataFrame) -> None:
    counts = df["churn"].value_counts().sort_index()
    labels = ["Retained", "Churned"]
    total  = len(df)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts.values, color=CLR_PALETTE, edgecolor="white", width=0.5)

    for bar, count in zip(bars, counts.values):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 80,
            f"{count:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.set_title("Customer Churn Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Number of Customers")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    sns.despine()
    save_figure(fig, "churn_distribution.png")


# ── Plot 2: Customer age distribution ────────────────────────────────────────

def plot_age_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for churn_val, label, colour in zip([0, 1], ["Retained", "Churned"], CLR_PALETTE):
        subset = df[df["churn"] == churn_val]["Customer_Age"]
        ax.hist(subset, bins=20, alpha=0.4, color=colour, density=True,
                label=f"{label} (n={len(subset):,})")
        subset.plot.kde(ax=ax, color=colour, linewidth=2)

    ax.set_title("Customer Age Distribution by Churn Status",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Customer Age")
    ax.set_ylabel("Density")
    ax.legend()
    sns.despine()
    save_figure(fig, "age_distribution.png")


# ── Plot 3: Churn rate by age group ──────────────────────────────────────────

def plot_churn_by_age_group(df: pd.DataFrame) -> None:
    df = df.copy()
    df["age_group"] = pd.cut(
        df["Customer_Age"],
        bins=[25, 35, 45, 55, 65, 75],
        labels=["26-35", "36-45", "46-55", "56-65", "66-75"]
    )
    churn_by_age = df.groupby("age_group", observed=True)["churn"].mean() * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        churn_by_age.index.astype(str),
        churn_by_age.values,
        color=CLR_CHURN, alpha=0.8, edgecolor="white"
    )
    for bar, val in zip(bars, churn_by_age.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.axhline(df["churn"].mean() * 100, color="navy", linestyle="--",
               linewidth=1.5, label=f"Overall avg ({df['churn'].mean()*100:.1f}%)")
    ax.set_title("Churn Rate by Age Group", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Churn Rate (%)")
    ax.legend()
    sns.despine()
    save_figure(fig, "churn_by_age.png")


# ── Plot 4: Credit limit distribution ────────────────────────────────────────

def plot_credit_limit_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for churn_val, label, colour in zip([0, 1], ["Retained", "Churned"], CLR_PALETTE):
        subset = df[df["churn"] == churn_val]["Credit_Limit"]
        axes[0].hist(subset, bins=30, alpha=0.5, color=colour, label=label, density=True)
    axes[0].set_title("Credit Limit — Raw", fontweight="bold")
    axes[0].set_xlabel("Credit Limit ($)")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    for churn_val, label, colour in zip([0, 1], ["Retained", "Churned"], CLR_PALETTE):
        subset = np.log1p(df[df["churn"] == churn_val]["Credit_Limit"])
        axes[1].hist(subset, bins=30, alpha=0.5, color=colour, label=label, density=True)
    axes[1].set_title("Credit Limit — Log Transformed", fontweight="bold")
    axes[1].set_xlabel("log(1 + Credit Limit)")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    fig.suptitle("Credit Limit Distribution (Raw vs Log Transform)",
                 fontsize=14, fontweight="bold", y=1.02)
    sns.despine()
    save_figure(fig, "credit_limit_dist.png")


# ── Plot 5: Transaction count vs churn ───────────────────────────────────────

def plot_transaction_count_vs_churn(df: pd.DataFrame) -> None:
    """
    FutureWarning fix: assign x variable to hue and set legend=False
    instead of passing palette without hue.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = df[["Total_Trans_Ct", "churn"]].copy()
    plot_df["Status"] = plot_df["churn"].map({0: "Retained", 1: "Churned"})

    palette = {"Retained": CLR_RETAIN, "Churned": CLR_CHURN}

    sns.boxplot(
        data=plot_df, x="Status", y="Total_Trans_Ct",
        hue="Status", palette=palette,
        width=0.4, ax=ax, order=["Retained", "Churned"],
        legend=False
    )
    sns.stripplot(
        data=plot_df.sample(min(500, len(plot_df)), random_state=42),
        x="Status", y="Total_Trans_Ct",
        hue="Status", palette=palette,
        alpha=0.3, size=3, ax=ax, order=["Retained", "Churned"],
        legend=False
    )

    ax.set_title("Transaction Count by Churn Status",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Customer Status")
    ax.set_ylabel("Total Transaction Count (12 months)")
    sns.despine()
    save_figure(fig, "trans_count_vs_churn.png")


# ── Plot 6: Correlation heatmap ───────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include="number").drop(columns=["CLIENTNUM"],
                                                          errors="ignore")
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap=CLR_HEATMAP, center=0, square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8}, ax=ax, annot_kws={"size": 8}
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    save_figure(fig, "correlation_heatmap.png")


# ── Plot 7: Churn rate by categorical features ───────────────────────────────

def plot_churn_by_category(df: pd.DataFrame) -> None:
    cat_cols = ["Income_Category", "Education_Level", "Card_Category", "Marital_Status"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, col in zip(axes, cat_cols):
        churn_rates = (
            df.groupby(col)["churn"].mean() * 100
        ).sort_values(ascending=False)

        bars = ax.bar(range(len(churn_rates)), churn_rates.values,
                      color=CLR_CHURN, alpha=0.8, edgecolor="white")
        for bar, val in zip(bars, churn_rates.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(churn_rates)))
        ax.set_xticklabels(churn_rates.index, rotation=30, ha="right", fontsize=9)
        ax.set_title(f"Churn Rate by {col.replace('_', ' ')}", fontweight="bold")
        ax.set_ylabel("Churn Rate (%)")
        ax.axhline(df["churn"].mean() * 100, color="navy",
                   linestyle="--", linewidth=1.2, alpha=0.7)

    fig.suptitle("Churn Rate by Customer Segment",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_figure(fig, "churn_by_category.png")


# ── Plot 8: Numeric feature distributions ────────────────────────────────────

def plot_numeric_distributions(df: pd.DataFrame) -> None:
    numeric_cols = [
        "Customer_Age", "Months_on_book", "Credit_Limit",
        "Total_Trans_Amt", "Total_Trans_Ct", "Avg_Utilization_Ratio",
        "Total_Revolving_Bal", "Total_Relationship_Count"
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for ax, col in zip(axes, numeric_cols):
        for churn_val, label, colour in zip([0, 1], ["Retained", "Churned"], CLR_PALETTE):
            subset = df[df["churn"] == churn_val][col]
            ax.hist(subset, bins=25, alpha=0.5, color=colour, label=label, density=True)
        ax.set_title(col.replace("_", " "), fontsize=9, fontweight="bold")
        ax.set_ylabel("Density")
        ax.tick_params(labelsize=8)

    axes[0].legend(fontsize=8)
    fig.suptitle("Numeric Feature Distributions by Churn Status",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_figure(fig, "numeric_distributions.png")


# ── Plot 9: Utilisation ratio vs churn ───────────────────────────────────────

def plot_utilisation_vs_churn(df: pd.DataFrame) -> None:
    """FutureWarning fix: assign hue explicitly."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = df[["Avg_Utilization_Ratio", "churn"]].copy()
    plot_df["Status"] = plot_df["churn"].map({0: "Retained", 1: "Churned"})

    sns.violinplot(
        data=plot_df, x="Status", y="Avg_Utilization_Ratio",
        hue="Status",
        palette={"Retained": CLR_RETAIN, "Churned": CLR_CHURN},
        inner="quartile", ax=ax, order=["Retained", "Churned"],
        legend=False
    )
    ax.set_title("Credit Utilisation Ratio by Churn Status",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Customer Status")
    ax.set_ylabel("Average Utilisation Ratio")
    sns.despine()
    save_figure(fig, "utilisation_vs_churn.png")


# ── Plot 10: Contacts count vs churn ─────────────────────────────────────────

def plot_contacts_vs_churn(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df = df[["Contacts_Count_12_mon", "churn"]].copy()
    plot_df["Status"] = plot_df["churn"].map({0: "Retained", 1: "Churned"})

    sns.countplot(
        data=plot_df, x="Contacts_Count_12_mon", hue="Status",
        palette={"Retained": CLR_RETAIN, "Churned": CLR_CHURN},
        ax=ax
    )
    ax.set_title("Bank Contact Frequency by Churn Status",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Number of Contacts in Last 12 Months")
    ax.set_ylabel("Number of Customers")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(title="Status")
    sns.despine()
    save_figure(fig, "contacts_vs_churn.png")


# ── Statistical summary ───────────────────────────────────────────────────────

def log_statistical_summary(df: pd.DataFrame) -> None:
    log.info("── Statistical Summary ─────────────────────────────────")
    log.info(f"  Total customers   : {len(df):,}")
    log.info(f"  Churned           : {df['churn'].sum():,} ({df['churn'].mean()*100:.2f}%)")
    log.info(f"  Retained          : {(~df['churn'].astype(bool)).sum():,}")
    log.info(f"  Avg age           : {df['Customer_Age'].mean():.1f} years")
    log.info(f"  Avg credit limit  : ${df['Credit_Limit'].mean():,.0f}")
    log.info(f"  Avg transactions  : {df['Total_Trans_Ct'].mean():.1f} / year")
    log.info(f"  Avg utilisation   : {df['Avg_Utilization_Ratio'].mean():.3f}")
    log.info("  Churn rate by card type:")
    for card, rate in (df.groupby("Card_Category")["churn"].mean() * 100).items():
        log.info(f"    {card:<15}: {rate:.1f}%")
    log.info("────────────────────────────────────────────────────────")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_eda() -> None:
    """Execute all EDA plots and log statistical summary."""
    log.info("═══════════════════════════════════════════════════════")
    log.info("  PHASE 2a — EXPLORATORY DATA ANALYSIS               ")
    log.info("═══════════════════════════════════════════════════════")

    df = load_data()
    log_statistical_summary(df)

    log.info("Generating plots...")
    plot_churn_distribution(df)
    plot_age_distribution(df)
    plot_churn_by_age_group(df)
    plot_credit_limit_distribution(df)
    plot_transaction_count_vs_churn(df)
    plot_correlation_heatmap(df)
    plot_churn_by_category(df)
    plot_numeric_distributions(df)
    plot_utilisation_vs_churn(df)
    plot_contacts_vs_churn(df)

    log.info("All 10 plots saved to outputs/figures/")
    log.info("Phase 2a complete — EDA successful ✓")


if __name__ == "__main__":
    run_eda()