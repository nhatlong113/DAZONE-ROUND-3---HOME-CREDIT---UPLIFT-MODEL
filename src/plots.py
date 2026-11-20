import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from .analysis import CARD_ARM, CASH_ARM
from .metrics import qini_curve, qini_metrics

sns.set_theme(style="whitegrid")


def annotate_bars(ax, fmt="{:.2f}%", horizontal=False, fontsize=9):
    for p in ax.patches:
        v = p.get_width() if horizontal else p.get_height()
        if np.isnan(v) or v == 0:
            continue
        xy = (v, p.get_y() + p.get_height() / 2) if horizontal else (p.get_x() + p.get_width() / 2, v)
        ax.annotate(fmt.format(v), xy, ha="left" if horizontal else "center", va="center" if horizontal else "bottom", fontsize=fontsize)


def matched_conversion_bar(table, title, xlabel, horizontal=False, save=None):
    long = table[[CASH_ARM, CARD_ARM]].reset_index(names="segment").melt(id_vars="segment", var_name="arm", value_name="rate")
    fig, ax = plt.subplots(figsize=(12, 7) if horizontal else (10, 6))
    if horizontal:
        sns.barplot(data=long, y="segment", x="rate", hue="arm", orient="h", ax=ax)
        ax.set_xlabel("Conversion Rate (%)")
        ax.set_ylabel(xlabel)
    else:
        sns.barplot(data=long, x="segment", y="rate", hue="arm", ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Conversion Rate (%)")
    annotate_bars(ax, horizontal=horizontal, fontsize=8)
    ax.set_title(title)
    ax.legend(title="Treatment → Outcome")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()


def qini_plot(y, w_event, score, title, save=None):
    x, q = qini_curve(y, w_event, score)
    ate = q[-1]
    baseline = ate * x
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#EAEAF2")
    ax.plot(x, q, color="#1f77b4", linewidth=2.0, label=f"Model  |  Qini={qini_metrics(y, w_event, score)['qini_norm']:.6f}", zorder=3)
    ax.plot([0, 1], [0, ate], linestyle="--", color="gray", linewidth=1.5, label="Random baseline", zorder=2)
    ax.fill_between(x, q, baseline, where=q >= baseline, interpolate=True, alpha=0.18, color="#4C78A8")
    ax.fill_between(x, q, baseline, where=q < baseline, interpolate=True, alpha=0.12, color="#D62728")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Fraction of Population Targeted")
    ax.set_ylabel("Incremental Gain (rate)")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    plt.show()


def importance_bar(fi, top=15, save=None):
    top_fi = fi.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_fi["feature"], top_fi["feature_importance"], color="#4C78A8")
    ax.set_xlabel("feature_importance = mean(LIME, SHAP, Permutation)")
    ax.set_title(f"Top {top} Feature Importance — Ensemble 5 models")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()
