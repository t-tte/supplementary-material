import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

sns.set_style("whitegrid")


def plot_ate_by_covariate(
    data, Xi_name, W_name, y_name, x_tick_labs=None, y_lab=None, title=None
):
    """
    Plots the average treatment effect per the value of a covariate.

    Args
    ---
    Xi_name : str
        The name of the covariate.
    W_name : str
        The name of the treatment.
    y_name : str
        The name of the outcome.

    Returns
    -------
    fig : figure
        The plotted figure.
    """
    data_grouped = data.groupby([Xi_name, W_name])[y_name].mean()
    ctr_mean = data_grouped.unstack()[0].values * 100
    trt_mean = data_grouped.unstack()[1].values * 100
    x_locs = np.arange(data[Xi_name].nunique())
    bar_width = 0.4
    fig, ax = plt.subplots()
    bars_ctr = ax.bar(
        x_locs - bar_width / 2, ctr_mean, bar_width, label="Control", color="lightgrey"
    )
    bars_trt = ax.bar(
        x_locs + bar_width / 2, trt_mean, bar_width, label="Treatment", color="grey"
    )

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    if x_tick_labs is not None:
        ax.set_xticks(x_locs)
        ax.set_xticklabels(x_tick_labs)

    if y_lab is not None:
        ax.set_ylabel(y_lab)

    if title is not None:
        ax.set_title(title)

    ax.legend()

    def label_bar_height(bars):
        for bar in bars:
            bar_height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar_height,
                str(round(bar_height)) + "%",
                ha="center",
                va="bottom",
            )

    label_bar_height(bars_ctr)
    label_bar_height(bars_trt)

    return fig


def plot_bucket_stats(data_neg, data_pos, X_names, X_labels=None, title=None):
    """
    Plots the average treatment effect per the value of a covariate.

    Args
    ---
    data_neg : DataFrame
        Data from the 1st subgroup.
    data_pos : DataFrame
        Data from the 2nd subgroup.
    X_names : lst
        List of covariate names (str).

    Returns
    -------
    fig : figure
        The plotted figure.
    """
    x_locs = np.arange(len(X_names))
    bar_width = 0.4

    fig, ax = plt.subplots()
    bars_neg = ax.bar(
        x_locs - bar_width / 2,
        data_neg.loc["mean", X_names].values * 100,
        bar_width,
        label="Most negative predicted effect",
        color="grey",
    )
    bars_pos = ax.bar(
        x_locs + bar_width / 2,
        data_pos.loc["mean", X_names].values * 100,
        bar_width,
        label="Most positive predicted effect",
        color="lightgrey",
    )

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    if X_labels is not None:
        ax.set_xticks(x_locs)
        ax.set_xticklabels(X_labels)

    ax.set_ylabel("Proportion of population")

    if title is not None:
        ax.set_title(title)

    def label_bar_height(bars):
        for bar in bars:
            bar_height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar_height,
                str(round(bar_height)) + "%",
                ha="center",
                va="bottom",
            )

    label_bar_height(bars_neg)
    label_bar_height(bars_pos)

    ax.legend()

    return fig
