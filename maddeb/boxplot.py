"""box plot function."""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.cbook import _reshape_2D


# Function adapted from matplotlib.cbook
def my_boxplot_stats(
    X, whis=1.5, bootstrap=None, labels=None, autorange=False, percents=[25, 75]
):
    """Return statistics computed for boxplot.

    Parameters
    ----------
    X: list/array
        data
    whis: float, valid string, or list of percentiles.
        limits of the whisker
    bootstrap: int
        number of iterations
    labels: list
        labels of the columns
    autorange: bool
        automatically determine the range for whiskers
    percents: list
        percentile range

    Returns
    -------
    bxpstats: list
        list of box plot stats

    """

    def _bootstrap_median(data, N=5000):
        # determine 95% confidence intervals of the median
        M = len(data)
        percentiles = [2.5, 97.5]

        bs_index = np.random.randint(M, size=(N, M))
        bsData = data[bs_index]
        estimate = np.median(bsData, axis=1, overwrite_input=True)

        CI = np.percentile(estimate, percentiles)
        return CI

    def _compute_conf_interval(data, med, iqr, bootstrap):
        if bootstrap is not None:
            # Do a bootstrap estimate of notch locations.
            # get conf. intervals around median
            CI = _bootstrap_median(data, N=bootstrap)
            notch_min = CI[0]
            notch_max = CI[1]
        else:

            N = len(data)
            notch_min = med - 1.57 * iqr / np.sqrt(N)
            notch_max = med + 1.57 * iqr / np.sqrt(N)

        return notch_min, notch_max

    # output is a list of dicts
    bxpstats = []

    # convert X to a list of lists
    X = _reshape_2D(X, "X")

    ncols = len(X)
    if labels is None:
        labels = itertools.repeat(None)
    elif len(labels) != ncols:
        raise ValueError("Dimensions of labels and X must be compatible")

    input_whis = whis
    for ii, (x, label) in enumerate(zip(X, labels)):

        # empty dict
        stats = {}
        if label is not None:
            stats["label"] = label

        # restore whis to the input values in case it got changed in the loop
        whis = input_whis

        # note tricksyness, append up here and then mutate below
        bxpstats.append(stats)

        # if empty, bail
        if len(x) == 0:
            stats["fliers"] = np.array([])
            stats["mean"] = np.nan
            stats["med"] = np.nan
            stats["q1"] = np.nan
            stats["q3"] = np.nan
            stats["cilo"] = np.nan
            stats["cihi"] = np.nan
            stats["whislo"] = np.nan
            stats["whishi"] = np.nan
            stats["med"] = np.nan
            continue

        # up-convert to an array, just to be safe
        x = np.asarray(x)

        # arithmetic mean
        stats["mean"] = np.mean(x)

        # median
        med = np.percentile(x, 50)

        # Altered line
        q1, q3 = np.percentile(x, (percents[0], percents[1]))

        # interquartile range
        stats["iqr"] = q3 - q1
        if stats["iqr"] == 0 and autorange:
            whis = "range"

        # conf. interval around median
        stats["cilo"], stats["cihi"] = _compute_conf_interval(
            x, med, stats["iqr"], bootstrap
        )

        # lowest/highest non-outliers
        if np.isscalar(whis):
            if np.isreal(whis):
                loval = q1 - whis * stats["iqr"]
                hival = q3 + whis * stats["iqr"]
            elif whis in ["range", "limit", "limits", "min/max"]:
                loval = np.min(x)
                hival = np.max(x)
            else:
                raise ValueError(
                    "whis must be a float, valid string, or list " "of percentiles"
                )
        else:
            loval = np.percentile(x, whis[0])
            hival = np.percentile(x, whis[1])

        # get high extreme
        wiskhi = np.compress(x <= hival, x)
        if len(wiskhi) == 0 or np.max(wiskhi) < q3:
            stats["whishi"] = q3
        else:
            stats["whishi"] = np.max(wiskhi)

        # get low extreme
        wisklo = np.compress(x >= loval, x)
        if len(wisklo) == 0 or np.min(wisklo) > q1:
            stats["whislo"] = q1
        else:
            stats["whislo"] = np.min(wisklo)

        # compute a single array of outliers
        stats["fliers"] = np.hstack(
            [np.compress(x < stats["whislo"], x), np.compress(x > stats["whishi"], x)]
        )

        # add in the remaining stats
        stats["q1"], stats["med"], stats["q3"] = q1, med, q3

    return bxpstats


def boxplot_func(
    df_in,
    x,
    y,
    z,
    xlim,
    ylim,
    x_scale,
    legend,
    x_label,
    y_label,
    y_label_hist,
    x_ticks=None,
    whis=[5, 95],
    percents=[25, 75],
    errors=None,
    legend_remove=False,
    legend_location="upper left",
    palette=["#3498db", "#e74c3c"],
    nbins=11,
):
    """Return boxplot figure, median and standard deviation.

    Parameters
    ----------
    df_in:
        input data
    x:
        labels for x
    y:
        labels for y
    z:
        labels for z
    xlim:
        x axis limits for plots
    ylim:
        y axis limits for plots
    zlim:
        z axis limits for plots
    x_scale:
        choice 'log' or not
    legend:
        legend to display
    x_label: str
        x axis label
    y_label:
        x axis label
    y_label_hist: str
        y label for the histogram
    x_ticks: list
        x_ticks for the histogram
    whis: int/float
        percentile limit for the whiskers
    percents: int/float
        percentile limits for the box
    errors:
        errors to drop if necessary
    legend_remove:
        boolean to remove the legend
    legend_location:
        location of the legend.
    palette:
        color palette
    nbins:
        number of bins to split data

    Returns
    -------
    fig:
        the final plot
    median: list
        medians computed by the function my_boxplot_stats
    q1: list
        starting percentile of the range computed in my_boxplot_stats
    q3: list
        ending percentile of the range as computed by my_boxplot_stats
    whislo: list
        lower value fo the whisker
    whishi: list
        higher value of the whisker

    """
    median = []
    q1 = []
    q3 = []
    whislo = []
    whishi = []

    import matplotlib as mpl

    mpl.rcdefaults()

    # Drop error if necessary
    if errors is not None:
        df_plot = df_in.drop(errors)
    else:
        df_plot = df_in

    df_plot = df_plot[(df_plot[x]>xlim[0]) & (df_plot[x]<xlim[1])]

    # Drop NaN in dataframe
    df_plot = df_plot.dropna()

    # Define bins
    if x_scale == "log":
        x_bins = np.geomspace(xlim[0], xlim[1], nbins + 1)
    else:
        x_bins = np.linspace(xlim[0], xlim[1], nbins + 1)

    x_bins[0] -= 1e-5
    x_bins[-1] += 1e-5

    idx = np.digitize(df_plot[x], x_bins)

    # Initialize figure
    fig, axes = plt.subplots(
        2, 1, figsize=(6, 4), gridspec_kw={"height_ratios": [10, 2]}
    )

    # Second plot: boxplot generated with seaborn split as a function of the parameter
    ax = axes[0]

    exp = np.unique(df_plot[z])
    N_exp = len(exp)
    handles = []

    for ik, key in enumerate(exp):
        stats = {}

        # Compute and save statistics
        for i in range(1, len(x_bins)):
            stats[i] = my_boxplot_stats(
                df_plot[y][np.logical_and(idx == i, df_plot[z] == key)].values,
                whis=whis,
                percents=percents,
            )[0]
            median.append(stats[i]["med"])
            q1.append(stats[i]["q1"])
            q3.append(stats[i]["q3"])
            whislo.append(stats[i]["whislo"])
            whishi.append(stats[i]["whishi"])
        # Plot boxplots from our computed statistics
        bp = ax.bxp(
            [stats[i] for i in range(1, len(x_bins))],
            positions=np.arange(len(x_bins) - 1)
            + 0.5
            + 0.9 * (ik - (N_exp - 1.0) / 2.0) / N_exp,
            widths=0.6 / len(np.unique(df_plot[z])),
            showfliers=False,
            patch_artist=True,
            boxprops={
                "facecolor": (*mpl.colors.to_rgba(palette[ik])[:3], 0.25),
                "edgecolor": palette[ik],
            },
        )

        handles.append(bp["boxes"][0])

        # Colour the lines in the boxplot blue
        for element in bp.keys():
            if element != "boxes":
                plt.setp(bp[element], color=palette[ik])

    if not legend_remove:
        ax.legend(
            handles,
            legend,
            frameon=False,
            loc=legend_location,
            borderpad=0.1,
            fontsize=10,
        )

    ax.set_xticks([])
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_ylim(ylim[0], ylim[1])

    # Top plot: distribution of the parameter
    if x_scale == "log":
        sns.histplot(df_plot[x], kde=False, log_scale=True, ax=axes[1], color="0.5")
        # axes[0].set_xlim(np.log10(xlim[0]), np.log10(xlim[1]))
    else:
        sns.histplot(df_plot[x], kde=False, ax=axes[1], color="0.5")
    axes[1].set_xlim(xlim[0], xlim[1])

    axes[1].set_yticks([])
    axes[1].set_ylabel(y_label_hist, fontsize=12)

    if x_ticks is not None:
        axes[1].set_xticks(x_ticks)
        axes[1].set_xticklabels(x_ticks) 

    axes[1].set_xlabel(x_label, fontsize=12)
    axes[1].xaxis.tick_bottom()

    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)

    return fig, median, q1, q3, whislo, whishi
