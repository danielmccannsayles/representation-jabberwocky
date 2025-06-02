"""Utilities - sentence viewer & lat scan. Changed from original code"""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize


def reader_sentence_view(
    words: list[str],
    scores: np.ndarray,
    selection_style: Literal["positive", "negative"] = "negative",
    invert_colors: bool = False,
):
    """Sentence visualizer"""
    assert len(words) == len(scores), "Mismatched lengths"

    # Simple data processing
    scores = np.array(scores)
    scores = (scores - np.mean(scores)) / np.std(scores)  # Standardize
    scores = np.clip(scores, -2, 2)  # Clip outliers
    norm = Normalize(vmin=scores.min(), vmax=scores.max())

    if selection_style == "negative":
        scores = np.clip(scores, -np.inf, 0)
    else:
        scores = np.clip(scores, 0, np.inf)

    # Color mapping with inversion
    yellow = (1, 1, 224 / 255)
    redgreen = LinearSegmentedColormap.from_list("rg", ["r", yellow, "g"], N=256)
    greenred = LinearSegmentedColormap.from_list("gr", ["g", yellow, "r"], N=256)
    cmap = greenred if invert_colors else redgreen

    fig, ax = plt.subplots(figsize=(12.8, 8), dpi=200)

    # Set limits for the x and y axes
    max_line_width = 1000
    ax.set_xlim(0, max_line_width)
    ax.set_ylim(0, 10)  # Dunno what exactly this is doing but it really needs it lol

    # Remove ticks and labels from the axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Starting position of the words in the plot
    x_start, y_start = 1, 9
    y_pad = 0.3

    # Initialize positions and maximum line width
    x, y = x_start, y_start

    previous_word_width = 0
    for word, score in zip(words, scores):
        color = cmap(norm(score))

        # Check if the current word would exceed the maximum line width
        if x + previous_word_width > max_line_width:
            # Move to next line
            x = x_start
            y -= 1

        # Compute the width of the current word
        text = ax.text(x, y, word, fontsize=13)
        previous_word_width = (
            text.get_window_extent(fig.canvas.get_renderer())
            .transformed(ax.transData.inverted())
            .width
        )

        # Add the text with background color
        text = ax.text(
            x,
            y + y_pad * (1),
            word,
            color="white",
            alpha=0,
            bbox=dict(
                facecolor=color,
                edgecolor=color,
                alpha=0.8,
                boxstyle=f"round,pad=0",
                linewidth=0,
            ),
            fontsize=13,
        )

        # Update the x position for the next word
        x += previous_word_width + 0.1


def reader_lat_scan(
    scores,
    starting_layer=1,
    bound=2.3,
    threshold=0,
    reverse_colors=False,
):
    """scores is 2d nd.arr"""

    # Clip & standardize scores
    # scores[np.abs(scores) < threshold] = 0
    # standardized_scores = scores.clip(-bound, bound)
    x_len = len(scores)
    y_len = len(scores[0])

    plot_scores = scores.T
    if reverse_colors:
        plot_scores = -plot_scores

    _, ax = plt.subplots(figsize=(5, 4), dpi=200)
    sns.heatmap(
        scores.T,
        cmap="coolwarm",
        linewidth=0.5,
        annot=False,
        fmt=".3f",
        vmin=-bound,
        vmax=bound,
    )
    ax.tick_params(axis="y", rotation=0)

    ax.set_xlabel("Token Position")
    ax.set_ylabel("Layer")

    cbar = ax.collections[0].colorbar
    cbar.set_label("Concept Activation")

    # x label appear every 5 ticks
    ax.set_xticks(np.arange(0, x_len, 5)[1:])
    ax.set_xticklabels(np.arange(0, x_len, 5)[1:])
    ax.tick_params(axis="x", rotation=0)

    ax.set_yticks(np.arange(0, y_len, 5)[1:])
    ax.set_yticklabels(np.arange(starting_layer, y_len + starting_layer, 5)[1:])
    ax.set_title("LAT Neural Activity")
    plt.show()
