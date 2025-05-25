### For now just copied exact from rep-engineering

import random
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from transformers import PreTrainedTokenizer


def honesty_function_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    user_tag: str = "",
    assistant_tag: str = "",
    seed: int = 0,
) -> (list, list):
    """
    Processes data to create training and testing datasets based on honesty.

    Args:
    - data_path (str): Path to the CSV containing the data.
    - tokenizer (PreTrainedTokenizer): Tokenizer to tokenize statements.
    - user_tag (str): Instruction template.
    - assistant_tag (str): Instruction template user tag.
    - seed (int): Random seed for reproducibility.

    Returns:
    - Tuple containing train and test data.
    """

    # Setting the seed for reproducibility
    random.seed(seed)

    # Load the data
    df = pd.read_csv(data_path)
    true_statements = df[df["label"] == 1]["statement"].values.tolist()
    false_statements = df[df["label"] == 0]["statement"].values.tolist()

    template_str = "Pretend you're {type} person making statements about the world."
    honest_statements = []
    untruthful_statements = []

    # Process statements
    for statement in true_statements:
        tokens = tokenizer.tokenize(statement)

        for idx in range(1, len(tokens) - 5):
            truncated_tokens = tokens[:idx]
            truncated_statement = tokenizer.convert_tokens_to_string(truncated_tokens)

            honest_statements.append(
                f"{user_tag} {template_str.format(type='an honest')} {assistant_tag} "
                + truncated_statement
            )
            untruthful_statements.append(
                f"{user_tag} {template_str.format(type='an untruthful')} {assistant_tag} "
                + truncated_statement
            )

    # Create training data
    ntrain = 512
    combined_data = [
        [honest, untruthful]
        for honest, untruthful in zip(honest_statements, untruthful_statements)
    ]
    train_data = combined_data[:ntrain]

    train_labels = []
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append([s == true_s for s in d])

    train_data = np.concatenate(train_data).tolist()

    # Create test data
    reshaped_data = np.array(
        [
            [honest, untruthful]
            for honest, untruthful in zip(
                honest_statements[:-1], untruthful_statements[1:]
            )
        ]
    ).flatten()
    test_data = reshaped_data[ntrain : ntrain * 2].tolist()

    print(f"Train data: {len(train_data)}")
    print(f"Test data: {len(test_data)}")

    return {
        "train": {"data": train_data, "labels": train_labels},
        "test": {"data": test_data, "labels": [[1, 0]] * len(test_data)},
    }


def process_data_for_detection_results(
    scores: np.ndarray,
    norm_style: Literal["mean", "flip"],
    selection_style: Literal["positive", "negative"],
    threshold=0,
):
    """Do some data processing - just extracted this, don't exactly know what it does yet.

    Used in detection_result"""
    mean, std = np.median(scores), scores.std()
    scores[(scores > mean + 5 * std) | (scores < mean - 5 * std)] = (
        mean  # get rid of outliers
    )
    mag = max(0.3, np.abs(scores).std() / 10)
    min_val, max_val = -mag, mag
    norm = Normalize(vmin=min_val, vmax=max_val)

    if norm_style == "mean":
        scores = scores - threshold  # change this for threshold
        scores = scores / np.std(scores[5:])
        scores = np.clip(scores, -mag, mag)
    else:  # flip
        scores = -scores

    scores[np.abs(scores) < 0.0] = 0

    # ofs = 0
    # scores = np.array([scores[max(0, i-ofs):min(len(scores), i+ofs)].mean() for i in range(len(scores))]) # add smoothing

    if selection_style == "negative":
        scores = np.clip(scores, -np.inf, 0)
        scores[scores == 0] = mag
    else:  # positive
        scores = np.clip(scores, 0, np.inf)

    return norm, scores


def plot_detection_results(
    words,
    scores: np.ndarray,
    norm_style: Literal["mean", "flip"] = "mean",
    selection_style: Literal["positive", "negative"] = "negative",
    threshold=0,
):
    """
    Only pass in the words & scores of the stuff we want to see"""
    assert len(words) == len(scores), "Mismatched lengths"

    norm, processed_scores = process_data_for_detection_results(
        scores, norm_style, selection_style, threshold
    )

    cmap = LinearSegmentedColormap.from_list(
        "rg", ["r", (255 / 255, 255 / 255, 224 / 255), "g"], N=256
    )
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
    for word, score in zip(words, processed_scores):
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

        word_height = (
            text.get_window_extent(fig.canvas.get_renderer())
            .transformed(ax.transData.inverted())
            .height
        )

        # Add the text with background color
        text = ax.text(
            x,
            y + y_pad * (1),  # this used to have iter
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


def better_lat_scan(
    scores,
    starting_layer=1,
    bound=2.3,
    threshold=0,
):
    """scores is 2d nd.arr"""

    # Clip & standardize scores
    scores[np.abs(scores) < threshold] = 1
    standardized_scores = scores.clip(-bound, bound)

    _, ax = plt.subplots(figsize=(5, 4), dpi=200)
    sns.heatmap(
        -standardized_scores.T,
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

    # x label appear every 5 ticks
    ax.set_xticks(np.arange(0, len(standardized_scores), 5)[1:])
    ax.set_xticklabels(np.arange(0, len(standardized_scores), 5)[1:])
    ax.tick_params(axis="x", rotation=0)

    ax.set_yticks(np.arange(0, len(standardized_scores[0]), 5)[1:])
    ax.set_yticklabels(
        np.arange(starting_layer, len(standardized_scores[0]) + starting_layer, 5)[
            ::-1
        ][1:]
    )
    ax.set_title("LAT Neural Activity")
    plt.show()
