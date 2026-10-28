import numpy as np
import matplotlib.pyplot as plt


def stacked_bar(data, row_labels=None, col_labels=None, title="", ax=None):
    data = np.asarray(data)
    nrows, ncols = data.shape

    if row_labels is None:
        row_labels = [str(i) for i in range(nrows)]
    if col_labels is None:
        col_labels = [str(j) for j in range(ncols)]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    x = np.arange(nrows)
    sorted_data = np.sort(data, axis=1)

    for j in range(ncols):
        bottom = sorted_data[:, j - 1] if j > 0 else np.zeros(nrows)
        height = sorted_data[:, j] - bottom
        ax.bar(x, height, bottom=bottom, label=col_labels[j])

    ax.set_xticks(x, row_labels, rotation=30, ha='right')
    ax.set_title(title)
    ax.legend()

    return fig, ax


if __name__ == "__main__":
    data = np.array([
        [3, 2, 5],
        [4, 1, 6],
        [2, 7, 3],
        [5, 3, 2],
    ])
    fig, ax = stacked_bar(data, row_labels=["A", "B", "C", "D"],
                          col_labels=["X", "Y", "Z"], title="Example")
    plt.show()
