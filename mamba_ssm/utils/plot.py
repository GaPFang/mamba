import matplotlib.pyplot as plt
import numpy as np
import os

def plot_matrix(matrix, dir, filename, vmin, vmax, xlabel, ylabel, cmap='viridis', colorbar=True, log=False):
    """
    Plotss a 2D matrix with colors based on their values.

    Parameters:
        matrix (2D array-like): The matrix to be visualized.
        cmap (str): The colormap to use (default is 'viridis').
        colorbar (bool): Whether to show the colorbar (default is True).
    """
    assert len(matrix.shape) == 2, "Matrix must be 2D"
    if matrix.shape[0] < matrix.shape[1]:
        matrix = matrix.T
        xlabel, ylabel = ylabel, xlabel
    if log:
        matrix = np.log(-matrix)
    matrix = np.array(matrix)  # Ensure the matrix is a NumPy array
    plt.figure(figsize=(30, 8))
    # plt.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    plt.imshow(matrix, cmap=cmap, aspect='auto')
    plt.title(f"{dir[1]} {dir[-1]} {filename}", fontsize=20)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    
    if colorbar:
        cbar = plt.colorbar()
        if log:
            cbar.set_label("log(Value)", fontsize=18)
        else:
            cbar.set_label("Value", fontsize=18)
        cbar.ax.tick_params(labelsize=18)
    
    dir_str = "/".join(dir)
    if not os.path.exists(dir_str):
        os.makedirs(dir_str)
    plt.savefig(f"{dir_str}/{filename}.png")

def plot_array(matrix, dir, filename, vmin, vmax, xlabel, ylabel, cmap='viridis', colorbar=True, log=False):
    """
    Draws a line chart for a 2D matrix.

    Parameters:
        matrix (2D array-like): The matrix where each row represents a line in the chart.
        x_values (array-like): Optional custom x-axis values. If None, uses column indices.
    """
    assert len(matrix.shape) == 2, "Matrix must be 2D"
    if matrix.shape[0] < matrix.shape[1]:
        matrix = matrix.T
        xlabel, ylabel = ylabel, xlabel
    matrix = np.array(matrix)  # Ensure the matrix is a NumPy array
    plt.figure(figsize=(30, 8))
    lines_list = [f"{ylabel}_{i}" for i in range(matrix.shape[0])]
    if log:
        matrix = np.log(-matrix)
    for i in range(matrix.shape[0]):
        plt.plot(matrix[i], label=lines_list[i], alpha=0.5)
    plt.legend()
    # plt.ylim(vmin, vmax)
    plt.title(f"{dir[1]} {dir[-1]} {filename}", fontsize=20)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel("value", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    if log:
        plt.ylabel("log(value)", fontsize=18)
    dir_str = "/".join(dir)
    if not os.path.exists(dir_str):
        os.makedirs(dir_str)
    plt.savefig(f"{dir_str}/{filename}.png")

def plot_hist(data, dir, filename, xlabel="Value", ylabel="Frequency"):
    """
    Plots a histogram of the given data.

    Parameters:
        data (array-like): The data to be visualized.
        bins (int): The number of bins to use (default is 10).
        xlabel (str): The label for the x-axis (default is 'Value').
        ylabel (str): The label for the y-axis (default is 'Frequency').
    """
    data = np.array(data).flatten()  # Ensure the data is a 1D NumPy array
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=10, color='skyblue', edgecolor='black', alpha=0.7, density=True)
    plt.title(f"{dir[1]} {dir[-1]} {filename}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    dir_str = "/".join(dir)
    if not os.path.exists(dir_str):
        os.makedirs(dir_str)
    plt.savefig(f"{dir_str}/{filename}.png")

def main():
    # Example usage
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 10]
    ])
    plot_matrix(matrix, ["test1", "test2"], "matrix", 0, 100, "c", "r", cmap='coolwarm')
    plot_array(matrix, ["test1", "test2"], "array", 0, 100, "c", "r", cmap='coolwarm')
    plot_hist([1, 2, 3, 4, 5, 6, 1, 8, 1, 10], ["test1", "test2"], "histogram", xlabel="Value", ylabel="Frequency")

if __name__ == "__main__":
    main()
