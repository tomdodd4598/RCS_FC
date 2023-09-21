import csv
import glob
import numpy as np
import os

from matplotlib import pyplot as plt


folders = glob.glob('Full Circuit\\e0_*')
slope = {}
# folders = glob.glob('e0_*')
cof = {}


def read_files(target, name):
    for folder in folders:
        folder_data = []
        files = sorted(glob.glob(os.path.join(folder, f'result\\*_{name}.csv')))
        for file in files:
            with open(file, 'r') as f:
                reader = csv.reader(f)
                file_data = [list(map(float, row)) for row in reader]
                folder_data.append(file_data)
        key = int(folder.split('_')[1])
        target[key] = folder_data


def read_slope(qubit, circuit, n, m):
    return slope[qubit][circuit][n][m]


def read_cof(qubit, circuit, n, m):
    return cof[qubit][circuit][n][m]


def read_results(n):
    path = f'Full Circuit\\e0_{n}\\result\\result.txt'

    file1 = open(path, 'r')
    y: list = file1.readlines()

    _2_minus_n = 1 / (2 ** n)
    for i in range(len(y)):
        y[i] = y[i].split(',')
        for j in range(len(y[i])):
            f = float(y[i][j])
            y[i][j] = f if f < _2_minus_n else _2_minus_n
            # y[i][j] = float(y[i][j])

    return y


def func(x, a, b):
    return a / x + b


# What does this function do?
def plot_predictions(qubit, circuit, n, m):
    data = read_results(qubit)[circuit]
    the_slope = read_slope(qubit, circuit, n, m)
    the_cof = read_cof(qubit, circuit, n, m)
    print(f'y = exp(f(x)), f(x) = a / x + b , a = {the_slope}, b = {the_cof}')

    def fit(x):
        return np.exp(func(x, the_slope, the_cof))

    plt.plot(list(range(1, qubit + 1)), data[1:])
    plt.plot(list(range(n, m + 1)), [fit(i) for i in range(n, m + 1)])
    plt.plot(list(range(1, qubit + 1)), [1 / (2 ** qubit) for _ in range(1, qubit + 1)])
    plt.yscale('log')
    plt.show()


def main():
    read_files(slope, 'slope')
    read_files(cof, 'cof')
    # plot_predictions(24, 10, 1, 2)


if __name__ == '__main__':
    main()
