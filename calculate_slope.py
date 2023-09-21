import csv
import numpy as np

from scipy.optimize import curve_fit, OptimizeWarning


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
def predict_slope(result, n, m):
    if n < m:
        x = np.array([i for i in range(n, m + 1)])
        y = np.array(result[n:m + 1])
        y = np.log(y)
    else:
        x = np.array([i for i in range(m, n + 1)])
        y = np.array(result[m:n + 1])
        y = np.log(y)
    try:
        # noinspection PyTupleAssignmentBalance
        params, _ = curve_fit(func, x, y)
    except (ValueError, RuntimeError, RuntimeWarning, OptimizeWarning) as _:
        # print(e)
        params = [0, 0]

    return params


def main():
    for i in range(12, 25, 2):
        slope = [[0 for _ in range(i + 1)] for _ in range(i + 1)]
        cof = [[0 for _ in range(i + 1)] for _ in range(i + 1)]

        # slope.append(math.log(result[1]) - math.log(result[3]))

        def write_csvs(result, name):
            length = len(result)
            for n in range(length):
                for m in range(length):
                    if m == n:
                        pass
                    else:
                        slope[n][m], cof[n][m] = predict_slope(result, m, n)

            with open(f'Full Circuit\\e0_{i}\\result\\{name}_slope.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(slope)
            with open(f'Full Circuit\\e0_{i}\\result\\{name}_cof.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(cof)

        # Why is the last row ignored?
        results = read_results(i)[:-1]
        # print(index)
        # print(result)

        for j in range(len(results)):
            write_csvs(results[j], j)

        # Why is the penultimate row singled out?
        write_csvs(results[-1], 'final')
    # print(slope)


if __name__ == '__main__':
    main()
