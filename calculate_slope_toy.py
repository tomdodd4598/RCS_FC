import csv
import math


def read_results(error, size):
    path = f'Toy Circuit\\{error}\\{size}\\result\\result.txt'

    file1 = open(path, 'r')
    y: list = file1.readlines()

    for i in range(len(y)):
        y[i] = y[i].split(',')
        for j in range(len(y[i])):
            y[i][j] = float(y[i][j])

    return y


def main():
    for error in range(11):
        for size in [1000, 10000, 100000, 1000000]:
            slope: list[list] = [[0 for _ in range(7)] for _ in range(7)]

            # slope.append(math.log(result[1]) - math.log(result[3]))

            def write_csvs(result, name):
                length = len(result)
                for n in range(length):
                    for m in range(length):
                        try:
                            slope[n][m] = (math.log(result[n]) - math.log(result[m])) / abs(n - m)
                        except (ValueError, RuntimeError, RuntimeWarning) as _:
                            pass
                with open(f'Toy Circuit\\{error}\\{size}\\result\\{name}_slope.csv', 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(slope)

            results = read_results(error, size)[:-1]

            for i in range(len(results)):
                write_csvs(results[i], f'exp_{i + 1}')

            write_csvs(results[-1], 'final')
    # print(slope)


if __name__ == '__main__':
    main()
