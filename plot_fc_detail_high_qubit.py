import cupy as cp
import math
import matplotlib.pyplot as plt
import numpy as np

from itertools import combinations
from tqdm import tqdm


@np.vectorize
def int_to_bin_list(s):
    return sum(list(map(int, str((bin(s)))[2:])))


def bin_to_int(a):
    return int(a, 2)


@np.vectorize
def hamming_weight(n):
    return bin(n).count('1')


def generate_bit_strings(n):
    # Generate bit_strings containing exactly one '1'.
    bit_strings_one = [
        ''.join('1' if i in combo else '0' for i in range(n))
        for combo in combinations(range(n), 1)
    ]
    bit_strings_one.reverse()
    # Generate bit_strings containing exactly two '1's.
    bit_strings_two = [
        ''.join('1' if i in combo else '0' for i in range(n))
        for combo in combinations(range(n), 2)
    ]
    bit_strings_two.reverse()
    return [
        cp.array([bin_to_int(a) for a in bit_strings_one]),
        cp.array([bin_to_int(a) for a in bit_strings_two])
    ]


def calculate(_filename, n, m, bin_order, s_bar_list):
    detail = {}
    z_file = [0 for _ in range(n + 1)]

    index = 0
    # print(1)

    # def binO(x): return np.array([bin_order[int(xx)] for xx in x])
    def bin_o(x):
        return bin_order[int(x)]

    bin_order_index = 0
    for s_bar_num in range(0, len(s_bar_list)):
        s_bar = s_bar_list[s_bar_num]
        for s_num in tqdm(range(len(s_bar)), leave=False):
            # print(2)
            s = s_bar[s_num]
            # print(3)
            # temp = cp.bitwise_and(s, file_list)
            temp = list(range(bin_order_index, bin_order_index + m))
            # def binO(x): return cp.array([int(sum(int_to_bin_list(i))) for i in x])
            # def binO(x): return bin_order[int(x)]

            # print(4)
            temp = (-1) ** bin_o(temp)
            # print(5)
            z_file[index] = z_file[index] + int(cp.sum(temp)) ** 2
            detail[int(s)] = int(cp.sum(temp)) / m
            bin_order_index = bin_order_index + m

        index = index + 1

    # print(z_file)

    z_plt: list = z_file.copy()

    for i in range(len(z_file)):
        z_plt[i] = z_plt[i] / (math.factorial(n) / (math.factorial(i) * math.factorial(n - i)))
        z_plt[i] = z_plt[i] / (m ** 2)
    # print(z_plt)
    return z_plt, detail


def plot_bar(ax, categories, values):
    bars = ax.bar(categories, values, color=['blue' if v > 0 else 'red' for v in values])
    # print(values)
    ax.axhline(0, color='black', linewidth=0.8)

    for bar in bars:
        y_val = bar.get_height()
        if y_val >= 0:
            label_position = y_val - 0.1
        else:
            label_position = y_val + 0.05
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_position,
            round(y_val, 2),
            color='white',
            ha='center',
            va='center'
        )

    # plt.show()


def plot_3d(ax, n, _detail):
    _x = np.arange(n).tolist()
    _y = np.arange(n).tolist()
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel().tolist(), _yy.ravel().tolist()
    z = [0] * len(x)
    dx = dy = [0.5] * len(z)
    dz = [2 ** x[i] + 2 ** y[i] if x[i] != y[i] else 0 for i in range(len(x))]
    ax.bar3d(x, y, z, dx, dy, dz)


def plot_heatmap(fig, ax, n, detail):
    _x = np.arange(n).tolist()
    _y = np.arange(n).tolist()
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel().tolist(), _yy.ravel().tolist()
    # z = [0] * len(x)
    # dx = dy = [0.5] * len(z)
    dz = [detail[2 ** x[i] + 2 ** y[i]] if x[i] != y[i] else 0 for i in range(len(x))]
    square_list = []

    for i in range(0, len(dz), n):
        square_list.append(dz[i:i + n])
    x_labels = [i for i in range(0, n)]
    ax.set_xticks(x_labels)
    ax.set_yticks(x_labels)
    x_labels = [str(i) for i in range(1, n + 1)]
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(x_labels)

    im = ax.imshow(square_list, cmap='RdBu', origin='lower', vmin=-max(max(dz), -min(dz)), vmax=max(max(dz), -min(dz)))

    fig.colorbar(im, ax=ax)  # Add color bar on specified axes


def main():
    for n in tqdm(range(12, 51, 2), desc='Processing qubit', leave=True):
        # n = 48
        # n = int(sys.argv[1])
        m = 500000
        d = 10
        # s_length = 2 ** n

        s_bar_list = generate_bit_strings(n)

        for file_num in tqdm(range(1, d + 1), desc='Processing circuit', leave=False):
            detail = {}

            # filename = 'bit_strings.txt'

            filename = f'Full Circuit\\e0_{n}\\measurements_n{n}_m14_s{file_num - 1}_e0_pEFGH.txt'

            # z_file = [0 for _ in range(n + 1)]
            f = open(filename, 'r')
            file_list_lines = f.readlines()
            file_list = cp.array([bin_to_int(i) for i in file_list_lines])
            for s_bar_num in range(0, len(s_bar_list)):
                s_bar = s_bar_list[s_bar_num]
                # temp = []
                # print(len(s_bar))
                for s_num in tqdm(range(len(s_bar)), leave=False):
                    s = s_bar[s_num]
                    temp = cp.bitwise_and(s, file_list)
                    temp = temp.get()
                    temp = hamming_weight(temp)
                    temp = (-1) ** temp
                    detail[int(s)] = int(np.sum(temp)) / m
            # print(detail)

            fig = plt.figure(figsize=(20, 18))
            ax1 = fig.add_subplot(211)
            plot_bar(
                ax1,
                [str(int(math.log(i, 2) + 1)) for i in s_bar_list[0]],
                [detail[int(i)] for i in s_bar_list[0]]
            )

            # ax2 = fig.add_subplot(223, projection='3d')
            # plot_3d(ax2, n, detail)

            ax3 = fig.add_subplot(212)
            plot_heatmap(fig, ax3, n, detail)
            # plt.tight_layout()
            # plt.show()
            plt.savefig(f'Full Circuit\\e0_{n}\\s{file_num - 1}')
            plt.close()
            plt.close(fig)


if __name__ == '__main__':
    main()
