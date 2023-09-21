import math
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm


def int_to_bin_list(s):
    return list(map(int, str((bin(s)))[2:]))


def bin_to_int(a):
    return int(a, 2)


def calculate(filename, n, m, bin_order, s_bar_list):
    detail = {}
    z_file = [0 for _ in range(n + 1)]
    f = open(filename, 'r')
    file_list_lines = f.readlines()
    file_list = np.array([bin_to_int(i) for i in file_list_lines])

    index = 0
    for s_bar_num in tqdm(range(1, len(s_bar_list)), leave=False, desc='Processing s_bar'):
        s_bar = s_bar_list[s_bar_num]
        for s_num in tqdm(range(len(s_bar)), leave=False):
            s = s_bar[s_num]
            temp = np.bitwise_and(s, file_list)

            def bin_o(x):
                return bin_order[x]

            temp = (-1) ** bin_o(temp)
            z_file[index] = z_file[index] + int(np.sum(temp)) ** 2
            detail[int(s)] = int(np.sum(temp)) / m
        if s_bar_num >= 2:
            break
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

    ax.axhline(0, color='black', linewidth=0.8)

    for bar in bars:
        y_val = bar.get_height()
        if y_val >= 0:
            label_position = y_val - 0.1
        else:
            label_position = y_val + 0.05
        ax.text(bar.get_x() + bar.get_width() / 2, label_position, round(y_val, 2), color='white', ha='center',
                va='center')

    # plt.show()


def plot_3d(ax, n, detail):
    _x = np.arange(n)
    _y = np.arange(n)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    z = np.zeros_like(x)
    dx = dy = np.ones_like(z) * 0.5
    dz = [detail[2 ** x[i] + 2 ** y[i]] if x[i] != y[i] else 0 for i in range(len(x))]
    ax.bar3d(x, y, z, dx, dy, dz)
    # plt.show()


def plot_heatmap(fig, ax, n, detail):
    _x = np.arange(n)
    _y = np.arange(n)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    # z = np.zeros_like(x)
    # dx = dy = np.ones_like(z) * 0.5
    dz = [detail[2 ** x[i] + 2 ** y[i]] if x[i] != y[i] else 0 for i in range(len(x))]
    square_list = []

    for i in range(0, len(dz), n):
        square_list.append(dz[i:i + n])
    x_labels = [i for i in range(0, n)]
    # y_labels = [str(i) for i in range(1, n + 1)]
    ax.set_xticks(x_labels)
    ax.set_yticks(x_labels)
    x_labels = [str(i) for i in range(1, n + 1)]
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(x_labels)

    im = ax.imshow(square_list, cmap='RdBu', origin='lower', vmin=-max(max(dz), -min(dz)), vmax=max(max(dz), -min(dz)))

    fig.colorbar(im, ax=ax)  # Add color bar on specified axes


def main():
    n = 12
    # n = int(sys.argv[1])
    m = 500000
    d = 10
    s_length = 2 ** n
    bin_order = np.array([
        int(sum(int_to_bin_list(i)))
        for i in tqdm(range(0, s_length), desc='Initializing Step 1', leave=False)
    ])

    s_bar_list = [[] for _ in range(0, n + 1)]

    for i in tqdm(range(0, s_length), desc='Initializing Step 2', leave=False):
        s_bar_list[sum(int_to_bin_list(i))].append(i)
    s_bar_list = [np.array(i) for i in s_bar_list]

    z = np.array([0 for _ in range(n + 1)])
    z_list = []

    for file_num in tqdm(range(1, d + 1), desc='Processing circuit'):
        filenames = f'Full Circuit\\e0_{n}\\measurements_n{n}_m14_s{file_num - 1}_e0_pEFGH.txt'
        # filenames = 'bit_strings.txt'

        z_file, detail = calculate(filenames, n, m, bin_order, s_bar_list)
        print(detail)
        fig = plt.figure(figsize=(20, 18))
        ax1 = fig.add_subplot(221)
        plot_bar(ax1, [str(int(math.log(i, 2) + 1)) for i in s_bar_list[1]], [detail[int(i)] for i in s_bar_list[1]])

        ax2 = fig.add_subplot(223, projection='3d')
        plot_3d(ax2, n, detail)

        ax3 = fig.add_subplot(224)
        plot_heatmap(fig, ax3, n, detail)
        # plt.tight_layout()
        plt.show()
        # plt.savefig(f'e0_{n}\\s{file_num - 1}')

        z_list.append(z_file)
        z = z + np.array(z_file)

    # z = z / d
    # z_list.append(z)

    # # ========================================================
    # f = open(f'e0_{n}\\result\\result.txt', 'w')
    # for i in z_list:
    #     f.write(','.join([str(j) for j in i]) + '\n')
    # # ========================================================

    # plt.plot(list(range(1, n + 1)), list(z.get())[1:])
    # plt.plot(list(range(1, n + 1)), [1 / s_length] * len(list(range(1, n + 1))))
    # plt.yscale('log')
    # plt.savefig(f'e0_{n}\\result\\finalResult_LOG')
    # # plt.show()
    # plt.clf()


if __name__ == '__main__':
    main()
