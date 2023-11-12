import cupy as cp
import math
import matplotlib.pyplot as plt

from tqdm import tqdm


# Convert int to list of bits
def int_to_bits(s):
    return list(map(int, str((bin(s)))[2:]))


# Convert binary string to int
def bin_to_int(a):
    return int(a, 2)


def calculate(filename, n, m, bin_orders, order_sublists):
    z_file: list = [0 for _ in range(n + 1)]
    f = open(filename, 'r')
    file_list_lines = f.readlines()
    file_list = cp.array([bin_to_int(i) for i in file_list_lines])

    index = 0
    for s_bar_num in tqdm(range(len(order_sublists)), leave=False, desc='Processing s_bar'):
        s_bar = order_sublists[s_bar_num]
        for s_num in tqdm(range(len(s_bar)), leave=False):
            s = s_bar[s_num]
            temp = cp.bitwise_and(s, file_list)

            def bin_order(x):
                return bin_orders[x]

            temp = (-1) ** bin_order(temp)
            z_file[index] = z_file[index] + int(cp.sum(temp)) ** 2

        index = index + 1

    z_plt = z_file.copy()

    for i in range(len(z_file)):
        z_plt[i] = z_plt[i] / (math.factorial(n) / (math.factorial(i) * math.factorial(n - i)))
        z_plt[i] = z_plt[i] / (m ** 2)
    return z_plt


def main():
    n = 12
    # n = int(sys.argv[1])
    # full_circuit = True
    m = 500000
    d = 10
    s_length = 2 ** n
    bin_orders = cp.array([
        int(sum(int_to_bits(i)))
        for i in tqdm(range(0, s_length), desc='Initializing Step 1', leave=False)
    ])

    order_sublists = [[] for _ in range(0, n + 1)]

    for i in tqdm(range(0, s_length), desc='Initializing Step 2', leave=False):
        order_sublists[sum(int_to_bits(i))].append(i)
    order_sublists = [cp.array(order_sublist) for order_sublist in order_sublists]

    z = cp.array([0 for _ in range(n + 1)])
    z_list = []
    # for file_num in tqdm(range(1, d + 1), desc='Processing circuit'):

    #     filename = f'n{n}\\measurements_patch_n{n}_m14_s{9 + file_num}_e18_pEFGH.txt'

    #     z_file = calculate(filename, n, m, bin_orders, order_sublists)
    #     z_list.append(z_file)
    #     z = z + cp.array(z_file)

    #     plt.plot(list(range(1, n + 1)), z_file[1:])
    #     plt.plot(list(range(1, n + 1)),[1/(2**n)]*len(list(range(1, n + 1))))
    #     plt.yscale('log')
    #     plt.savefig(f'n{n}\\result\\measurements_patch_n{n}_m14_s{9 + file_num}_e18_pEFGH_LOG')
    #     plt.clf()

    #     plt.plot(list(range(1, n + 1)), z_file[1:])
    #     plt.plot(list(range(1, n + 1)),[1/(2**n)]*len(list(range(1, n + 1))))
    #     plt.savefig(f'n{n}\\result\\measurements_patch_n{n}_m14_s{9 + file_num}_e18_pEFGH')
    #     plt.clf()

    # z = z / d
    # z_list.append(z)

    # # ========================================================
    # f = open(f'n{n}\\result\\result.txt', 'w')
    # for i in z_list:
    #     f.write(','.join([str(j) for j in i]) + '\n')
    # # ========================================================

    # plt.plot(list(range(1, n + 1)), list(z.get())[1:])
    # plt.plot(list(range(1, n + 1)),[1/(2**n)]*len(list(range(1, n + 1))))
    # plt.yscale('log')
    # plt.savefig(f'n{n}\\result\\finalResult_LOG')
    # # plt.show()
    # plt.clf()

    # plt.plot(list(range(1, n + 1)), list(z.get())[1:])
    # plt.plot(list(range(1, n + 1)),[1/(2**n)]*len(list(range(1, n + 1))))
    # plt.savefig(f'n{n}\\result\\finalResult')
    # # plt.show()
    # plt.clf()

    # ========================================================
    # ========================================================
    # ========================================================

    for file_num in tqdm(range(1, d + 1), desc='Processing circuit'):
        filename = f'Full Circuit\\e0_{n}\\measurements_n{n}_m14_s{file_num - 1}_e0_pEFGH.txt'

        z_file = calculate(filename, n, m, bin_orders, order_sublists)
        z_list.append(z_file)
        z = z + cp.array(z_file)

        plt.plot(list(range(1, n + 1)), z_file[1:])
        plt.plot(list(range(1, n + 1)), [1 / s_length] * len(list(range(1, n + 1))))
        plt.yscale('log')
        plt.savefig(f'Full Circuit\\e0_{n}\\result\\{file_num - 1}_LOG')
        plt.clf()

    z = z / d
    z_list.append(z)

    # ========================================================
    f = open(f'Full Circuit\\e0_{n}\\result\\result.txt', 'w')
    for i in z_list:
        f.write(','.join([str(j) for j in i]) + '\n')
    # ========================================================

    plt.plot(list(range(1, n + 1)), list(z.get())[1:])
    plt.plot(list(range(1, n + 1)), [1 / s_length] * len(list(range(1, n + 1))))
    plt.yscale('log')
    plt.savefig(f'Full Circuit\\e0_{n}\\result\\finalResult_LOG')
    # plt.show()
    plt.clf()

    # ========================================================
    # ========================================================
    # ========================================================

    # z = calculate('e0_12\measurements_n12_m14_s0_e0_pEFGH.txt', 12, m, bin_orders, order_sublists)
    # plt.plot(list(range(1,n + 1)), list(z)[1:])
    # plt.plot(list(range(1, n + 1)),[1/(2**n)]*len(list(range(1, n + 1))))
    # plt.yscale('log')
    # # plt.savefig(f'n{n}\\GBSLOG')
    # plt.show()
    # plt.clf()

    # plt.plot(list(range(1,n + 1)), list(z)[1:])
    # plt.plot(list(range(1, n + 1)),[1/(2**n)]*len(list(range(1, n + 1))))
    # # plt.savefig(f'n{n}\\GBS')
    # plt.show()
    # plt.clf()


if __name__ == '__main__':
    main()
