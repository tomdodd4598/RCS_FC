import cupy as cp
import math

from tqdm import tqdm


# Convert int to list of bits
def int_to_bits(s):
    return list(map(int, str((bin(s)))[2:]))


# Convert binary string to int
def bin_to_int(a):
    return int(a, 2)


def calculate_order_averages(order, n, m, bin_orders, order_sublists, data):
    s_bar_list = order_sublists[order]
    result: list = [0 for _ in range(len(data))]

    index = 0
    for file_list in data:
        for s_num in range(len(s_bar_list)):
            s_bar = s_bar_list[s_num]
            temp = cp.bitwise_and(s_bar, file_list)

            def bin_order(x):
                return bin_orders[x]

            temp = (-1) ** bin_order(temp)
            result[index] = result[index] + int(cp.sum(temp)) ** 2

        index = index + 1

    for i in range(len(result)):
        result[i] = result[i] / (math.factorial(n) / (math.factorial(order) * math.factorial(n - order)))
        result[i] = result[i] / (m ** 2)
    return result


def read_files(filenames):
    data = []
    for filename in filenames:
        f = open(filename, 'r')
        file_list_lines = f.readlines()
        file_list = cp.array([bin_to_int(i) for i in file_list_lines])
        data.append(file_list)
    return data


def main():
    n = 16
    # n = int(sys.argv[1])
    full_circuit = True
    m = 500000
    # d = 10
    s_length = 2 ** n
    bin_orders = cp.array([
        int(sum(int_to_bits(i)))
        for i in tqdm(range(0, s_length), desc='Initializing Step 1', leave=False)
    ])

    order_sublists = [[] for _ in range(0, n + 1)]

    for i in tqdm(range(0, s_length), desc='Initializing Step 2', leave=False):
        order_sublists[sum(int_to_bits(i))].append(i)
    order_sublists = [cp.array(order_sublist) for order_sublist in order_sublists]

    # z = cp.array([0 for _ in range(n + 1)])
    # z_list = []

    if full_circuit:
        filenames = [f'Full Circuit\\e0_{n}\\measurements_n{n}_m14_s{i}_e0_pEFGH.txt' for i in range(10)]
    else:
        filenames = [f'Patch Circuit\\n{n}\\measurements_patch_n{n}_m14_s1{i}_e18_pEFGH.txt' for i in range(10)]
    result_list = [1]
    average_list = [1]
    for k in range(1, n + 1):
        print(f'k={k}')
        data = read_files(filenames)
        result = calculate_order_averages(k, n, m, bin_orders, order_sublists, data)
        result_list.append(result)
        average = sum(result) / len(result)
        average_list.append(average)
        # print(average_list)
        slope = abs(round(math.log(average_list[k - 1]) - math.log(average_list[k]), 5))

        # print(slope)
        if slope < 0.001:
            average_list = average_list + [[average] for _ in range(k, n + 1)]
            break
    print(average_list)


if __name__ == '__main__':
    main()
