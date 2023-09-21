import math
from tqdm import tqdm
import cupy as cp


def int_to_bin_list(s):
    return list(map(int, str((bin(s)))[2:]))


def bin_to_int(a):
    return int(a, 2)


def calculate_order_averages(order, n, m, bin_order, s_bar_list, data):
    s_bar = s_bar_list[order]
    result: list = [0 for _ in range(len(data))]
    index = 0
    for file_list in data:
        for s_num in range(len(s_bar)):
            s = s_bar[s_num]
            temp = cp.bitwise_and(s, file_list)

            def bin_o(x):
                return bin_order[x]

            temp = (-1) ** bin_o(temp)
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
    n = 14
    # n = int(sys.argv[1])
    full_circuit = True
    m = 500000
    # d = 10
    s_length = 2 ** n
    bin_order = cp.array([
        int(sum(int_to_bin_list(i)))
        for i in tqdm(range(0, s_length), desc='Initializing Step 1', leave=False)
    ])

    s_bar_list = [[] for _ in range(0, n + 1)]

    for i in tqdm(range(0, s_length), desc='Initializing Step 2', leave=False):
        s_bar_list[sum(int_to_bin_list(i))].append(i)
    s_bar_list = [cp.array(i) for i in s_bar_list]

    # z = cp.array([0 for _ in range(n + 1)])
    # z_list = []

    if full_circuit:
        filenames = [f'Full Circuit\\e0_{n}\\measurements_n{n}_m14_s{i}_e0_pEFGH.txt' for i in range(10)]
    else:
        filenames = [f'Patch Circuit\\n{n}\\measurements_patch_n{n}_m14_s1{i}_e18_pEFGH.txt' for i in range(10)]
    result_list = [1]
    average_list = [1]
    for i in range(1, n + 1):
        data = read_files(filenames)
        result = calculate_order_averages(i, n, m, bin_order, s_bar_list, data)
        result_list.append(result)
        average = sum(result) / len(result)
        average_list.append(average)
        # print(average_list)
        slope = abs(
            round(math.log(average_list[i - 1]) - math.log(average_list[i]), 5))

        print(slope)
        if slope < 0.001:
            average_list = average_list + [[average] for _ in range(i, n + 1)]
            break
    print(average_list)


if __name__ == '__main__':
    main()
