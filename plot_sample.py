import matplotlib.pyplot as plt

'''
def read_results(qub, n):
    path = f'{qub}\\sample\\{n}.txt'
    
    file1 = open(path, 'r')
    y: list = file1.readlines()
    
    for i in range(len(y)):
        y[i] = y[i].split(',')
        for j in range(len(y[i])):
            y[i][j] = float(y[i][j])
    
    return y
'''


def read_sample_data(qub, n, size):
    path = f'Full Circuit\\e0_{qub}\\sample\\{size}\\{n}.txt'
    
    file1 = open(path, 'r')
    y: list = file1.readlines()
    
    for i in range(len(y)):
        y[i] = y[i].split(',')
        for j in range(len(y[i])):
            y[i][j] = float(y[i][j])
    
    return y


def read_average_result(n):
    path = f'Full Circuit\\e0_{n}\\result\\result.txt'
    
    file1 = open(path, 'r')
    y: list = file1.readlines()
    
    for i in range(len(y)):
        y[i] = y[i].split(',')
        for j in range(len(y[i])):
            y[i][j] = float(y[i][j])
    
    return y


def main():
    qubits = [x for x in range(12, 25, 2)]
    sample_sizes = [5, 7, 10]
    sample_indices = [i for i in range(1, 11)]
    for qub in qubits:
        result = read_average_result(qub)[-1]

        colors = ['red', 'blue', 'green']
        color_index = 0

        x = range(0, 1 + qub)

        for size in sample_sizes:
            for index in sample_indices:
                name = f'{index}_{size}'
                points = read_sample_data(qub, name, size)
                for exp in points:
                    plt.plot(x[1:], exp[1:], marker='*', linestyle='--', alpha=0.1, color=colors[color_index], label=f'{size} Groups')
                plt.plot(x[1:], [1 / (500000 / size) for _ in x[1:]], linestyle='-', color=colors[color_index], label='1/m')
            color_index += 1

        plt.yscale('log')
        plt.xlabel('Order')
        plt.ylabel('Average Correlator')
        plt.title(f'{qub} Qubit, Separated Into 5/7/10 Groups')
        plt.plot(x[1:], result[1:], marker='.', linestyle='-', color='black', label='Overall Average')
        plt.plot(x[1:], [1 / 500000 for _ in x[1:]], linestyle='-', color='black', label='1/m')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.savefig(f'Full Circuit\\e0_{qub}\\sample\\sample_result')
        plt.clf()


if __name__ == '__main__':
    main()
