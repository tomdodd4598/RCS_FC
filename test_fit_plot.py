import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


# Define possible functional forms, such as exponential functions
def func(x, a, b):
    return a * np.power(x, b)


def main():
    # x is the index of the input
    # y is the actual data value
    y = np.array([
        0.00010734521023030304,
        7.802284341818183e-05,
        6.534640731151516e-05,
        5.638831208888889e-05,
        5.1025149513419915e-05,
        4.574705914343434e-05,
        4.2067672087272726e-05,
        3.728420493090909e-05,
        3.0994441163636365e-05,
        2.6831631200000004e-05,
        9.3911584e-06
    ])
    x = np.array(range(1, len(y) + 1))

    # Use curve_fit for fitting, and p_opt is the optimal parameter value obtained by fitting
    # noinspection PyTupleAssignmentBalance
    p_opt, p_cov = curve_fit(func, x, y)

    # Print the parameter values obtained by fitting
    print(f'Parameter values obtained by fitting: {p_opt}')

    # Use the parameter values obtained by fitting to generate the y value of the fitted curve
    y_fit = func(x, *p_opt)

    # Create new figure
    plt.figure()

    # Plot raw data points
    plt.plot(x, y, 'bo', label='ori')

    # Draw fitting curve
    plt.plot(x, y_fit, 'r-', label='fit')

    # Set the y-axis to logarithmic scale
    plt.yscale('log')

    # Add legend
    plt.legend()

    # Show figure
    plt.show()


if __name__ == '__main__':
    main()
