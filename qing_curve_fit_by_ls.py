import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as optimization

def qing_square_func(xdata):
    xlen = len(xdata)
    square_xdata = np.zeros(xlen)
    for i in range(0, xlen):
        square_xdata[i] = xdata[i] * xdata[i]

    return square_xdata


def qing_quadratic_func(x, a, b, c):
    return a + b * x + c * x * x


def qing_ls(dsp_of_testy, xmin, xmax):
    f1 = plt.figure(1)

    xdata = np.array(range(xmin, xmax + 1, 1))
    ydata = dsp_of_testy[xmin:xmax + 1]
    plt.plot(xdata, ydata, 'b', label='origin')

    xlen = xmax - xmin + 1
    square_xdata = qing_square_func(xdata)
    A = np.vstack([square_xdata, xdata, np.ones(xlen)]).T

    print(A)
    print(ydata)
    # print(A.shape)
    # print(B.shape)
    # qing_draw_1d_narray(dsp_of_testy, 0, length)
    #
    m2, m1, m0 = np.linalg.lstsq(A, ydata)[0]
    print('m2 = %f' % m2, ', m1=%f' % m1, ', m0=%f' % m0)
    plt.plot(xdata, m2 * square_xdata + m1 * xdata + m0, 'r', label='fitted')
    plt.legend()
    plt.show()


# def qing_scipy_ls


def qing_curve_fit(dsp_of_testy, xmin, xmax):
    f1 = plt.figure(1)
    xdata = np.array(range(xmin, xmax + 1, 1))
    ydata = dsp_of_testy[xmin:xmax + 1]
    xlen = len(xdata)

    x0 = np.array([0.0, 0.0, 0.0])
    sigma = np.ones(xlen)
    plt.plot(xdata, ydata, 'b', label='origin')

    # print (optimization.curve_fit(qing_quadratic_func, xdata, ydata, x0, sigma))
    # print('a=%f'%a, ',b=%f'%b, ',c=%f'%c)
    abc = optimization.curve_fit(
        qing_quadratic_func, xdata, ydata, x0, sigma)[0]
    func_xdata = qing_quadratic_func(xdata, abc[0], abc[1], abc[2])
    plt.plot(xdata, func_xdata, 'r', label='fitted')

    plt.legend()
    plt.show()


def main():
    pass

if __name__ == '__main__':
    main()
