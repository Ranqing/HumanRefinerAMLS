# re-write from MLS1D/.m

import sys
import numpy as np
from matplotlib import pyplot as plt
from qing_mls import qing_1d_mls


def qing_test_1d_mls_fitting():
    print('qing_test_1d_mls_fitting')

    l = 10.0
    m = 1
    scale = 3

    dx = 0.5
    xi = np.arange(0, l, dx, dtype=float)
    xi = np.append(xi, [l])
    nnodes = len(xi)
    print('xi: ', xi, '\nnnodes = ', nnodes, end='\n')

    dx = 0.05
    x = np.arange(0, l, dx, dtype=float)
    x = np.append(x, [l])
    npoints = len(x)
    print('x: ', x, '\nnpoints = ', npoints, end='\n')

    scale = 3
    dx = 0.5
    dm = scale * dx * np.ones(nnodes)
    print('dm: ', dm, end='\n')

    # calculating weighting function
    PHI, DPHI, DDPHI = qing_1d_mls(m, nnodes, xi, npoints, x, dm, 'GAUSS', 3.0)
    print('PHI shape: ', PHI.shape)
    print('DPHI shape: ', DPHI.shape)
    print('DDPHI shape: ', DDPHI.shape)
    fid1 = open('shp.dat', 'w')
    fid2 = open('dshp.dat', 'w')
    fid3 = open('ddshp.dat', 'w')
    fid1.write('%10s%10s%10s%10s\n' % (' ', 'N0', 'N10', 'N20'))
    fid2.write('%10s%10s%10s%10s\n' % (' ', 'N1', 'N10', 'N20'))
    fid3.write('%10s%10s%10s%10s\n' % (' ', 'N1', 'N10', 'N20'))

    for j in range(0, npoints):
        fid1.write('%10.4f' % x[j])
        fid2.write('%10.4f' % x[j])
        fid3.write('%10.4f' % x[j])
        fid1.write('%10.4f%10.4f%10.4f\n' %
                   (PHI[j][0], PHI[j][10], PHI[j][20]))
        fid2.write('%10.4f%10.4f%10.4f\n' %
                   (DPHI[j][0], DPHI[j][10], DPHI[j][20]))
        fid3.write('%10.4f%10.4f%10.4f\n' %
                   (DDPHI[j][0], DDPHI[j][10], DDPHI[j][20]))

    fid1.close()
    fid2.close()
    fid3.close()

    f1 = plt.figure(1)
    sub1 = plt.subplot(311)
    sub1.plot(x, PHI[:, 0], label='fitting data')
    # sub1.grid(True)
    sub1.legend()

    sub2 = plt.subplot(312)
    sub2.plot(x, DPHI[:, 0], label='one-order derivatives')
    # sub2.grid(True)
    sub2.legend()

    sub3 = plt.subplot(313)
    sub3.plot(x, DDPHI[:, 0], label='second-order derivatives')
    # sub3.grid(True)
    sub3.legend()

    # plt.legend()

    plt.show()

    print('\nstart to fitting curve sin(x)', end='\n')
    yi = np.sin(xi)
    y = np.sin(x)
    yh = np.dot(PHI, np.transpose(yi))  # approximate function
    err = np.linalg.norm(np.transpose(y) - yh) / np.linalg.norm(y) * 100

    dy = np.cos(x)
    dyh = np.dot(DPHI, np.transpose(yi))  # first order derivative
    errd = np.linalg.norm(np.transpose(dy) - dyh) / np.linalg.norm(dy) * 100

    ddy = -np.sin(x)
    ddyh = np.dot(DDPHI, np.transpose(yi))  # second order derivative
    errdd = np.linalg.norm(np.transpose(ddy) - ddyh) / \
        np.linalg.norm(ddy) * 100

    fid1 = open('fun.dat', 'w')
    fid2 = open('dfun.dat', 'w')
    fid3 = open('ddfun.dat', 'w')
    fid1.write('%10s%10s%10s\n' % (' ', 'Exact', 'Appr'))
    fid2.write('%10s%10s%10s\n' % (' ', 'Exact', 'Appr'))
    fid3.write('%10s%10s%10s\n' % (' ', 'Exact', 'Appr'))

    for j in range(0, npoints):
        fid1.write('%10.4f' % (x[j]))
        fid1.write('%10.4f%10.4f\n' % (y[j], yh[j]))
        fid2.write('%10.4f' % (x[j]))
        fid2.write('%10.4f%10.4f\n' % (dy[j], dyh[j]))
        fid3.write('%10.4f' % (x[j]))
        fid3.write('%10.4f%10.4f\n' % (ddy[j], ddyh[j]))

    fid1.close()
    fid2.close()
    fid3.close()

    # blue - input data; yellow - fitting
    fig = plt.figure(2)
    sub1 = plt.subplot(311)
    sub1.plot(x, y, x, yh)
    sub2 = plt.subplot(312)
    sub2.plot(x, dy, x, dyh)
    sub3 = plt.subplot(313)
    sub3.plot(x, ddy, x, ddyh)

    plt.show()
    sys.exit()

    pass
