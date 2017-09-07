from qing_operation import *
from qing_weight import *
from matplotlib import pyplot as plt
import cv2
import numpy as np
import scipy.optimize as optimization


def qing_read_txt(txtname):
    # for line in open(txtname):
    #     print(line, end=' ')
    # pass
    # disp = numpy.loadtxt(txtname)
    # print(disp.shape)
    with open(txtname, 'r') as f:
        data = f.readlines()
        for line in data:
            print(line)
            odom = line.split()
            numbers_float = map(float, odom)
            # print(numbers_float)


def qing_save_txt(mtx, txtname):
    np.savetxt(txtname, mtx[:, :], fmt="%d")
    print('saving ' + txtname)


# range of x-axis is generated automatically
def qing_draw_1d_narray(array_data, xmin, xmax):
    f1 = plt.figure(1)
    xdata = range(xmin, xmax, 1)
    plt.plot(xdata, array_data[xmin:xmax])
    plt.xlabel('x')
    plt.ylabel('d')
    plt.show()


def qing_1d_median_filter(dsp_of_testy, wnd_sz):
    rangex = len(dsp_of_testy)
    offset = int(wnd_sz * 0.5)
    dmax = dsp_of_testy.max()
    dmin = dsp_of_testy.min()
    # print('dmax = %d' % dmax, 'dmin = %d' % dmin, 'wnd_sz = %d' % wnd_sz)

    for x in range(0, rangex):
        dhist = np.zeros((int(dmax - dmin + 1), 1))
        for j in range(-offset, offset + 1):
            xj = min(rangex - 1, x + j)
            xj = max(0, xj)
            d = int(dsp_of_testy[xj])
            if not d == 0:
                dhist[d] += 1

            # if x == 312:
            #     print('x = %d ' % x, 'xj = %d, ' % xj, 'd = %d' % d)

        count = 0
        middle = 0
        for j in range(0, len(dhist)):
            # if x == 312:
            #     print('j = %d, ' % j, 'hist = %d' % dhist[j])
            count += dhist[j]
            if count * 2 > wnd_sz:
                middle = j
                break
            pass

        dsp_of_testy[x] = middle + dmin


def qing_move_outliers(dsp_of_testy, xmin, xmax):
    wnd_sz = 5
    qing_1d_median_filter(dsp_of_testy, wnd_sz)


def qing_square_func(xdata):
    xlen = len(xdata)
    square_xdata = np.zeros(xlen)
    for i in range(0, xlen):
        square_xdata[i] = xdata[i] * xdata[i]

    # for i in range(0, xlen):
    #     print('i = %d' % xdata[i], 'i*i = %d' % square_xdata[i])
    return square_xdata


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


def qing_quadratic_func(x, a, b, c):
    return a + b * x + c * x * x

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


# SYNTAX: [PHI, DPHI, DDPHI] = MLS1DShape(m, nnodes, xi, npoints, x, dm, wtype, para)
#
# INPUT PARAMETERS
#    m - Total number of basis functions (1: Constant basis;  2: Linear basis;  3: Quadratic basis)
#    nnodes  - Total number of nodes used to construct MLS approximation
#    npoints - Total number of points whose MLS shape function to be evaluated
#    xi(nnodes) - Coordinates of nodes used to construct MLS approximation
#    x(npoints) - Coordinates of points whose MLS shape function to be evaluated
#    dm(nnodes) - Radius of support of nodes
#    wtype - Type of weight function
#    para  - Weight function parameter
#
# OUTPUT PARAMETERS
#    PHI   - MLS Shpae function
#    DPHI  - First order derivatives of MLS Shpae function
#    DDPHI - Second order derivatives of MLS Shpae function
#
def qing_1d_mls(m, nnodes, xi, npoints, x, dmI, wtype, para):
    # print('nnodes = ', nnodes, 'npoints = ',
    #       npoints, end='\n')        # 1 x nnodes
    wi = np.zeros(nnodes)
    dwi = np.zeros(nnodes)          # 1 x nnodes
    ddwi = np.zeros(nnodes)         # 1 x nnodes

    PHI = np.zeros((npoints, nnodes))        # npoints x nnodes
    DPHI = np.zeros((npoints, nnodes))       # npoints x nnodes
    DDPHI = np.zeros((npoints, nnodes))      # npoints x nnodes

    # loop over all evalutaion points to calculate value of shape function
    # FI(x)
    print()
    for j in range(0, npoints):
        # print('-------------------------------------------------------------')
        # print('j = ', j, '\tx = ', x[j])
        # print('-------------------------------------------------------------')

        # detemine weight function and their dericatives at every node
        for i in range(0, nnodes):
            di = x[j] - xi[i]
            # print('i = ', i, '\tx = ', x[j], '\txi = ', xi[
            #       i], '\tdi = ', di,  end='\n')

            # print(wtype, para, di, dmI[i])
            wi[i], dwi[i], ddwi[i] = qing_weight(wtype, para, di, dmI[i])
            # print('wi = ', wi[i], '\tdwi = ', dwi[i], '\tddwi = ', ddwi[i])
            # print('\n')

        # sys.exit()

        # evaluate basis p, B Matrix and their derivatives
        if m == 1:  # Shepard function
            p = np.ones(nnodes)    # 1 x nnodes
            p = np.reshape(p, (1, nnodes))
            px = [1]               # 1 x 1
            dpx = [0]              # 1 x 1
            ddpx = [0]             # 1 x 1

            # element multiplication
            B = p * wi
            DB = p * dwi
            DDB = p * ddwi
            pass
        elif m == 2:
            p = np.array([np.ones(nnodes), xi])     # 2 x nnodes
            p = np.reshape(p, (2, nnodes))
            px = np.array(([1], [x[j]]))            # 2 x 1
            dpx = np.array(([0], [1]))              # 2 x 1
            ddpx = np.array(([0], [0]))             # 2 x 1

            B = p * np.array(([wi], [wi]))          # 2 x 1
            DB = p * np.array(([dwi],  [dwi]))      # 2 x 1
            DDB = p * np.array(([ddwi], [ddwi]))    # 2 x 1
            pass
        elif m == 3:
            p = np.array(([np.ones(nnodes), xi, xi * xi]))    # 3 x nnodes
            p = np.reshape(p, (3, nnodes))
            px = np.array(([1], x[j], x[j] * x[j]))          # 3 x 1
            dpx = np.array(([0], [1], [2 * x[j]]))
            ddpx = np.array(([0], [0], [2]))

            B = p * np.array(([wi], [wi], [wi]))
            DB = p * np.array(([dwi], [dwi], [dwi]))
            DDB = p * np.array(([ddwi], [ddwi], [ddwi]))
            pass
        else:
            print('invalid order of basis')

        # print('DEBUG')
        # print('p : ', p, end='\n')
        # print('px : ', px, end='\n')
        # print('dpx : ', dpx, end='\n')
        # print('ddpx : ', ddpx, end='\n')
        # print('wi : ', wi, end='\n')
        # print('dwi : ', dwi, end='\n')
        # print('ddwi : ', ddwi, end='\n')
        # print('B = p .* wi: ', B, end='\n')
        # print('DB = p .* dwi ', DB, end='\n')
        # print('DDB = p .*  ddwi: ', DDB, end='\n')

        # evaluate matrices A and its derivatives
        A = np.zeros((m, m))
        DA = np.zeros((m, m))
        DDA = np.zeros((m, m))

        for i in range(0, nnodes):
            pp = np.dot(p[:, i], np.transpose(p[:, i]))

            A = A + wi[i] * pp
            DA = DA + dwi[i] * pp
            DDA = DDA + ddwi[i] * pp
            pass

        # if np.linalg.det(A):
        #     Ainv = np.linalg.inv(A)      # if A is not singular how to deal
        # else:
        #     Ainv = np.zeros((m, m))

        Ainv = np.linalg.inv(A)

        # print('A = ', A, end='\n')
        # print('invA = ', Ainv, end='\n')

        rx = np.dot(Ainv, px)
        PHI[j, :] = np.dot(np.transpose(rx), B)

        drx = np.dot(Ainv, (dpx - np.dot(DA, rx)))
        DPHI[j, :] = np.dot(np.transpose(drx), B) + \
            np.dot(np.transpose(rx), DB)

        ddrx = np.dot(Ainv, (ddpx - 2 * np.dot(DA, drx) - np.dot(DDA, rx)))
        DDPHI[j, :] = np.dot(np.transpose(
            ddrx), B) + 2 * np.dot(np.transpose(drx), DB) + np.dot(np.transpose(rx), DDB)
        pass

    return PHI, DPHI, DDPHI
    pass


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

    pass


# using mls to fit disparity data along a scanline
def qing_mls(dsp_of_testy, xmin, xmax):
    f1 = plt.figure(1)
    dx = 1

    xdata = np.array(range(xmin, xmax + 1, dx))
    ydata = dsp_of_testy[xmin:xmax + 1]
    xlen = len(xdata)
    plt.plot(xdata, ydata, 'b', label='origin')

    # print(A.shape)
    # print(B.shape)
    # qing_draw_1d_narray(dsp_of_testy, 0, length)
    #

    dx = 10
    xnode = np.array(range(xmin, xmax + 1, dx))
    ynode = dsp_of_testy[xmin:xmax + 1:10]
    plt.plot(xnode, ynode, 'r', label='node')
    # print(xnode)
    # print(ynode)

    nnodes = len(xnode)
    scale = 3
    dm = scale * dx * np.ones(nnodes)
    # print(nnodes)
    # print(scale)
    # print(dm)
    # sys.exit()

    # m, nnodes, xi, npoints, x, dmI, wtype, para
    # m = 1
    # l = 10.0
    # dx = 0.5

    # xi = np.arange(0, l, dx, dtype = float)
    # print('xi: ', end='\t')
    # print(xi)
    # nnodes = len(xi)
    # dx = dx * 0.1
    # x = np.arange(0, l, dx, dtype = float)
    # xlen = len(x)
    # print('x: ', end='\t')
    # print(x)

    # PHI, DPHI, DDPHI = qing_1d_mls(m, nnodes, xi, xlen, x, dm, 'GAUSS', 3.0)
    # sys.exit()

    plt.legend()
    plt.show()

    PHI, DPHI, DDPHI = qing_1d_mls(
        1, nnodes, xnode, xlen, xdata, dm, 'GAUSS', 3.0)

    # print('PHI shape: ', PHI.shape)
    # print('DPHI shape: ', DPHI.shape)
    # print('DDPHI shape: ', DDPHI.shape)
    fid1 = open('disp_shp.dat', 'w')
    fid2 = open('disp_dshp.dat', 'w')
    fid3 = open('disp_ddshp.dat', 'w')
    fid1.write('%10s%10s%10s%10s\n' % (' ', 'N0', 'N10', 'N20'))
    fid2.write('%10s%10s%10s%10s\n' % (' ', 'N1', 'N10', 'N20'))
    fid3.write('%10s%10s%10s%10s\n' % (' ', 'N1', 'N10', 'N20'))

    npoints = xlen
    for j in range(0, npoints):
        fid1.write('%10.4f' % xdata[j])
        fid2.write('%10.4f' % xdata[j])
        fid3.write('%10.4f' % xdata[j])
        fid1.write('%10.4f%10.4f%10.4f\n' %
                   (PHI[j][0], PHI[j][10], PHI[j][20]))
        fid2.write('%10.4f%10.4f%10.4f\n' %
                   (DPHI[j][0], DPHI[j][10], DPHI[j][20]))
        fid3.write('%10.4f%10.4f%10.4f\n' %
                   (DDPHI[j][0], DDPHI[j][10], DDPHI[j][20]))

    fid1.close()
    fid2.close()
    fid3.close()

    yhdata = np.dot(PHI, np.transpose(ynode))  # approximate function
    err = np.linalg.norm(np.transpose(ydata) - yhdata) / \
        np.linalg.norm(ydata) * 100

    fig = plt.figure(2)
    # sub1 = plt.subplot(311)
    plt.plot(xdata, ydata, xnode, ynode, xdata, yhdata)
    print('err = ', err, end='\n')

    # sub2 = plt.subplot(312)
    # sub2.plot(x, dy, x, dyh)

    # sub3 = plt.subplot(313)
    # sub3.plot(x, ddy, x, ddyh)

    plt.show()


def adaptive_mls(workdir, dspname, mskname, dsptxt):
    dspmtx = cv2.imread(dspname, 0)
    mskmtx = cv2.imread(mskname, 0)
    ret, thresh_msk = cv2.threshold(mskmtx, 75, 255, cv2.THRESH_BINARY)

    height, width = dspmtx.shape
    print('height = ', height)
    print('width = ', width)

    save_dsp_txt_name = 'disp.txt'
    qing_save_txt(dspmtx, save_dsp_txt_name)

    # cv2.imshow("dsp", dspmtx)
    # cv2.imshow("msk", thresh_msk)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # dsp_data = qing_read_txt(dsptxt)
    # qing_read_txt(dsptxt)
    read_data = np.loadtxt(save_dsp_txt_name)
    dsp_data = np.reshape(read_data, (height, width))
    # print(type(dsp_data))
    # print(dsp_data.shape)

    # testy = 500
    # xmin = 80
    # xmax = 500
    testy = 100
    xmin = 180
    xmax = 380
    dsp_of_testy = dsp_data[testy, :]    # the testy-th row

    # qing_draw_1d_narray(dsp_of_testy, xmin, xmax)
    qing_move_outliers(dsp_of_testy, xmin, xmax)
    # qing_ls(dsp_of_testy, xmin, xmax)
    # qing_curve_fit(dsp_of_testy, xmin, xmax)
    qing_mls(dsp_of_testy, xmin, xmax)
    # qing_draw_1d_narray(dsp_of_testy, xmin, xmax)


def main():
    workdir = './data/'
    imgname = workdir + 'crop_imgL_2.jpg'        # 550x950
    mskname = workdir + 'crop_mskL_2.jpg'
    dspname = workdir + 'final_disp_l_2.jpg'
    dsptxt = workdir + 'final_disp_l_2.txt'
    adaptive_mls(workdir, dspname, mskname, dsptxt)

    # qing_test_1d_mls_fitting()

if __name__ == '__main__':
    main()
