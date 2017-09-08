from qing_operation import *
from qing_weight import *
from qing_io import *
from qing_filter import *
from matplotlib import pyplot as plt
import cv2
import numpy as np
import scipy.optimize as optimization


# range of x-axis is generated automatically
def qing_draw_1d_narray(testy, array_data, xmin, xmax, filename):
    # f1 = plt.figure(figsize=(100, 100))
    f1 = plt.figure(testy)
    xdata = range(xmin, xmax, 1)
    plt.plot(xdata, array_data[xmin:xmax])
    plt.xlabel('x')
    plt.ylabel('d')
    # plt.show()
    print('saving ' + filename)
    plt.savefig(filename)


def qing_move_outliers(dsp_of_testy, xmin, xmax, wnd_sz, threshold=10):
    for i in range(xmin, xmax + 1):
        if dsp_of_testy[i] <= threshold:
            dsp_of_testy[i] = 0

    # wnd_sz = 5
    qing_1d_median_filter(dsp_of_testy[xmin:xmax + 1], wnd_sz)


def qing_move_outliers_new(data, wnd_sz, threshold):
    n = len(data)
    # print ('n = ', n)
    for i in range(0, n):
        if data[i] <= threshold:
            data[i] = 0
    qing_1d_median_filter(data, wnd_sz)


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

# using mls to fit disparity data along a scanline


def qing_mls(dsp_of_testy, xmin, xmax):
    wnd_sz = 5
    qing_1d_median_filter(dsp_of_testy[xmin:xmax + 1], wnd_sz)

    dx = 1
    xdata = np.array(range(xmin, xmax + 1, dx))
    ydata = dsp_of_testy[xmin:xmax + 1]
    xlen = len(xdata)

    # print(A.shape)
    # print(B.shape)
    # qing_draw_1d_narray(dsp_of_testy, 0, length)
    #

    dx = 10 * dx
    xnode = np.array(range(xmin, xmax + 1, dx))
    ynode = dsp_of_testy[xmin:xmax + 1:dx]
    nnodes = len(xnode)

    f1 = plt.figure(1)
    plt.plot(xdata, ydata, 'b', label='origin')
    plt.plot(xnode, ynode, 'r', label='node')
    # print(xnode)
    # print(ynode)
    plt.legend()
    plt.show()

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
    print('err = ', err, end='\n')

    fig = plt.figure(2)
    # sub1 = plt.subplot(311)
    plt.plot(xdata, ydata, label='origin')
    plt.plot(xnode, ynode, label='node')
    plt.plot(xdata, yhdata, label='fitting')

    # sub2 = plt.subplot(312)
    # sub2.plot(x, dy, x, dyh)
    # sub3 = plt.subplot(313)
    # sub3.plot(x, ddy, x, ddyh)

    plt.legend()
    plt.show()

    qing_save_1d_txt(dsp_of_testy[xmin:xmax + 1], 'dsp_before_mls.txt')
    dsp_of_testy[xmin:xmax + 1] = yhdata[0:xlen]
    qing_save_1d_txt(dsp_of_testy[xmin:xmax + 1], 'dsp_after_mls.txt')

    # test
    test_a = np.loadtxt('dsp_before_mls.txt')
    test_b = np.loadtxt('dsp_after_mls.txt')
    test_len = len(test_a)
    test_x = np.arange(0, test_len, 1)
    print('test_len = ', test_len)

    fig = plt.figure(3)
    plt.plot(test_x, test_a, label='before')
    plt.plot(test_x, test_b, label='after')
    plt.legend()
    plt.show()


def qing_get_msk_segments(msk_of_testy):
    maxvalue = len(msk_of_testy)
    xmin = np.zeros(maxvalue)
    xmax = np.zeros(maxvalue)

    pre_i = -1
    i = 0
    segments = 0

    while i < maxvalue:
        if pre_i == -1:
            if msk_of_testy[i] == 255:
                xmin[segments] = i
                xmax[segments] = i
                segments += 1

                pass
            pass
        else:
            if msk_of_testy[pre_i] == 0 and msk_of_testy[i] == 255:
                xmin[segments] = i
                xmax[segments] = i
                segments += 1

                pass

            if msk_of_testy[pre_i] == 255 and msk_of_testy[i] == 0:
                xmax[segments - 1] = pre_i
                pass

            if i == maxvalue - 1 and msk_of_testy[i] == 255:
                xmax[segments - 1] = i

                pass
            pass

        pre_i += 1
        i += 1
        pass

    # print('segments = ', segments, end = '\n')
    # for j in range(0, segments):
    #     print('%d-th seg: [%d, %d] - '%(j, xmin[j], xmax[j]), end = '\t')
    #     print(msk_of_testy[int(xmin[j]):int(xmax[j]+1)], end = '\n')

    return segments, xmin, xmax

    # sys.exit()


def qing_mls_stable(dsp_of_testy, xmin, xmax):
    # wnd_sz = 5
    # qing_1d_median_filter(dsp_of_testy[xmin:xmax + 1], wnd_sz)

    dx = 1
    xdata = np.array(range(xmin, xmax + 1, dx))
    ydata = dsp_of_testy[xmin:xmax + 1:dx]
    ndatas = len(xdata)

    dx = 10 * dx
    xnode = np.array(range(xmin, xmax + 1, dx))
    ynode = dsp_of_testy[xmin:xmax + 1:dx]
    nnodes = len(xnode)

    m = 1
    scale = 3
    dm = scale * dx * np.ones(nnodes)

    #1, nnodes, xnode, xlen, xdata, dm, 'GAUSS', 3.0
    PHI, DPHI, DDPHI = qing_1d_mls(1, nnodes, xnode, ndatas, xdata, dm, 'GAUSS', 3.0)
    print('end of moving least square fitting in [%d - %d].' % (xmin, xmax), end='\t')

    fit_ydata = np.dot(PHI, np.transpose(ynode))  # approximate function
    err = np.linalg.norm(np.transpose(ydata) - fit_ydata) / np.linalg.norm(ydata) * 100
    print('err = ', err, end='\n')

    dsp_of_testy[xmin:xmax + 1] = fit_ydata[0:ndatas]  # copy


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
    outdir = './output'
    qing_mkdir(outdir)

    # test_msk = np.array([0,255,0, 255,255,255])
    # print(test_msk)
    # t_segments, t_xmin, t_xmax = qing_get_msk_segments(test_msk)
    # sys.exit()

    for testy in range(0, height):
        # testy = 200
        segments = 10
        xmin = np.zeros(segments)
        xmax = np.zeros(segments)
        dsp_of_testy = dsp_data[testy, :]  # slicing, just reference
        msk_of_testy = thresh_msk[testy, :]

        segments, xmin, xmax = qing_get_msk_segments(msk_of_testy)
        print('segments = ', segments, end='\n')
        for j in range(0, segments):
            print('%d-th seg: [%d, %d]' % (j, xmin[j], xmax[j]), end='\n')

        # when dsp_of_testy changes , then dsp_data changes as well
        pngname = outdir + '/init_dsp_' + str(testy) + '.png'
        qing_draw_1d_narray(testy, dsp_of_testy, int(
            xmin[0]), int(xmax[segments - 1]), pngname)

        for j in range(0, segments):
            i_xmin = int(xmin[j])
            i_xmax = int(xmax[j])
            qing_move_outliers_new(dsp_of_testy[i_xmin:i_xmax + 1], 5, 10)

        pngname = outdir + '/mf_dsp_' + str(testy) + '.png'
        qing_draw_1d_narray(testy, dsp_of_testy, int(
            xmin[0]), int(xmax[segments - 1]), pngname)

        # sys.exit()

        for j in range(0, segments):
            i_xmin = int(xmin[j])
            i_xmax = int(xmax[j])
            qing_mls_stable(dsp_of_testy, i_xmin, i_xmax)

        pngname = outdir + '/mls_disp_' + str(testy) + '.png'
        qing_draw_1d_narray(testy, dsp_of_testy, int(
            xmin[0]), int(xmax[segments - 1]), pngname)

        # break

    filename = outdir + '/mls_disp_x.txt'  # mls result along x-direction
    print('saving ' + filename + ' in float format.', end='\n')
    qing_save_txt(dsp_data, filename, '%f')

    return filename, dsp_data
    # above

    # test codes
    # testy = 100
    # xmin = 180
    # xmax = 380
    # dsp_of_testy = dsp_data[testy, :]    # the testy-th row, just refernce

    # # qing_draw_1d_narray(dsp_of_testy, xmin, xmax)
    # # qing_move_outliers(dsp_of_testy, xmin, xmax)
    # # qing_ls(dsp_of_testy, xmin, xmax)
    # # qing_curve_fit(dsp_of_testy, xmin, xmax)
    # qing_draw_1d_narray(testy, dsp_of_testy, xmin, xmax, 'test_disp_before.png')
    # qing_mls(dsp_of_testy, xmin, xmax)
    # # qing_mls_stable(dsp_of_testy, xmin, xmax)
    # print('\n')
    # qing_draw_1d_narray(testy, dsp_of_testy, xmin, xmax, 'test_disp_after.png')
    # qing_draw_1d_narray(dsp_of_testy, xmin, xmax)


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


def qing_read_dsp_txt(txtname, dspname=''):
    if dspname == '':
        dsp_data = np.loadtxt(txtname)
        print(dsp_data.shape)
        return dsp_data

    dspmtx = cv2.imread(dspname, 0)
    height, width = dspmtx.shape
    # print('height = ', height)
    # print('width = ', width)

    save_dsp_txt_name = 'disp.txt'
    qing_save_txt(dspmtx, save_dsp_txt_name)
    read_data = np.loadtxt(save_dsp_txt_name)
    dsp_data = np.reshape(read_data, (height, width))
    return dsp_data


def qing_read_stereo_txt(txtname):
    st_0_x = 1000
    st_0_y = 880
    st_1_x = 700
    st_1_y = 880
    data = np.loadtxt(txtname)
    stereo_mtx = np.reshape(data, (4, 4))

    return st_0_x, st_0_y, (st_0_x - st_1_x), stereo_mtx
    pass


def dsp_to_depth(dsp, thresh_msk, imgmtx, stereo_mtx, st_x, st_y, base_d, scale, plyname):

    height, width = dsp.shape
    pointcnt = 0

    for y in range(0, height):
        for x in range(0, width):
            if thresh_msk[y, x] == 0:
                dsp[y, x] = 0.
                continue
            dsp[y, x] += base_d
            pointcnt += 1

    print('%d points generated!' % (pointcnt), end='\n')

    points = np.ndarray((pointcnt, 3))
    print(points.shape)
    colors = np.ndarray((pointcnt, 3))
    print(colors.shape)
    uvd1 = np.zeros(4)
    print(uvd1.shape)
    xyzw = np.zeros(4)
    print(xyzw.shape)
    cnt = 0
    for y in range(0, height):
        for x in range(0, width):
            if thresh_msk[y, x] == 0:
                continue

            uvd1[0] = x + st_x
            uvd1[1] = y + st_y
            uvd1[2] = dsp[y, x]
            uvd1[3] = 1.0

            xyzw = np.dot(stereo_mtx, uvd1)

            # xyzw[0] = qmtx[ 0] * uvd1[0] + qmtx[ 1] * uvd1[1] + qmtx[ 2] * uvd1[2] + qmtx[ 3] * uvd1[3];
            # xyzw[1] = qmtx[ 4] * uvd1[0] + qmtx[ 5] * uvd1[1] + qmtx[ 6] * uvd1[2] + qmtx[ 7] * uvd1[3];
            # xyzw[2] = qmtx[ 8] * uvd1[0] + qmtx[ 9] * uvd1[1] + qmtx[10] * uvd1[2] + qmtx[11] * uvd1[3];
            # xyzw[3] = qmtx[12] * uvd1[0] + qmtx[13] * uvd1[1] + qmtx[14] * uvd1[2] + qmtx[15] * uvd1[3];

            points[cnt, 0] = xyzw[0] / xyzw[3]
            points[cnt, 1] = xyzw[1] / xyzw[3]
            points[cnt, 2] = xyzw[2] / xyzw[3]
            colors[cnt, 0] = imgmtx[y, x, 2]
            colors[cnt, 1] = imgmtx[y, x, 1]
            colors[cnt, 2] = imgmtx[y, x, 0]
            cnt+=1


    print('saving ' + plyname)
    qing_save_ply(plyname, pointcnt, points, colors)
    pass


def main():
    workdir = './test/'
    imgname = workdir + 'crop_imgL_2.jpg'        # 550x950
    mskname = workdir + 'crop_mskL_2.jpg'
    dspname = workdir + 'final_disp_l_2.jpg'
    dsptxt = workdir + 'final_disp_l_2.txt'
    stereotxt = workdir + 'stereo.txt'

    ply_name = workdir + 'init.ply'
    init_dsp_data = qing_read_dsp_txt(dsptxt, dspname)
    # sys.exit()

    st_x, st_y, base_d, stereo_mtx = qing_read_stereo_txt(stereotxt)
    # print( st_x, st_y, base_d, stereo_mtx)

    mskmtx = cv2.imread(mskname, 0)
    ret, thresh_msk = cv2.threshold(mskmtx, 75, 255, cv2.THRESH_BINARY)
    print(type(thresh_msk), thresh_msk.shape)
    imgmtx = cv2.imread(imgname, 1)
    print(type(imgmtx), imgmtx.shape)

    scale = 4
    f_st_x = st_x * 1.0 / scale
    f_st_y = st_y * 1.0 / scale
    f_base_d = base_d * 1.0 / scale
    stereo_mtx[0, 3] /= scale
    stereo_mtx[1, 3] /= scale
    stereo_mtx[2, 3] /= scale
    print('scaled stereo_mtx: ', end='\n')
    print(stereo_mtx)

    # dsp_to_depth(init_dsp_data, thresh_msk, imgmtx, stereo_mtx,
    #              f_st_x, f_st_y, f_base_d, scale, ply_name)
    # sys.exit()

    output_dsp_txt, dsp_data = adaptive_mls(workdir, dspname, mskname, dsptxt)

    ply_name = workdir + 'mls.ply'
    dsp_to_depth(dsp_data, thresh_msk, imgmtx, stereo_mtx,
                 f_st_x, f_st_y, f_base_d, scale, ply_name)

    # qing_test_1d_mls_fitting()

if __name__ == '__main__':
    main()
