from qing_operation import *
from qing_io import *
from qing_filter import *
from qing_mls import *

from matplotlib import pyplot as plt
import cv2

# range of x-axis is generated automatically


def qing_draw_1d_narray(testy, array_data, xmin, xmax, filename):
    f1 = plt.figure(testy)
    xdata = range(xmin, xmax, 1)
    plt.plot(xdata, array_data[xmin:xmax])
    plt.xlabel('x')
    plt.ylabel('d')
    # plt.show()
    print('saving ' + filename)
    plt.savefig(filename)
    pass


def qing_move_outliers(dsp_of_testy, xmin, xmax, wnd_sz, threshold=10):
    for i in range(xmin, xmax + 1):
        if dsp_of_testy[i] <= threshold:
            dsp_of_testy[i] = 0

    # wnd_sz = 5
    qing_1d_median_filter(dsp_of_testy[xmin:xmax + 1], wnd_sz)
    pass


def qing_move_outliers_new(data, wnd_sz, threshold):
    n = len(data)
    for i in range(0, n):
        if data[i] <= threshold:
            data[i] = 0
    qing_1d_median_filter(data, wnd_sz)
    pass


# using mls to fit disparity data along a scanline
# for debug before calling in adaptive_mls
# corresponding to qing_mls_stable
# def qing_mls_beta(dsp_of_testy, xmin, xmax):
#     wnd_sz = 5
#     qing_1d_median_filter(dsp_of_testy[xmin:xmax + 1], wnd_sz)

#     dx = 1
#     xdata = np.array(range(xmin, xmax + 1, dx))
#     ydata = dsp_of_testy[xmin:xmax + 1]
#     xlen = len(xdata)

#     dx = 10 * dx
#     xnode = np.array(range(xmin, xmax + 1, dx))
#     ynode = dsp_of_testy[xmin:xmax + 1:dx]
#     nnodes = len(xnode)

#     # f1 = plt.figure(1)
#     # plt.plot(xdata, ydata, 'b', label='origin')
#     # plt.plot(xnode, ynode, 'r', label='node')
#     # plt.legend()
#     # plt.show()

#     scale = 3
#     dm = scale * dx * np.ones(nnodes)

#     PHI, DPHI, DDPHI = qing_1d_mls(
#         1, nnodes, xnode, xlen, xdata, dm, 'GAUSS', 3.0)

#     # print('PHI shape: ', PHI.shape)
#     # print('DPHI shape: ', DPHI.shape)
#     # print('DDPHI shape: ', DDPHI.shape)
#     fid1 = open('disp_shp.dat', 'w')
#     fid2 = open('disp_dshp.dat', 'w')
#     fid3 = open('disp_ddshp.dat', 'w')
#     fid1.write('%10s%10s%10s%10s\n' % (' ', 'N0', 'N10', 'N20'))
#     fid2.write('%10s%10s%10s%10s\n' % (' ', 'N1', 'N10', 'N20'))
#     fid3.write('%10s%10s%10s%10s\n' % (' ', 'N1', 'N10', 'N20'))

#     npoints = xlen
#     for j in range(0, npoints):
#         fid1.write('%10.4f' % xdata[j])
#         fid2.write('%10.4f' % xdata[j])
#         fid3.write('%10.4f' % xdata[j])
#         fid1.write('%10.4f%10.4f%10.4f\n' %
#                    (PHI[j][0], PHI[j][10], PHI[j][20]))
#         fid2.write('%10.4f%10.4f%10.4f\n' %
#                    (DPHI[j][0], DPHI[j][10], DPHI[j][20]))
#         fid3.write('%10.4f%10.4f%10.4f\n' %
#                    (DDPHI[j][0], DDPHI[j][10], DDPHI[j][20]))

#     fid1.close()
#     fid2.close()
#     fid3.close()

#     yhdata = np.dot(PHI, np.transpose(ynode))  # approximate function
#     err = np.linalg.norm(np.transpose(ydata) - yhdata) / \
#         np.linalg.norm(ydata) * 100
#     print('err = ', err, end='\n')

#     fig = plt.figure(2)
#     # sub1 = plt.subplot(311)
#     plt.plot(xdata, ydata, label='origin')
#     plt.plot(xnode, ynode, label='node')
#     plt.plot(xdata, yhdata, label='fitting')

#     # sub2 = plt.subplot(312)
#     # sub2.plot(x, dy, x, dyh)
#     # sub3 = plt.subplot(313)
#     # sub3.plot(x, ddy, x, ddyh)

#     plt.legend()
#     plt.show()

#     qing_save_1d_txt(dsp_of_testy[xmin:xmax + 1], 'dsp_before_mls.txt')
#     dsp_of_testy[xmin:xmax + 1] = yhdata[0:xlen]
#     qing_save_1d_txt(dsp_of_testy[xmin:xmax + 1], 'dsp_after_mls.txt')

#     # test
#     test_a = np.loadtxt('dsp_before_mls.txt')
#     test_b = np.loadtxt('dsp_after_mls.txt')
#     test_len = len(test_a)
#     test_x = np.arange(0, test_len, 1)
#     print('test_len = ', test_len)

#     fig = plt.figure(3)
#     plt.plot(test_x, test_a, label='before')
#     plt.plot(test_x, test_b, label='after')
#     plt.legend()
#     plt.show()


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
    pass

    # sys.exit()


# a stable version of mls along a scanline
def qing_mls_1d_stable(dsp_of_testy, xmin, xmax):
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
    PHI, DPHI, DDPHI = qing_1d_mls(
        1, nnodes, xnode, ndatas, xdata, dm, 'GAUSS', 3.0)
    print(
        'end of moving least square fitting in [%d - %d].' % (xmin, xmax), end='\t')

    fit_ydata = np.dot(PHI, np.transpose(ynode))  # approximate function
    err = np.linalg.norm(np.transpose(ydata) - fit_ydata) / \
        np.linalg.norm(ydata) * 100
    print('err = ', err, end='\n')

    dsp_of_testy[xmin:xmax + 1] = fit_ydata[0:ndatas]  # copy
    pass


def qing_rewrite_dsp(workdir, imgname, dspname):
    f_dspname = workdir + dspname
    f_imgname = workdir + imgname
    dspdatas = qing_read_txt(f_dspname)
    dspdatas_1d = []
    rows = len(dspdatas)
    for r in range(rows):
        cols = len(dspdatas[r])
        for c in range(cols):
            dspdatas_1d.append(dspdatas[r][c])
    # print('size = ', len(dspdatas_1d))

    imgmtx = qing_read_img(f_imgname)
    height, width = imgmtx.shape
    new_f_dspname = workdir + 'rewrite_' + dspname
    writer = open(new_f_dspname, 'w')
    for idx, d in enumerate(dspdatas_1d):
        writer.write('%f' % d)
        if (int(idx + 1)) % width == 0:
            writer.write('\n')
        else:
            writer.write(' ')
            pass
    writer.close()
    return new_f_dspname


def qing_read_stereo_txt(txtname):
    st_0_x = 1000
    st_0_y = 880
    st_1_x = 700
    st_1_y = 880
    data = np.loadtxt(txtname)
    stereo_mtx = np.reshape(data, (4, 4))

    return st_0_x, st_0_y, (st_0_x - st_1_x), stereo_mtx
    pass


def adaptive_mls_1d(workdir, dspname, mskname, dsptxt):
    dspmtx = cv2.imread(dspname, 0)
    mskmtx = cv2.imread(mskname, 0)
    ret, thresh_msk = cv2.threshold(mskmtx, 75, 255, cv2.THRESH_BINARY)
    # cv2.imshow("dsp", dspmtx)
    # cv2.imshow("msk", thresh_msk)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    height, width = dspmtx.shape
    print('height = %d\twidth = %d\n' % (height, width))

    # qing_save_2d_txt(dspmtx, save_dsp_txt_name)
    save_dsp_txt_name = qing_rewrite_dsp(workdir, dspname, dsptxt)
    # type: ndarray
    read_data = np.loadtxt(save_dsp_txt_name)
    dsp_data = np.reshape(read_data, (height, width))

    outdir = './output'
    qing_mkdir(outdir)

    # testy = 500
    # xmin = 80
    # xmax = 500
    # test_msk = np.array([0,255,0, 255,255,255])
    # print(test_msk)
    # t_segments, t_xmin, t_xmax = qing_get_msk_segments(test_msk)
    # sys.exit()

    # 1d_mls along x-direction
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
            qing_mls_1d_stable(dsp_of_testy, i_xmin, i_xmax)

        pngname = outdir + '/mls_disp_' + str(testy) + '.png'
        qing_draw_1d_narray(testy, dsp_of_testy, int(
            xmin[0]), int(xmax[segments - 1]), pngname)

        # break

    filename = outdir + '/mls_disp_x.txt'  # mls result along x-direction
    print('saving ' + filename + ' in float format.', end='\n')
    qing_save_2d_txt(dsp_data, filename, '%f')

    # 1d_mls along y-direction
    for testx in range(0, width):
        segments = 10
        ymin = np.zeros(segments)
        ymax = np.zeros(segments)
        dsp_of_testx = dsp_data[:, testx]
        msk_of_testx = thresh_msk[:, testx]
        # print('dsp_of_testx: ', dsp_of_testx.shape)
        # print(dsp_of_testx)
        # print('dsp_of_testx: ', msk_of_testx.shape)
        # print(msk_of_testx)

        segments, ymin, ymax = qing_get_msk_segments(msk_of_testx)
        print('segments = ', segments, end='\n')
        for j in range(0, segments):
            print('%d-th seg: [%d, %d]' % (j, ymin[j], ymax[j]), end='\n')

        # pngname = outdir + '/init_dsp_x_' + str(testx) + '.png'
        # qing_draw_1d_narray(testx, dsp_of_testx, int(
        #     ymin[0]), int(ymax[segments - 1]), pngname)

        # sys.exit()
        for j in range(0, segments):
            i_ymin = int(ymin[j])
            i_ymax = int(ymax[j])
            qing_move_outliers_new(dsp_of_testx[i_ymin:i_ymax + 1], 5, 10)

        # pngname = outdir + '/mf_dsp_x_' + str(testx) + '.png'
        # qing_draw_1d_narray(testx, dsp_of_testx, int(
        #     ymin[0]), int(ymax[segments - 1]), pngname)

        for j in range(0, segments):
            i_ymin = int(ymin[j])
            i_ymax = int(ymax[j])
            qing_mls_1d_stable(dsp_of_testx, i_ymin, i_ymax)

        pngname = outdir + '/mls_disp_x_' + str(testx) + '.png'
        qing_draw_1d_narray(testx, dsp_of_testx, int(
            ymin[0]), int(ymax[segments - 1]), pngname)

    filename = outdir + '/mls_disp_y_after_x.txt'  # mls result along x-direction
    print('saving ' + filename + ' in float format.', end='\n')
    qing_save_2d_txt(dsp_data, filename, '%f')

    return filename, dsp_data
    pass


def qing_dsp_to_depth(dsp, thresh_msk, imgmtx, stereo_mtx, st_x, st_y, base_d, scale, plyname):

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
            cnt += 1

    print('saving ' + plyname)
    qing_save_ply(plyname, pointcnt, points, colors)
    pass


def main():
    # from qing_msl1d_in_python import *
    # qing_test_1d_mls_fitting()

    workdir = './test/'
    imgname = workdir + 'crop_imgL_2.jpg'        # 550x950
    mskname = workdir + 'crop_mskL_2.jpg'
    dspname = workdir + 'final_disp_l_2.jpg'
    dsptxt = workdir + 'final_disp_l_2.txt'
    stereotxt = workdir + 'stereo.txt'

    # about image and mask data
    mskmtx = cv2.imread(mskname, 0)
    ret, thresh_msk = cv2.threshold(mskmtx, 75, 255, cv2.THRESH_BINARY)
    print(type(thresh_msk), thresh_msk.shape)
    imgmtx = cv2.imread(imgname, 1)
    print(type(imgmtx), imgmtx.shape)

    # about calibration data
    st_x, st_y, base_d, stereo_mtx = qing_read_stereo_txt(stereotxt)
    # print( st_x, st_y, base_d, stereo_mtx)
    scale = 4
    f_st_x = st_x * 1.0 / scale
    f_st_y = st_y * 1.0 / scale
    f_base_d = base_d * 1.0 / scale
    stereo_mtx[0, 3] /= scale
    stereo_mtx[1, 3] /= scale
    stereo_mtx[2, 3] /= scale
    print('scaled stereo_mtx: ', end='\n')
    print(stereo_mtx)
    sys.exit()

    # ply_name = workdir + 'init.ply'
    # dsp_name = qing_rewrite_dsp(workdir, dsptxt, dspname)
    # init_dsp_data = qing_read_dsp_txt(dsp_name)
    # sys.exit()
    # dsp_to_depth(init_dsp_data, thresh_msk, imgmtx, stereo_mtx,
    #              f_st_x, f_st_y, f_base_d, scale, ply_name)
    # sys.exit()

    output_dsp_txt, dsp_data = adaptive_mls_1d(
        workdir, dspname, mskname, dsptxt)
    ply_name = workdir + 'mls_y_after_x.ply'
    pointcnt, points, colors = qing_dsp_to_depth(dsp_data, thresh_msk, imgmtx, stereo_mtx,
                      f_st_x, f_st_y, f_base_d, scale)
    qing_save_ply(plyname, pointcnt, points, colors)
    print('saving ', plyname, '\t%d points.'%(pointcnt))


if __name__ == '__main__':
    main()
