import numpy as np


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
    return numbers_float
    pass


def qing_save_txt(mtx, txtname, format='%d'):
    np.savetxt(txtname, mtx[:, :], fmt=format)
    print('saving ' + txtname)
    pass


def qing_save_1d_txt(mtx, txtname):
    np.savetxt(txtname, mtx[:], fmt="%f")
    print('saving ' + txtname)
    pass


def qing_save_ply(plyname, pointcnt, points, colors):
    fobj = open(plyname, 'w')
    fobj.write('ply\n')
    fobj.write('format ascii 1.0\n')
    fobj.write('element vertex %d\n' % (pointcnt))
    fobj.write('property float x\n')
    fobj.write('property float y\n')
    fobj.write('property float z\n')
    fobj.write('property uchar red\n')
    fobj.write('property uchar green\n')
    fobj.write('property uchar blue\n')
    fobj.write('end_header\n')

    for i in range(0, pointcnt):
        fobj.write('%f %f %f %d %d %d\n' % (points[i, 0], points[i, 1], points[
                   i, 2], colors[i, 0], colors[i, 1], colors[i, 2]))
    fobj.close()
    pass
