import numpy as np


def qing_read_2d_txt(txtname):
    list_of_lists = []
    with open(txtname, 'r') as f:
        data = f.readlines()
        # print('number of lines = %d'%(len(data)))
        for line in data:
            odom = line.split()
            numbers_float = list(map(float, odom))
            list_of_lists.append(numbers_float)
            # print(numbers_float)
        # cnt = len(list_of_lists)
        # print('total = %d' % (cnt))
    return list_of_lists
    pass


def qing_save_2d_txt(mtx, txtname, format='%d'):
    np.savetxt(txtname, mtx[:, :], fmt=format)
    print('saving ' + txtname)
    pass


def qing_save_1d_txt(mtx, txtname):
    np.savetxt(txtname, mtx[:], fmt="%f")
    print('saving ' + txtname)
    pass


def qing_read_img(imgname):
    imgmtx = cv2.imread(imgname, 0)
    if imgmtx is None:
        print(imgname, ' is not exist.', end='\n')
        sys.exit()
    return imgmtx


def qing_init_2d_array(w, h):
    matrix = [[float(0) for x in range(w)] for y in range(h)]
    return matrix


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
