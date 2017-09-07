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


def qing_save_txt(mtx, txtname):
    np.savetxt(txtname, mtx[:, :], fmt="%d")
    print('saving ' + txtname)
