import numpy as np

# data_in: ndarray
# wnd_sz: filter window size


def qing_1d_median_filter(data_in, wnd_sz):
    rangex = len(data_in)
    offset = int(wnd_sz * 0.5)
    dmax = data_in.max()
    dmin = data_in.min()
    # print('dmax = %d' % dmax, 'dmin = %d' % dmin, 'wnd_sz = %d' % wnd_sz)

    for x in range(0, rangex):
        dhist = np.zeros((int(dmax - dmin + 1), 1))
        for j in range(-offset, offset + 1):
            xj = min(rangex - 1, x + j)
            xj = max(0, xj)
            d = int(data_in[xj])
            if not d == 0:
                idx = int(d - dmin)
                dhist[idx] += 1

        count = 0
        middle = 0
        for j in range(0, len(dhist)):
            count += dhist[j]
            if count * 2 > wnd_sz:
                middle = j
                break
            pass

        data_in[x] = middle + dmin
        pass
