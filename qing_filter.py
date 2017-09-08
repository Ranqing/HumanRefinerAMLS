import numpy as np


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
                idx = int(d - dmin)
                dhist[idx] += 1

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
