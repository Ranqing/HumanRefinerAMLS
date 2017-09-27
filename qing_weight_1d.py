import numpy as np

# re-write from MLS1D/Weight.m
# INPUT PARAMETERS
#    type - Type of weight function
#    para - Weight function parameter
#    di   - Distance
#    dmi  - Support size
# OUTPUT PARAMETERS
#    w    - Value of weight function at r
#    dwdx - Value of first order derivative of weight function with respect to x at r
#    dwdxx- Value of Second order derivative of weight function with respect to x at r
#


def Gauss(beta, r):
    # print('Gauss:\tbeta = ', beta, '\tr = ',  r, end = '\t')
    w = 0.0
    dwdr = 0.0
    dwdrr = 0.0

    if r <= 1.0:
        b2 = beta * beta
        r2 = r * r
        eb2 = np.exp(-b2)

        w = (np.exp(-b2 * r2) - eb2) / (1.0 - eb2)
        dwdr = -2 * b2 * r * np.exp(-b2 * r2) / (1.0 - eb2)
        dwdrr = -2 * b2 * np.exp(-b2 * r2) * (1 - 2 * b2 * r2) / (1.0 - eb2)

    # print(' --> \tw = %f' % w, 'dwdr = %f' % dwdr, 'dwdrr = %f' % dwdrr)
    return w, dwdr, dwdrr
    pass


def Cubic(r):
    w = 0.0
    dwdr = 0.0
    dwdrr = 0.0

    if r <= 1.0:
        w = 1 - 6 * r**2 + 8 * r**3 - 3 * r**4
        dwdr = -12 * r + 24 * r**2 - 12 * r**3
        dwdrr = -12 + 48 * r - 36 * r**2

    print('Cubic: w = %f' % w, 'dwdr = %f' % dwdr, 'dwdrr = %f' % dwdrr)
    return w, dwdr, dwdrr
    pass


def power_function(arfa, r):
    w = 0.0
    dwdr = 0.0
    dwdrr = 0.0

    if r <= 1.0:
        a2 = arfa * arfa
        r2 = r * r
        w = np.exp(-r2 / a2)
        dwdr = (-2 * r / a2) * exp(-r2 / a2)
        dwdrr = (-2 / a2 + (-2 * r / a2) ** 2) * np.exp(-r2 / a2)

    print('power_func: w = %f' % w, 'dwdr = %f' % dwdr, 'dwdrr = %f' % dwdrr)
    return w, dwdr, dwdrr
    pass


def Spline(r):
    w = 0.0
    dwdr = 0.0
    dwdrr = 0.0

    if r <= 0.5:
        w = 2 / 3 - 4 * r**2 + 4 * r**3
        dwdr = -8 * r + 12 * r**2
        dwdrr = -8 + 24 * r
    elif r > 0.5 and r <= 1.0:
        w = 4 / 3 - 4 * r + 4 * r**2 - 4 * r**3 / 3
        dwdr = -4 + 8 * r - 4 * r**2
        dwdrr = 8 - 8 * r

    print('spline: w = %f' % w, 'dwdr = %f' % dwdr, 'dwdrr = %f' % dwdrr)
    return w, dwdr, dwdrr
    pass


def CSRBF2(r):
    w = 0.0
    dwdr = 0.0
    dwdrr = 0.0

    if r <= 1.0:
        w = (1 - r) ^ 6 * (6 + 36 * r + 82 * r **
                           2 + 72 * r**3 + 30 * r**4 + 5 * r**5)
        dwdr = 11 * r * (r + 2) * (5 * r**3 + 15 * r **
                                   2 + 18 * r + 4) * (r - 1)**5
        dwdrr = 22 * (25 * r ^ 5 + 100 * r**4 + 142 * r **
                      3 + 68 * r**2 - 16 * r - 4) * (r - 1)**4

    print('CSRBF2: w = %f' % w, 'dwdr = %f' % dwdr, 'dwdrr = %f' % dwdrr)
    return w, dwdr, dwdrr
    pass


def qing_weight_1d(type, para, di, dmI):

    r = abs(di) / dmI
    if di >= 0.0:
        drdx = 1.0 / dmI
    else:
        drdx = -1.0 / dmI

    # print('r = ',r, end = '\t')
    # print('drdx = ', drdx, end = '\n')

    w = 0.0
    dwdr = 0.0
    dwdrr = 0.0

    # evaluate weight function and its first and second order of derivatives
    # with respect r at r
    if type == 'GAUSS':
        w, dwdr, dwdrr = Gauss(para, r)
    elif type == 'CUBIC':
        w, dwdr, dwdrr = Cubic(r)
    elif type == 'SPLIN':
        w, dwdr, dwdrr = Spline(r)
    elif type == 'power':
        w, dwdr, dwdrr = power_function(para, r)
    elif type == 'CSRBF':
        w, dwdr, dwdrr = CSRBF2(r)
    else:
        print('Invalid type of weight function...')
        pass

    dwdx = dwdr * drdx
    dwdxx = dwdrr * drdx * drdx
    return w, dwdx, dwdxx
    pass


def main():
    pass

if __name__ == '__main__':
    main()
