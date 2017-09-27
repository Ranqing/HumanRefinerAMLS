import numpy as np


def Gauss(beta, r):
    w = 0.0
    dwdr = 0.0

    if r <= 1.0:
        b2 = beta * beta
        r2 = r * r
        eb2 = np.exp(-b2)

        w = (np.exp(-b2 * r2) - eb2) / (1.0 - eb2)
        dwdr = -2 * b2 * r * np.exp(-b2 * r2) / (1.0 - eb2)

    return w, dwdr


def Cubic(r):
    w = 0.0
    dwdr = 0.0

    if r <= 1.0:
        w = 1 - 6 * r**2 + 8 * r**3 - 3 * r**4
        dwdr = -12 * r + 24 * r**2 - 12 * r**3

    return w, dwdr


def Spline3(r):
    w = 0.0
    dwdr = 0.0

    if r <= 0.5:
        w = 2 / 3 - 4 * r**2 + 4 * r**3
        dwdr = -8 * r + 12 * r**2
    elif r > 0.5 and r <= 1.0:
        w = 4 / 3 - 4 * r + 4 * r**2 - 4 * r**3 / 3
        dwdr = -4 + 8 * r - 4 * r**2

    return w, dwdr


def Spline5(r):
    w = 0.0
    dwdr = 0.0

    if r <= 1.0:
        w = 1 - 10 * r**3 + 15 * r**4 - 6 * r**5
        dwdr = -30 * r**2 + 60 * r**3 - 30 * r**4

    return w, dwdr


def power_function(arfa, r):
    w = 0.0
    dwdr = 0.0

    if r <= 1.0:
        a2 = arfa * arfa
        r2 = r * r
        w = np.exp(-r2 / a2)
        dwdr = (-2 * r / a2) * np.exp(-r2 / a2)

    return w, dwdr


def CSRBF2(r):
    w = 0.0
    dwdr = 0.0

    if r <= 1.0:
        w = (1 - r)**6 * (6 + 36 * r + 82 * r **
                          2 + 72 * r**3 + 30 * r**4 + 5 * r**5)
        dwdr = 11 * r * (r + 2) * (5 * r**3 + 15 * r **
                                   2 + 18 * r + 4) * (r - 1)**5

    return w, dwdr


def CSRBF1(r):
    w = 0.0
    dwdr = 0.0

    if r <= 1.0:
        w = (1 - r)**4 * (4 + 16 * r + 12 * r**2 + 3 * r**3)
        dwdr = -4 * (1 - r)**3 * (4 + 16 * r + 12 * r**2 + 3 *
                                  r**3) + (1 - r)**4 * (16 + 24 * r + 9 * r**2)

    return w, dwdr


def CSRBF3(r):
    w = 0.0
    dwdr = 0.0

    return w, dwdr


def CSRBF4(r):
    w = 0.0
    dwdr = 0.0

    return w, dwdr


def CSRBF5(r):
    w = 0.0
    dwdr = 0.0

    return w, dwdr


# from qing_weight_2d import *

# EVALUATE WEIGHT FUNCTION
#
# SYNTAX:[w, dwdx, dwdy] = rectangleWeight(type, para, x,y,xI,yI,dmIx,dmIy)
#
# INPUT PARAMETERS
#    type - Type of weight function
#    para - Weight function parameter
#    x,y   - gauss point coordinates
#    xI,yI  -  nodal point coordinate
#    dmIx - Support size with respect to x direction
#    dmIy - Support size with respect to y direction
# OUTPUT PARAMETERS
#    w    - Value of weight function at r
#    dwdx - Value of first order derivative of weight function with respect to x at r
# dwdy - Value of first order derivative of weight function with respect
# to y at r


def rectangle_weight(wtype, para, x, y, xI, yI, dmIx, dmIy):
    # print('wtype = ', wtype, end='\t')
    # print('para = ', para, end='\t')
    # print('x = ', x, end='\t')
    # print('y = ', y, end='\t')
    # print('xI = ', xI, end='\t')
    # print('yI = ', yI, end='\t')
    # print('dmIx = ', dmIx, end='\t')
    # print('dmIy = ', dmIy, end='\n')
    # return 0, 0, 0
    # define the support size is a rectangle
    rx = np.abs(x - xI) / dmIx
    ry = np.abs(y - yI) / dmIy

    # symbol of derivatives
    if rx == 0:
        drdx = 0
    elif x - xI > 0:
        drdx = 1 / dmIx
    elif x - xI < 0:
        drdx = -1 / dmIx

    if ry == 0:
        drdy = 0
    elif y - yI > 0:
        drdy = 1 / dmIy
    elif y - yI < 0:
        drdy = -1 / dmIy

    # EVALUATE WEIGHT FUNCTION AND ITS FIRST AND SECOND ORDER OF DERIVATIVES
    # WITH RESPECT r AT r
    if wtype == 'GAUSS':
        wx, dwdrx = Gauss(para, rx)
        wy, dwdry = Gauss(para, ry)
        # print('HERE, wx = %f, dwdrx = %f'%(wx, dwdrx), end = '\t')
        # print('wy = %f, dwdry = %f'%(wy, dwdry), end = '\n')
    elif wtype == 'CUBIC':
        wx, dwdrx = Cubic(rx)
        wy, dwdry = Cubic(ry)
    elif wtype == 'SPLI3':
        wx, dwdrx = Spline3(rx)
        wy, dwdry = Spline3(ry)
    elif wtype == 'SPLI5':
        wx, dwdrx = Spline5(rx)
        wy, dwdry = Spline5(ry)
    elif wtype == 'power':
        wx, dwdrx = power_function(para, rx)
        wy, dwdry = power_function(para, ry)
    elif wtype == 'CRBF1':
        wx, dwdrx = CSRBF1(rx)
        wy, dwdry = CSRBF1(ry)
    elif wtype == 'CRBF2':
        wx, dwdrx = CSRBF2(rx)
        wy, dwdry = CSRBF2(ry)
    elif wtype == 'CRBF3':
        wx, dwdrx = CSRBF3(rx)
        wy, dwdry = CSRBF3(ry)
    elif wtype == 'CRBF4':
        wx, dwdrx = CSRBF4(rx)
        wy, dwdry = CSRBF4(ry)
    elif wtype == 'CRBF5':
        wx, dwdrx = CSRBF5(rx)
        wy, dwdry = CSRBF5(ry)
    elif wtype == 'CRBF6':
        wx, dwdrx = CSRBF6(rx)
        wy, dwdry = CSRBF6(ry)
    else:
        print('Invalid type of weight function.')
        pass

    w = wx * wy
    dwdx = wy * dwdrx * drdx
    dwdy = wx * dwdry * drdy

    return w, dwdx, dwdy
