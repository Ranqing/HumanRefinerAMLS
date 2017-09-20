import sys
import os
import numpy as np

# function [w, dwdx, dwdy] = Weight2D(type, para, x,y,xI,yI,dmI)
# EVALUATE WEIGHT FUNCTION
#
# SYNTAX: [w, dwdr, dwdrr] = GaussWeight(type, para, di, dmi)
#
# INPUT PARAMETERS
#    type - Type of weight function
#    para - Weight function parameter
#    x,y   - gauss point coordinates matrix
#    xI,yI  -  nodal point coordinate
#    dmI - Support size
# OUTPUT PARAMETERS
#    w    - Value of weight function at r
#    dwdx - Value of first order derivative of weight function with respect to x at r
# dwdy - Value of first order derivative of weight function with respect
# to y at r


def weight2d(wtype, para, x, y, xI, yI, dmI):
    # print('shape of x: ', x.shape)
    # print('shape of y: ', y.shape)
    r = np.sqrt((x - xI)**2 + (y - yI)**2) / dmI
    nnodes_x = len(x)
    nnodes_y = len(y)
    w = np.zeros(nnodes_x, nnodes_y)
    dwdx = np.zeros(nnodes_x, nnodes_y)
    dwdy = np.zeros(nnodes_x, nnodes_y)
    drdx = np.zeros(nnodes_x, nnodes_y)
    drdy = np.zeros(nnodes_x, nnodes_y)
    dwdr = np.zeros(nnodes_x, nnodes_y)
    dwdr = np.zeros(nnodes_x, nnodes_y)

    for j in range(0, nnodes_y):
        for i in range(0, nnodes_x):
            if r[i][j] == 0:
                drdx[i][j] = 0
                drdy[i][j] = 0
            else:
                drdx[i][j] = x[i][j] / (dmI**2 * r[i][j])
                drdy[i][j] = y[i][j] / (dmI**2 * r[i][j])

    # EVALUATE WEIGHT FUNCTION AND ITS FIRST AND SECOND ORDER OF DERIVATIVZES
    # WITH RESPECT r AT r

    if wtype == 'GAUSS':
        w[i][j], dwdr[i][j] = Gauss(para, r[i][j])
    elif wtype == 'CUBIC':
        w[i][j], dwdr[i][j] = Cubic(r[i][j])
    elif wtype == 'SPLI3':
        w[i][j], dwdr[i][j] = Spline3(r[i][j])
    elif wtype == 'SPLI5':
        w[i][j], dwdr[i][j] = Spline5(r[i][j])
    elif wtype == 'SPLIB':
        w[i][j], dwdr[i][j] = BSpline(dmI / 2, r[i][j])
    elif wtype == 'power':
        w[i][j], dwdr[i][j] = power_function(para, r[i][j])
    elif wtype == 'CRBF1':
        w[i][j], dwdr[i][j] = CSRBF1(r[i][j])
    elif wtype == 'CRBF2':
        w[i][j], dwdr[i][j] = CSRBF2(r[i][j])
    elif wtype == 'CRBF3':
        w[i][j], dwdr[i][j] = CSRBF3(r[i][j])
    elif wtype == 'CRBF4':
        w[i][j], dwdr[i][j] = CSRBF4(r[i][j])
    elif wtype == 'CRBF5':
        w[i][j], dwdr[i][j] = CSRBF5(r[i][j])
    elif wtype == 'CRBF6':
        w[i][j], dwdr[i][j] = CSRBF6(r[i][j])
    else:
        print('Invalid type of weight function')

    dwdx = dwdr * drdx
    dwdy = dwdr * drdy

    return w, dwdx, dwdy


# shape function
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

    if r <= 1.0 and r > 0.5:
        w = 4 / 3 - 4 * r + 4 * r**2 - (4 * r**3) / 3
        dwdr = -4 + 8 * r - 4 * r**2
    elif r <= 0.5:
        w = 2 / 3 - 4 * r**2 + 4 * r**3
        dwdr = -8 * r + 12 * r**2

    return w, dwdr


def Spline5(r):
    w = 0.0
    dwdr = 0.0

    if r <= 1.0:
        w = 1 - 10 * r**3 + 15 * r**4 - 6 * r**5
        dwdr = -30 * r**2 + 60 * r**3 - 30 * r**4

    return w, dwdr


def BSpline(h, r):
    w = 0.0
    dwdr = 0.0

    if i <= 1.0 and i > 0.5:
        w = 2 / (pi * h**3) * (1 - r)**3
        dwdr = -6 / (pi * h**3) * (1 - r)**2
    elif i <= 0.5:
        w = 1 / (pi * h**3) * (1 - 6 * r**2 + 6 * r**3)
        dwdr = 1 / (pi * h**3) * (-12 * r + 18 * r**2)
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


def CSRBF1(r):
    w = 0.0
    dwdr = 0.0

    if r <= 1.0:
        w = (1 - r)**4 * (4 + 16 * r + 12 * r**2 + 3 * r**3)
        dwdr = -4 * (1 - r)**3 * (4 + 16 * r + 12 * r**2 + 3 *
                                  r**3) + (1 - r)**4 * (16 + 24 * r + 9 * r**2)

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


def CSRBF3(r):
    w = 0.0
    dwdr = 0.0

    if r <= 1.0:
        w = 1 / 3 + r**2 - 4 / 3 * r**3 + 2 * r**2 * np.log(r)
        dwdr = 4 * r - 4 * r**2 + 4 * r * np.log(r)

    return w, dwdr


def CSRBF4(r):
    w = 0.0
    dwdr = 0.0

    if r <= 1.0:
        w = 1 / 15 + 19 / 6 * r**2 - 16 / 3 * r**3 + 3 * r**4 - \
            16 / 15 * r**5 + 1 / 6 * r ^ 6 + 2 * r**2 * log(r)
        dwdr = 25 / 3 * r - 16 * r**2 + 12 * r**3 - \
            16 / 3 * r**4 + r**5 + 4 * r * np.log(r)

    return w, dwdr


def CSRBF5(r):
    w = 0.0
    dwdr = 0.0

    if r <= 1.0:
        w = (1 - r)**6 * (35 * r**2 + 18 * r + 3)
        dwdr = -6 * (1 - r)**5 * (35 * r**2 + 18 * r + 3) + \
            (1 - r)**6 * (70 * r + 18)

    return w, dwdr


def CSRBF6(r):
    w = 0.0
    dwdr = 0.0

    if r <= 1.0:
        w = (1 - r)**8 * (32 * r**3 + 25 * r**2 + 8 * r + 1)
        dwdr = -8 * (1 - r)**7 * (32 * r**3 + 25 * r**2 + 8 *
                                  r + 1) + (1 - r)**8 * (96 * r**2 + 50 * r + 8)

    return w, dwdr


def main():
    pass


if __name__ == '__main__':
    main()
