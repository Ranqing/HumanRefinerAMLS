from qing_operation import *
from qing_weight_2d import *
from qing_rectangle_weight import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# c represents color


def qing_plot_wireframe(ax, xdata, ydata, zdata, clr):
    print('qing_plot_wireframe color is ', clr)
    ax.plot_wireframe(xdata, ydata, zdata, color=clr, rstride=1, cstride=1)
    pass

# clrmp represents colormap


def qing_plot_surface(ax, fig, xdata, ydata, zdata, clrmp):
    zmax = np.max(zdata)
    zmin = np.min(zdata)
    print('qing_plot_surface range of z-axis %f - %f' % (zmin, zmax))

    surf = ax.plot_surface(xdata, ydata, zdata,
                           cmap=clrmp, linewidth=0, antialiased=False)

    ax.set_zlim(zmin - 0.1, zmax + 0.1)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.4, aspect=5, cmap=clrmp)
    pass


def get_pmatrices(m, nnodes, xj, yj, XI, YI):
    if m == 1:
        p = np.ones(nnodes)
        pxy = np.array([1])
        dpdx = np.array([0])
        dpdy = np.array([0])
    elif m == 3:
        p = np.array([np.ones(nnodes), XI, YI])
        pxy = np.array([[1], [xj], [yj]])
        dpdx = np.array([[0], [1], [0]])
        dpdy = np.array([[0], [0], [1]])
    elif m == 6:
        p = np.array([np.ones(nnodes), XI, YI, XI * XI, XI * YI, YI * YI])
        pxy = np.array([[1], [xj], [yj], [xj * xj], [xj * yj], [yj * yj]])
        dpdx = np.array([[0], [1], [0], [2 * xj], [yj], [0]])
        dpdy = np.array([[0], [0], [1], [xj], [2 * yj], [0]])
    else:
        print('Invalid order of basis')

    # print('p = ', p.shape, end = '\t')
    # print('pxy = ', pxy.shape, end = '\t')
    # print('dpdx = ', dpdx.shape, end = '\t')
    # print('dpdy = ', dpdy.shape, end = '\n')
    return p, pxy, dpdx, dpdy
    pass


def get_bmatrices(m, p, wI, dwdxI, dwdyI):
    if m == 1:
        B = p * wI
        DBdx = p * dwdxI
        DBdy = p * dwdyI
    elif m == 3:
        B = p * np.array([wI, wI, wI])
        DBdx = p * np.array([dwdxI, dwdxI, dwdxI])
        DBdy = p * np.array([dwdyI, dwdyI, dwdyI])
    elif m == 6:
        B = p * np.array([wI, wI, wI, wI, wI, wI])
        DBdx = p * np.array([dwdxI, dwdxI, dwdxI, dwdxI, dwdxI, dwdxI])
        DBdy = p * np.array([dwdyI, dwdyI, dwdyI, dwdyI, dwdyI, dwdyI])
    else:
        print('Invalid order of basis')

    # print('B = ', B.shape, end = '\t')
    # print('DBdx = ', DBdx.shape, end = '\t')
    # print('DBdy = ', DBdy.shape, end = '\n')
    return B, DBdx, DBdy
    pass


def get_amatrices(m, nnodes, p, wI, dwdxI, dwdyI):
    A = np.zeros((m, m))
    DAdx = np.zeros((m, m))
    DAdy = np.zeros((m, m))
    for i in range(0, nnodes):
        pcol = np.reshape(np.copy(p[:, i]), (3, 1))
        pcol_t = np.transpose(pcol)
        pp = np.dot(pcol, pcol_t)
        A = A + np.dot(wI[i], pp)
        DAdx = DAdx + np.dot(dwdxI[i], pp)
        DAdy = DAdy + np.dot(dwdyI[i], pp)

    # print('A = ', A.shape, end = '\t')
    # print('DAdx = ', DAdx.shape, end = '\t')
    # print('DAdy = ', DAdy.shape, end = '\n')
    return A, DAdx, DAdy
    pass


# SHAPE FUNCTION OF 2D MLS APPROXIMATION
#
# SYNTAX: [PHI, DPHI, DDPHI] = MLS2DShape(m, nnodes, xI,yI, npoints, xi,yi, dmI, type, para)
#
# INPUT PARAMETERS
#    m - Total number of basis functions (1: Constant basis;  2: Linear basis;  3: Quadratic basis)
#    nnodes  - Total number of nodes used to construct MLS approximation
#    npoints - Total number of points whose MLS shape function to be evaluated
#    xI,yI(nnodes) - Coordinates of nodes used to construct MLS approximation. 1-d array
#    xi,yi(npoints) - Coordinates of points whose MLS shape function to be evaluated. 1-d array
#    dm(nnodes) - Radius of support of nodes
#    wtype - Type of weight function
#    para  - Weight function parameter
#
# OUTPUT PARAMETERS
#    PHI   - MLS Shpae function
#    DPHIx  - First order derivatives of MLS Shpae function to x
#    DPHIy - First order derivatives of MLS Shpae function to y
#
def mls2dshape(m, nnodes, xI, yI, npoints, x, y, dmI, wtype, para):
    DmI = []
    wI = np.zeros(nnodes)
    dwdxI = np.zeros(nnodes)
    dwdyI = np.zeros(nnodes)

    # initialize shape function matrices
    PHI = np.zeros((npoints, nnodes))
    DPHIx = np.zeros((npoints, nnodes))
    DPHIy = np.zeros((npoints, nnodes))

    xII = np.zeros(nnodes)
    yII = np.zeros(nnodes)
    xII = np.copy(xI)
    yII = np.copy(yI)

    print('xI shape: ', xI.shape)
    print('yI shape: ', yI.shape)
    print('x shape: ', x.shape)
    print('y shape: ', y.shape)

    for j in range(0, npoints):
        DmI = np.copy(dmI)
        for i in range(0, nnodes):
            wI[i], dwdxI[i], dwdyI[i] = rectangle_weight(
                wtype, para, x[j], y[j], xI[i], yI[i], DmI[i], DmI[i])
        # print('j = %d, i = %d, x = %f, y = %f, xi = %f, yi = %f, wI = %f,
        # dwdxI = %f, dwdyI = %f' %
        # (j, i, x[j], y[j], xI[i], yI[i], wI[i], dwdxI[i], dwdyI[i]))

        p, pxy, dpdx, dpdy = get_pmatrices(m, nnodes, x[j], y[j], xII, yII)
        B, DBdx, DBdy = get_bmatrices(m, p, wI, dwdxI, dwdyI)
        A, DAdx, DAdy = get_amatrices(m, nnodes, p, wI, dwdxI, dwdyI)

        ARcond = 1 / np.linalg.cond(A, 1)
        print('After calculation, ARcond = ', ARcond, end='\t')
        while ARcond <= 9.999999e-015:
            DmI = 1.1 * DmI
            for i in range(0, nnodes):
                wI[i], dwdxI[i], dwdyI[i] = rectangle_weight(
                    wtype, para, x[j], y[j], xI[i], yI[i], DmI[i], DmI[i])

            xII = np.copy(xI)
            yII = np.copy(yI)
            p, pxy, dpdx, dpdy = get_pmatrices(m, nnodes, x[j], y[j], xII, yII)
            B, DBdx, DBdy = get_bmatrices(m, p, wI, dwdxI, dwdyI)
            A, DAdx, DAdy = get_amatrices(m, nnodes, p, wI, dwdxI, dwdyI)

            ARcond = 1 / np.linalg.cond(A, 1)
            print('\nIter: ARcond = ', ARcond)
            pass

        print('A condition statisfied.', end='\n')
        Ainv = np.linalg.inv(A)
        rxy = np.dot(Ainv, pxy)
        PHI[j, :] = np.dot(np.transpose(rxy), B)

        drdx = np.dot(Ainv, (dpdx - np.dot(DAdx, rxy)))
        DPHIx[j, :] = np.dot(np.transpose(drdx), B) + \
            np.dot(np.transpose(rxy), DBdx)

        drdy = np.dot(Ainv, (dpdy - np.dot(DAdy, rxy)))
        DPHIy[j, :] = np.dot(np.transpose(drdy), B) + \
            np.dot(np.transpose(rxy), DBdy)

    return PHI, DPHIx, DPHIy
    pass


def test_2d_mls():
    # nodes
    dx = 0.5
    vI = np.arange(-2, 2.1, dx)
    xI, yI = np.meshgrid(vI, vI)
    nnodes = len(xI) * len(yI)
    xNodes = np.zeros(nnodes)
    yNodes = np.zeros(nnodes)
    for i in range(0, nnodes):
        iy = int(i % len(yI))
        ix = int(i / len(yI))
        xNodes[i] = xI[iy][ix]
        iy = int(i % len(xI))
        ix = int(i / len(xI))
        yNodes[i] = yI[iy][ix]

    # print('nnodes = %d' % (nnodes))
    # print('xI = ', xI.shape, end='\n')
    # print(xI)
    # print('yI = ', yI.shape, end='\n')
    # print(yI)
    # print('xNodes = ', xNodes.shape, end='\n')
    # print(xNodes)
    # print('yNodes = ', yNodes.shape, end='\n')
    # print(yNodes)
    # sys.exit()

    # points
    dx = 0.1
    v = np.arange(-2, 2.1, dx)
    x, y = np.meshgrid(v, v)
    npoints = len(x) * len(y)
    xPoints = np.zeros(npoints)
    yPoints = np.zeros(npoints)
    for i in range(0, npoints):
        iy = int(i % len(y))
        ix = int(i / len(y))
        xPoints[i] = x[iy][ix]
        iy = int(i % len(x))
        ix = int(i / len(x))
        yPoints[i] = y[iy][ix]

    # print('npoints = %d' % (npoints))
    # print('x = ', x.shape, end='\n')
    # print(x)
    # print('y = ', y.shape, end='\n')
    # print(y)
    # print('xPoints = ', xPoints.shape, end='\n')
    # print(xPoints)
    # print('yPoints = ', yPoints.shape, end='\n')
    # print(yPoints)
    # sys.exit()

    # radius of support of every node
    # scale * node inter
    scale = 2.4
    dx = 0.5
    dmI = scale * dx * np.ones(nnodes)
    # dmI = np.reshape(dmI, (1,nnodes))

    print('nnodes = %d, dx = 0.5' % (nnodes))
    print('npoints = %d, dx = 0.1' % (npoints))
    # print('dmI = ', dmI.shape, end = '\n')
    # print(dmI)

    # Evaluate MLS Shape function at all evaluation points x
    PHI, DPHIx, DPHIy = mls2dshape(
        3, nnodes, xNodes, yNodes, npoints, xPoints, yPoints, dmI, 'GAUSS', 3.0)

    print('end of mls2dshape.', end='\n')
    print('PHI: ', PHI.shape,  end='\n')
    print('DPHIx: ', DPHIx.shape, end='\n')
    print('DPHIy: ', DPHIy.shape, end='\n')
    print('start curve fitting.', end='\n')

    zPoints = xPoints * np.exp(-xPoints**2 - yPoints**2)
    zNodes = xNodes * np.exp(-xNodes**2 - yNodes**2)
    zPoints_fitted = np.dot(PHI, np.transpose(zNodes))
    print('z points: ', zPoints.shape)
    print('z nodes: ', zNodes.shape)
    print('z points fitted: ', zPoints_fitted.shape)

    err = np.linalg.norm(zPoints - zPoints_fitted) / np.linalg.norm(zPoints)
    print('fitted err = %f' % (err), end='\n')

    # print('ZI ', ZI.shape)
    # print(ZI)
    # print('zNodes ', zNodes.shape)
    # print(zNodes)
    # print(''

    # plotting wireframe
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    qing_plot_wireframe(ax, xPoints, yPoints, zPoints, 'g')
    # qing_plot_wireframe(ax, xNodes, yNodes, zNodes, 'b')
    qing_plot_wireframe(ax, xPoints, yPoints, zPoints_fitted, 'b')
    plt.show()

    # plotting surface
    z = np.reshape(zPoints, (len(y), len(x)))
    zI = np.reshape(zNodes, (len(yI), len(xI)))
    z_fitted = np.reshape(zPoints_fitted, (len(y), len(x)))
    print('reshape z: ', z.shape)
    print('reshape zI: ', zI.shape)
    print('reshape z_fitted: ', z_fitted.shape)

    fig2 = plt.figure(2)
    ax2 = fig2.gca(projection='3d')
    qing_plot_surface(ax2, fig2, x, y, z, plt.cm.winter)
    qing_plot_surface(ax2, fig2, x, y, z_fitted, plt.cm.coolwarm)
    plt.show()

    # evaluate derivatives
    dzdx, dzdy = np.gradient(z)
    dzdx_fitted = np.dot(DPHIx, np.transpose(zNodes))
    dzdy_fitted = np.dot(DPHIy, np.transpose(zNodes))
    dd_zx = np.reshape(dzdx, (1, npoints))
    dd_zy = np.reshape(dzdy, (1, npoints))
    dd_zx_fitted = np.reshape(dzdx_fitted, (1, npoints))
    dd_zy_fitted = np.reshape(dzdy_fitted, (1, npoints))
    err_dx = np.linalg.norm(dd_zx - dd_zx_fitted) / np.linalg.norm(dd_zx)
    err_dy = np.linalg.norm(dd_zy - dd_zy_fitted) / np.linalg.norm(dd_zy)
    print('fitted err in dx = %f' % (err_dx))
    print('fitted err in dy = %f' % (err_dy))

    pass


def main():
    test_2d_mls()
    pass


if __name__ == '__main__':
    main()
