from qing_operation import *
from qing_weight_2d import *
from qing_rectangle_weight import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# c represents color
def qing_plot_wireframe(ax, xdata, ydata, zdata, clr):
    print('qing_plot_wireframe c = ', clr)
    ax.plot_wireframe(xdata, ydata, zdata, color=clr, rstride=1, cstride=1)
    # ax.legend()

# clrmp represents colormap


def qing_plot_surface(ax, fig, xdata, ydata, zdata, clrmp):
    # print('qing_plot_surface clrmp = ', clrmp)
    zmax = np.max(zdata)
    zmin = np.min(zdata)
    print('zmax = %f, zmin = %f' % (zmax, zmin), end='\n')
    surf = ax.plot_surface(xdata, ydata, zdata,
                           cmap=clrmp, linewidth=0, antialiased=False)

    ax.set_zlim(zmin - 0.1, zmax + 0.1)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # ax.set_xlabel(r'X')
    # ax.set_ylabel(r'Y')
    # ax.set_zlabel(r"Z-axis = weight $w(X,Y)$")
    fig.colorbar(surf, shrink=0.4, aspect=5, cmap=clrmp)
    # ax.legend()

    pass

# SHAPE FUNCTION OF 2D MLS APPROXIMATION
#
# SYNTAX: [PHI, DPHI, DDPHI] = MLS2DShape(m, nnodes, xI,yI, npoints, xi,yi, dmI, type, para)
#
# INPUT PARAMETERS
#    m - Total number of basis functions (1: Constant basis;  2: Linear basis;  3: Quadratic basis)
#    nnodes  - Total number of nodes used to construct MLS approximation
#    npoints - Total number of points whose MLS shape function to be evaluated
#    xI,yI(nnodes) - Coordinates of nodes used to construct MLS approximation
#    xi,yi(npoints) - Coordinates of points whose MLS shape function to be evaluated
#    dm(nnodes) - Radius of support of nodes
#    wtype - Type of weight function
#    para  - Weight function parameter
#
# OUTPUT PARAMETERS
#    PHI   - MLS Shpae function
#    DPHIx  - First order derivatives of MLS Shpae function to x
#    DPHIy - First order derivatives of MLS Shpae function to y
#
# INITIALIZE WEIGHT FUNCTION MATRICES


def mls2dshape(m, nnodes, xI, yI, npoints, x, y, dmI, wtype, para):

    # print('m = ', m)
    # print('nnodes = ', nnodes)
    # print('xI = ', xI.shape, end='\n')
    # print(xI)
    # print('yI = ', yI.shape, end='\n')
    # print(yI)
    # print('x = ', x.shape, end='\n')
    # print(x)
    # print('y = ', y.shape, end='\n')
    # print(y)
    # sys.exit()

    point_h, point_w = x.shape
    node_h, node_w = xI.shape
    # print('point: ', point_h, point_w, end='\n')
    # print(npoints, ' points.')
    # print('node:  ', node_h, node_w, end='\n')
    # print(nnodes, ' nodes.')
    # sys.exit()

    # DPHIy - First order derivatives of MLS Shape function to y
    DmI = []
    wI = np.zeros(nnodes)
    dwdxI = np.zeros(nnodes)
    dwdyI = np.zeros(nnodes)
    xII = np.zeros(nnodes)
    yII = np.zeros(nnodes)

    # initialize shape function matrices
    PHI = np.zeros((npoints, nnodes))
    DPHIx = np.zeros((npoints, nnodes))
    DPHIy = np.zeros((npoints, nnodes))

    # LOOP OVER ALL EVALUATION POINTS TO CALCULATE VALUE OF SHAPE FUNCTION
    # Fi(x)
    for j in range(0, npoints):    # point index
        DmI = np.copy(dmI)
        jy = int(j % point_h)       # point row - y
        jx = int(j / point_h)       # point col - x

        for i in range(0, nnodes):  # node index
            iy = int(i % node_h)    # node row
            ix = int(i / node_h)    # node col
            wI[i], dwdxI[i], dwdyI[i] = rectangle_weight(
                wtype, para, x[jy][jx], y[jy][jx], xI[iy][ix], yI[iy][ix], DmI[i], DmI[i])
            xII[i] = xI[iy][ix]
            yII[i] = yI[iy][ix]
            # print('j = %d, i = %d, x = %f, y = %f, xi = %f, yi = %f, wI = %f,
            # dwdxI = %f, dwdyI = %f' %
            # (j, i, x[jy][jx], y[jy][jx], xI[iy][ix], yI[iy][ix], wI[i], dwdxI[i], dwdyI[i]))

        # sys.exit()
        # if j == 2:
        # sys.exit()
        # HERE

        # evaluate basis p, B matrix and their derivatives
        if m == 1:    # Shepard function
            p = np.ones(nnodes)
            p = np.reshape(p, (1, nnodes))
            pxy = np.array([1])
            dpdx = np.array([0])
            dpdy = np.array([0])

            B = p * wI
            DBdx = p * dwdxI
            DBdy = p * dwdyI
        elif m == 3:
            # print('m=3')
            p = np.array([np.ones(nnodes), xII, yII])
            # print('p = ', p.shape)
            # print(p)
            pxy = np.array([[1], [x[jy][jx]], [y[jy][jx]]])
            # print('pxy = ', pxy.shape)
            # print(pxy)
            dpdx = np.array([[0], [1], [0]])
            # print('dpdx = ',dpdx.shape)
            # print(dpdx)
            dpdy = np.array([[0], [0], [1]])
            # print('dpdy = ', dpdy.shape)
            # print(dpdy)

            B = p * np.array([wI, wI, wI])
            # print('B = ', B.shape)
            # print(B)
            DBdx = p * np.array([dwdxI, dwdxI, dwdxI])
            # print('DBdx = ', DBdx.shape)
            # print(DBdx)
            DBdy = p * np.array([dwdyI, dwdyI, dwdyI])
            # print('DBdy = ', DBdy.shape)
            # print(DBdy)
            # sys.exit()
        elif m == 6:
            p = np.array([np.ones(nnodes), xII, yII,
                          xII * xII, xII * yII, yII * yII])
            pxy = np.array([[1], [x[jy][jx]], [y[jy][jx]], [x[jy][jx] * x[jy][jx]],
                            [x[jy][jx] * y[jy][jx]], [y[jy][jx] * y[jy][jx]]])
            dpdx = np.array([[0], [1], [0], [2 * x[jy][jx]], [y[jy][jx]], [0]])
            dpdy = np.array([[0], [0], [1], [0], [x[jy][jx]], [2 * y[jy][jx]]])

            B = p * np.array([wI, wI, wI, wI, wI, wI])
            DBdx = p * np.array([dwdxI, dwdxI, dwdxI, dwdxI, dwdxI, dwdxI])
            DBdy = p * np.array([dwdyI, dwdyI, dwdyI, dwdyI, dwdyI, dwdyI])
        else:
            print('Invalid order of basis.')

        # Evaluate matrics A and Its Derivatives

        A = np.zeros((m, m))
        DAdx = np.zeros((m, m))
        DAdy = np.zeros((m, m))

        for i in range(0, nnodes):
            # print('i = %d'%(i), end = '\t')
            # print('p[:,%d] = '%(i), p[:, i], end = '\n')
            # pp = np.dot(p[:, i], np.transpose(p[:, i]))
            # pcol = np.copy(p[:,i])
            # pcol = np.reshape(pcol, (3,1))
            pcol = np.reshape(np.copy(p[:, i]), (3, 1))
            pcol_t = np.transpose(pcol)

            # print('pcol = ', pcol.shape, end = '\n')
            # print(pcol)
            # print('pcol\' = ', pcol_t.shape, end = '\n')
            # print(pcol_t)

            pp = np.dot(pcol, pcol_t)
            # print('pp = ', pp.shape, end = '\t')
            # print(pp)

            A = A + np.dot(wI[i], pp)
            DAdx = DAdx + np.dot(dwdxI[i], pp)
            DAdy = DAdy + np.dot(dwdyI[i], pp)
            # print('wI[%d] = %f' % (i, wI[i]), 'A = ', A.shape, end='\n')
            # print(A)
            # print()

        ARcond = 1 / np.linalg.cond(A, 1)  # equalivent to rcond() in matlab
        print('After all nodes, ARcond = ', ARcond, end='\t')

        # print('A = ', A.shape)
        # print(A)
        # print('DAdx = ', DAdx.shape)
        # print(DAdx)
        # print('DAdy = ', DAdy.shape)
        # print(DAdy)
        # print('ARcond = ', ARcond)
        # print('xII = ', xII.shape)
        # print('yII = ', yII.shape)
        # print('DmI = ', DmI.shape, end='\n')
        # print('element = ', DmI[0])

        while ARcond <= 9.999999e-015:
            DmI = 1.1 * DmI
            print('\nIter: ARcond = ', ARcond, end='\t')
            # print('DmI = ', DmI.shape, end='\t')
            # print('element = ', DmI[0])

            for i in range(0, nnodes):
                iy = int(i % node_h)    # node row
                ix = int(i / node_h)    # node col
                wI[i], dwdxI[i], dwdyI[i] = rectangle_weight(
                    wtype, para, x[jy][jx], y[jy][jx], xI[iy][ix], yI[iy][ix], DmI[i], DmI[i])
                xII[i] = xI[iy][ix]
                yII[i] = yI[iy][ix]

            if m == 1:
                p = np.ones(nnodes)
                pxy = np.array([1])
                dpdx = np.array([0])
                dpdy = np.array([0])

                B = p * wI
                DBdx = p * dwdxI
                DBdy = p * dwdyI
            elif m == 3:
                p = np.array([np.ones(nnodes), xII, yII])
                pxy = np.array([[1], [x[jy][jx]], [y[jy][jx]]])
                dpdx = np.array([[0], [1], [0]])
                dpdy = np.array([[0], [0], [1]])

                B = p * np.array([wI, wI, wI])
                DBdx = p * np.array([dwdxI, dwdxI, dwdxI])
                DBdy = p * np.array([dwdyI, dwdyI, dwdyI])
            elif m == 6:
                p = np.array([np.ones(nnodes), xII, yII,
                              xII * xII, xII * yII, yII * yII])
                pxy = np.array(
                    [[1], [x[j]], [y[j]], [x[j] * x[j]], [x[j] * y[j]], [y[j] * y[j]]])
                dpdx = np.array([[0], [1], [0], [2 * x[j]], [y[j]], [0]])
                dpdy = np.array([[0], [0], [1], [x[j]], [2 * y[j]], [0]])

                B = p * np.array([wI, wI, wI, wI, wI, wI])
                DBdx = p * np.array([dwdxI, dwdxI, dwdxI, dwdxI, dwdxI, dwdxI])
                DBdy = p * np.array([dwdyI, dwdyI, dwdyI, dwdyI, dwdyI, dwdyI])
            else:
                print('Invalid order of basis')

            # Evaluate matrices A and its derivatives

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
            ARcond = 1 / np.linalg.cond(A)
            pass  # end of while ARcond < epislon

        print('ARcond < 9.999999e-015', end='\n')
        if np.linalg.det(A) == 0:
            print('det(A = 0) so inv(A) is not existed', end='\n')

        Ainv = np.linalg.inv(A)
        rxy = np.dot(Ainv, pxy)
        PHI[j:] = np.dot(np.transpose(rxy), B)  # shape function

        drdx = np.dot(Ainv, (dpdx - np.dot(DAdx, rxy)))
        DPHIx[j, :] = np.dot(np.transpose(drdx), B) + \
            np.dot(np.transpose(rxy), DBdx)

        drdy = np.dot(Ainv, (dpdy - np.dot(DAdy, rxy)))
        DPHIy[j, :] = np.dot(np.transpose(drdy), B) + \
            np.dot(np.transpose(rxy), DBdy)
        pass

    return PHI, DPHIx, DPHIy


def test_2d_mls():
    # nodes
    dx = 0.5
    vI = np.arange(-2, 2.1, dx)
    xI, yI = np.meshgrid(vI, vI)
    nnodes = len(xI) * len(yI)
    xNodes = np.zeros((1, nnodes))
    for i in range(0, nnodes):
        iy = int(i % len(yI))
        ix = int(i / len(yI))
        xNodes[0][i] = xI[iy][ix]
    yNodes = np.zeros((1, nnodes))
    for i in range(0, nnodes):
        iy = int(i % len(xI))
        ix = int(i / len(xI))
        yNodes[0][i] = yI[iy][ix]

    # print('nnodes = %d' % (nnodes))
    # print('xI = ', xI.shape, end='\n')
    # print(xI)
    # print('yI = ', yI.shape, end='\n')
    # print(yI)
    # print('xNodes = ', xNodes.shape, end='\n')
    # print(xNodes)
    # print('yNodes = ', yNodes.shape, end = '\n')
    # print(yNodes)

    # points
    dx = 0.1
    v = np.arange(-2, 2.1, dx)
    x, y = np.meshgrid(v, v)
    npoints = len(x) * len(y)
    xPoints = np.zeros((1, npoints))
    for i in range(0, npoints):
        iy = int(i % len(y))
        ix = int(i / len(y))
        xPoints[0][i] = x[iy][ix]
    yPoints = np.zeros((1, npoints))
    for i in range(0, npoints):
        iy = int(i % len(x))
        ix = int(i / len(x))
        yPoints[0][i] = y[iy][ix]

    # print('npoints = %d' % (npoints))
    # print('x = ', x.shape, end='\n')
    # print(x)
    # print('y = ', y.shape, end='\n')
    # print(y)
    # print('xPoints = ', xPoints.shape, end='\n')
    # print('yPoints = ', yPoints.shape, end='\n')

    # radius of support of every node
    # scale * node inter
    scale = 3
    dx = 0.5
    dmI = scale * dx * np.ones(nnodes)
    # dmI = np.reshape(dmI, (1,nnodes))

    print('nnodes = %d, dx = 0.5' % (nnodes))
    print('npoints = %d, dx = 0.1' % (npoints))
    # sys.exit()

    # print('dmI = ', dmI.shape, end = '\n')
    # print(dmI)

    # Evaluate MLS Shape function at all evaluation points x
    PHI, DPHIx, DPHIy = mls2dshape(
        3, nnodes, xI, yI, npoints, x, y, dmI, 'GAUSS', 3.0)

    print('end of mls2dshape.\t start curve fitting')
    print('PHI: ', PHI.shape,  end='\n')
    # print(PHI)
    # sys.exit()

    ZI = xI * np.exp(-xI**2 - yI**2)
    Z = x * np.exp(-x**2 - y**2)
    node_h, node_w = ZI.shape
    point_h, point_w = Z.shape
    # print('ZI = ', ZI.shape, end='\n')
    # print(ZI)
    # print('z = ', z.shape, end='\n')
    # print(z)

    ZPoints = np.zeros((1, npoints))
    print('ZPoints = ', ZPoints.shape, end='\n')

    for i in range(0, npoints):
        iy = int(i % point_h)
        ix = int(i / point_h)
        ZPoints[0][i] = Z[iy][ix]

    ZNodes = np.zeros((1, nnodes))
    print('ZNodes = ', ZNodes.shape, end='\n')
    for i in range(0, nnodes):
        iy = int(i % node_h)
        ix = int(i / node_h)
        ZNodes[0][i] = ZI[iy][ix]

    # plotting
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    z = np.reshape(ZPoints, (len(y), len(x)))
    zI = np.reshape(ZNodes, (len(yI), len(xI)))
    # qing_plot_wireframe(ax, xPoints, yPoints, ZPoints, 'g')
    # qing_plot_wireframe(ax, xNodes, yNodes, ZNodes, 'b')
    # plt.show()

    # fig2 = plt.figure()
    # ax2 = fig.gca(projection='3d')

    # qing_plot_surface(ax, fig, xI, yI, zI, plt.cm.coolwarm)

    # function approximation
    ZHPoints = np.dot(PHI, np.transpose(ZNodes))
    print('xPoints = ', xPoints.shape)
    print('yPoints = ', yPoints.shape)
    print('ZHPoints = ', ZHPoints.shape)
    # relative error norm in approximation function
    err = np.linalg.norm(np.transpose(ZPoints) - ZHPoints) / \
        np.linalg.norm(ZPoints)
    print('fitting err = %f' % (err))
    # zh = np.reshape(ZHPoints, (len(y), len(x)))

    # qing_plot_surface(ax, fig, x, y, z, plt.cm.winter)
    # qing_plot_surface(ax, fig, x, y, zh, plt.cm.coolwarm)
    qing_plot_wireframe(ax, xPoints, yPoints, ZPoints, 'g')
    qing_plot_wireframe(ax, xPoints, yPoints, np.transpose(ZHPoints), 'b')
    plt.show()

    # converse one-dimension data to two-dimension data
    # ZPoints_sh = np.reshape(ZPoints, (len(y), len(x)))

    # dZdX, dZdY = np.gradient(ZPoints_sh)
    # ddzx = np.reshape(dZdX, (1, npoints))
    # ddzy = np.reshape(dZdY, (1, npoints))
    # ddzxh = np.dot(DPHIx, np.transpose(ZNodes))
    # err_dx = np.linalg.norm(np.transpose(ddzx) - ddzxh) / np.linalg.norm(ddzx)
    # print('fiiting err in dx = %f' % (err_dx))
    # ddzyh = np.dot(DPHIy, np.transpose(ZNodes))
    # err_dy = np.linalg.norm(np.transpose(ddzy) - ddzyh) / np.linalg.norm(ddzy)
    # print('fitting err in dy = %f' % (err_dy))

    # print('dZx = ', dZx.shape)
    # print(dZx)
    # print('dZy = ', dZy.shape)
    # print(dZy)

    # plot result


def main():
    test_2d_mls()
    pass


if __name__ == '__main__':
    main()
