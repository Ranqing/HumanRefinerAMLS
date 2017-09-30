from qing_operation import *
from qing_mls import qing_2d_mls

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


def test_2d_mls():
    # nodes
    dx = 0.25
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
    # scale = 2.0
    scale = 1.0
    dx = 0.5
    dmI = scale * dx * np.ones(nnodes)
    # dmI = np.reshape(dmI, (1,nnodes))

    print('nnodes = %d, dx = 0.5' % (nnodes))
    print('npoints = %d, dx = 0.1' % (npoints))
    # print('dmI = ', dmI.shape, end = '\n')
    # print(dmI)

    # Evaluate MLS Shape function at all evaluation points x
    PHI, DPHIx, DPHIy = qing_2d_mls(
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
