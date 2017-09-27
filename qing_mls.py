from qing_weight_1d import *
from qing_weight_2d import *

# SYNTAX: [PHI, DPHI, DDPHI] = MLS1DShape(m, nnodes, xi, npoints, x, dm, wtype, para)
#
# INPUT PARAMETERS
#    m - Total number of basis functions (1: Constant basis;  2: Linear basis;  3: Quadratic basis)
#    nnodes  - Total number of nodes used to construct MLS approximation
#    npoints - Total number of points whose MLS shape function to be evaluated
#    xi(nnodes) - Coordinates of nodes used to construct MLS approximation
#    x(npoints) - Coordinates of points whose MLS shape function to be evaluated
#    dm(nnodes) - Radius of support of nodes
#    wtype - Type of weight function
#    para  - Weight function parameter
#
# OUTPUT PARAMETERS
#    PHI   - MLS Shpae function
#    DPHI  - First order derivatives of MLS Shpae function
#    DDPHI - Second order derivatives of MLS Shpae function
#
def qing_1d_mls(m, nnodes, xi, npoints, x, dmI, wtype, para):

    wi = np.zeros(nnodes)           # 1 x nnodes
    dwi = np.zeros(nnodes)          # 1 x nnodes
    ddwi = np.zeros(nnodes)         # 1 x nnodes

    PHI = np.zeros((npoints, nnodes))        # npoints x nnodes
    DPHI = np.zeros((npoints, nnodes))       # npoints x nnodes
    DDPHI = np.zeros((npoints, nnodes))      # npoints x nnodes

    # loop over all evalutaion points to calculate value of shape function
    # FI(x)

    for j in range(0, npoints):
        # print('-------------------------------------------------------------')
        # print('j = ', j, '\tx = ', x[j])
        # print('-------------------------------------------------------------')

        # detemine weight function and their dericatives at every node
        for i in range(0, nnodes):
            di = x[j] - xi[i]
            # print('i = ', i, '\tx = ', x[j], '\txi = ', xi[
            #       i], '\tdi = ', di,  end='\n')

            # print(wtype, para, di, dmI[i])
            wi[i], dwi[i], ddwi[i] = qing_weight(wtype, para, di, dmI[i])
            # print('wi = ', wi[i], '\tdwi = ', dwi[i], '\tddwi = ', ddwi[i])
            # print('\n')

        # sys.exit()

        # evaluate basis p, B Matrix and their derivatives
        if m == 1:  # Shepard function
            p = np.ones(nnodes)    # 1 x nnodes
            p = np.reshape(p, (1, nnodes))
            px = [1]               # 1 x 1
            dpx = [0]              # 1 x 1
            ddpx = [0]             # 1 x 1

            # element multiplication
            B = p * wi
            DB = p * dwi
            DDB = p * ddwi
            pass
        elif m == 2:
            p = np.array([np.ones(nnodes), xi])     # 2 x nnodes
            p = np.reshape(p, (2, nnodes))
            px = np.array(([1], [x[j]]))            # 2 x 1
            dpx = np.array(([0], [1]))              # 2 x 1
            ddpx = np.array(([0], [0]))             # 2 x 1

            B = p * np.array(([wi], [wi]))          # 2 x 1
            DB = p * np.array(([dwi],  [dwi]))      # 2 x 1
            DDB = p * np.array(([ddwi], [ddwi]))    # 2 x 1
            pass
        elif m == 3:
            p = np.array(([np.ones(nnodes), xi, xi * xi]))    # 3 x nnodes
            p = np.reshape(p, (3, nnodes))
            px = np.array(([1], x[j], x[j] * x[j]))          # 3 x 1
            dpx = np.array(([0], [1], [2 * x[j]]))
            ddpx = np.array(([0], [0], [2]))

            B = p * np.array(([wi], [wi], [wi]))
            DB = p * np.array(([dwi], [dwi], [dwi]))
            DDB = p * np.array(([ddwi], [ddwi], [ddwi]))
            pass
        else:
            print('invalid order of basis')

        # print('DEBUG')
        # print('p : ', p, end='\n')
        # print('px : ', px, end='\n')
        # print('dpx : ', dpx, end='\n')
        # print('ddpx : ', ddpx, end='\n')
        # print('wi : ', wi, end='\n')
        # print('dwi : ', dwi, end='\n')
        # print('ddwi : ', ddwi, end='\n')
        # print('B = p .* wi: ', B, end='\n')
        # print('DB = p .* dwi ', DB, end='\n')
        # print('DDB = p .*  ddwi: ', DDB, end='\n')

        # evaluate matrices A and its derivatives
        A = np.zeros((m, m))
        DA = np.zeros((m, m))
        DDA = np.zeros((m, m))

        for i in range(0, nnodes):
            pp = np.dot(p[:, i], np.transpose(p[:, i]))

            A = A + wi[i] * pp
            DA = DA + dwi[i] * pp
            DDA = DDA + ddwi[i] * pp
            pass

        # if np.linalg.det(A):
        #     Ainv = np.linalg.inv(A)      # if A is not singular how to deal
        # else:
        #     Ainv = np.zeros((m, m))

        Ainv = np.linalg.inv(A)

        # print('A = ', A, end='\n')
        # print('invA = ', Ainv, end='\n')

        rx = np.dot(Ainv, px)
        PHI[j, :] = np.dot(np.transpose(rx), B)

        drx = np.dot(Ainv, (dpx - np.dot(DA, rx)))
        DPHI[j, :] = np.dot(np.transpose(drx), B) + \
            np.dot(np.transpose(rx), DB)

        ddrx = np.dot(Ainv, (ddpx - 2 * np.dot(DA, drx) - np.dot(DDA, rx)))
        DDPHI[j, :] = np.dot(np.transpose(
            ddrx), B) + 2 * np.dot(np.transpose(drx), DB) + np.dot(np.transpose(rx), DDB)
        pass

    return PHI, DPHI, DDPHI
    pass



# functions related to mls in 2d

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
def qing_2d_mls(m, nnodes, xI, yI, npoints, x, y, dmI, wtype, para):
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

