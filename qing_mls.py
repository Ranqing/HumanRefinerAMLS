from qing_weight import *


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
