import numpy as np

def quaternion_from_matrix(matrix, isprecise=False):
    '''Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> numpy.allclose(quaternion_from_matrix(R, isprecise=False),
    ...                quaternion_from_matrix(R, isprecise=True))
    True

    '''

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]

        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0

        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0.0:
        np.negative(q, q)

    return q

def compute_R_diff(R_gt, R):
    eps = 1e-15
    if R_gt.shape[0] == 4:
        q_gt = R_gt
    else:
        q_gt = quaternion_from_matrix(R_gt)
    if R.shape[0] == 4:
        q = R
    else: 
        q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)
    return np.rad2deg(np.abs(err_q))

def projM2R(m):
    u, _, v = np.linalg.svd(m)
    R = u@v
    R = R.T
    # R = np.sqrt(3)* R / np.sqrt(np.sum(np.square(R))+1e-6) #SVD guaranteed
    if np.linalg.det(R)<0:
        R = R[[1,0,2]]
    return R

def eigenrs(locws, Rs, N, normalized = False):
    # normalized: if we use normalized Laplacian matrix of not
    L = np.eye(N*3)
    # construct D
    Ddiag = np.zeros([N])
    for pid in range(locws.shape[0]):
        i, j, w = int(locws[pid,0]), int(locws[pid,1]), locws[pid,2]
        Ddiag[i] += w
    if not normalized:
        for i in range(N):
            xs, xe, ys, ye = i*3, i*3+3, i*3, i*3+3
            L[xs:xe, ys:ye] = np.eye(3)*Ddiag[i]
    # construct -A
    for pid in range(locws.shape[0]):
        i, j, w = int(locws[pid,0]), int(locws[pid,1]), locws[pid,2]
        xs, xe, ys, ye = i*3, i*3+3, j*3, j*3+3
        if normalized:
            L[xs:xe, ys:ye] = -w*Rs[pid]/np.sqrt(Ddiag[i]*Ddiag[j]+1e-8)
        else:
            L[xs:xe, ys:ye] = -w*Rs[pid]
    # Rs
    w, v = np.linalg.eig(L)
    w, v = w.real.astype(np.float32), v.real.astype(np.float32)
    # small to large
    args = np.argsort(w)
    w = w[args[0:3]]
    v = v[:,args[0:3]]
    Rpre = []
    for i in range(N):
        m = v[i*3:(i+1)*3,:]
        r = projM2R(m)
        Rpre.append(r[None,:,:])
    return np.concatenate(Rpre, axis=0)

def leastsquare(locws, ts, pcrs, N):
    '''
    base :  RiPi+ti = RjPj+tj
            Pi = RijPj+tij
            --> Ri(RijPj+tij)+ti = RjPj+tj
            --> RiRij = Rj   Ritij+ti = tj
            --> -ti + tj = Ritij (3 formular)
    least square:
        B: (p*3)*(N*3)
        P: (p*3)*(p*3)
        L: Ritij--> (p*3)        
    '''
    tpre = np.zeros([N,3])
    p = locws.shape[0]
    B = np.zeros([p*3, N*3])
    P = np.eye(p*3)
    L = np.zeros([p*3])
    # get B and P
    for pid in range(locws.shape[0]):
        # the pid-th pair
        i, j, w = int(locws[pid,0]), int(locws[pid,1]), locws[pid,2]
        P[pid*3:(pid+1)*3] *= w
        B[pid*3:(pid+1)*3, i*3:(i+1)*3] = -np.eye(3)
        B[pid*3:(pid+1)*3, j*3:(j+1)*3] = np.eye(3)
    # get L
    for pid in range(locws.shape[0]):
        # the pid-th pair
        i, j, w = int(locws[pid,0]), int(locws[pid,1]), locws[pid,2]
        L[pid*3:(pid+1)*3] = pcrs[i]@ts[pid] - (tpre[j]-tpre[i])
    # delta
    deltat = np.linalg.pinv(B.T@P@B)@(B.T@P@L)
    tpre += deltat.reshape(N,3)
    
    # final T
    Tpre = []
    for i in range(N):
        r = pcrs[i]
        t = tpre[i]
        T = np.eye(4)
        T[0:3,0:3] = r
        T[0:3,-1] = t
        Tpre.append(T[None,:,:])
    Tpre = np.concatenate(Tpre, axis=0)
    return Tpre 

def LaplacianT(locws, Rs, ts, N, calt = False):
    '''
    Generate the pc transformation to global coor from the pair transformations(Tpair graph)
    locws: p*3 [(i,j,w)]
    Rs: p*3*3
    ts: p*3
    N: the number of point cloud
    '''
    # for Rpres
    Rpre = eigenrs(locws, Rs, N)
    # for tpres --> Ts
    if calt:
        Tpre = leastsquare(locws, ts, Rpre, N)
    else:
        Tpre = np.eye(4)[None,:,:].repeat(N,axis=0)
        for i in range(N):
            Tpre[i,0:3,0:3]=Rpre[i]
    return Tpre
 
def error_reweight(iterid, locws, Ts, Tpre, all_iters = 50):
    for pid in range(locws.shape[0]):
        i, j = int(locws[pid,0]), int(locws[pid,1])
        Ti = Tpre[i]
        Tj = Tpre[j]
        Tijfit = np.linalg.inv(Ti)@Tj
        Tijpre = Ts[i,j]      
        rdiff = compute_R_diff(Tijfit[0:3,0:3], Tijpre[0:3,0:3])
        # re-weighting
        rdiff = 2*rdiff*(iterid+1)/np.sum(np.arange(all_iters)+1)
        rdiff = np.exp(-rdiff)
        locws[pid,-1] *= rdiff
    return locws

def keep_symmetry(locws,N):
    mat = np.zeros([N,N])
    for pid in range(locws.shape[0]):
        i, j, w = int(locws[pid,0]), int(locws[pid,1]), locws[pid,2]
        if mat[j,i]>0:
            mat[i,j] = min(w,mat[i,j])
            mat[j,i] = min(w,mat[j,i])
        else:
            mat[i,j] = w
            mat[j,i] = w
    # check
    for pid in range(locws.shape[0]):
        locws[pid,2] = mat[int(locws[pid,0]),int(locws[pid,1])]
    return locws
    
def pair2globalT_cycle(W_edges, Ts, iters):
    # Input\W_edges: N*N
    # Input\Ts:      N*N*4*4
    # Input\iters:   100
    # Intermediate:
    #   locws --> p*[i,j,w] (None zero w)
    #   Rs --> p*[3*3]
    # Output\Tpre:   N*4*4
    # Output\locs:  pair indexes and pair transformations
    N = W_edges.shape[0]
    locws = []
    Rs = []
    ts = []
    for i in range(Ts.shape[0]):
        for j in range(Ts.shape[1]):
            if W_edges[i,j]>0:
                locws.append(np.array([[i,j,W_edges[i,j]]]))
                Rs.append(Ts[i,j,0:3,0:3][None,:,:])
                ts.append(Ts[i,j,0:3,-1][None,:])
    Rs = np.concatenate(Rs, axis=0)
    ts = np.concatenate(ts, axis=0)
    locws = np.concatenate(locws, axis=0)
    # cycle re-weighting
    Tpre = LaplacianT(locws, Rs, ts, N)
    for i in range(iters):
        locws = error_reweight(i, locws, Ts, Tpre, all_iters = iters)
        
        # # Keep
        # temp = locws[(locws[:, 0]<5) & (locws[:, 1]<5)]
        # temp[:, -1:] = 1
        # locws[(locws[:, 0]<5) & (locws[:, 1]<5)] = temp

        # temp = locws[(locws[:, 0]>=5) & (locws[:, 1]>=5)]
        # temp[:, -1:] = 1
        # locws[(locws[:, 0]>=5) & (locws[:, 1]>=5)] = temp

        locws = keep_symmetry(locws,N) 
        Tpre = LaplacianT(locws, Rs, ts, N, calt=False)
    Tpre = LaplacianT(locws, Rs, ts, N, calt=True)
    return Tpre, locws