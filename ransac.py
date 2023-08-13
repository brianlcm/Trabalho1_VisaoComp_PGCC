import numpy as np

def HomographyGenerator(pairs):
    A = []
    for x1, y1, x2, y2 in pairs:
        A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
    A = np.array(A)

    # Singular Value Decomposition (SVD)
    U, S, V = np.linalg.svd(A)

    # V has shape (9, 9) for any number of input pairs. V[-1] is the eigenvector
    # of (A^T)A with the smalles eigenvalue. Reshape into 3x3 matrix.
    H = np.reshape(V[-1], (3, 3))

    # Normalization
    H = (1 / H.item(8)) * H
    return H

def updateN(epslon, p):

    N = np.log(1-p)/np.log(1-(1-epslon)**4)

    return N

def dist(pair, H):
    # points in homogeneous coordinates
    p1 = np.array([pair[0], pair[1], 1])
    p2 = np.array([pair[2], pair[3], 1])

    p2_estimate = np.dot(H, np.transpose(p1))
    p2_estimate = (1 / p2_estimate[2]) * p2_estimate

    return np.linalg.norm(np.transpose(p2) - p2_estimate)

def runRANSAC(pointsMap):
    # Define os parametros do algoritmo do RANSAC
    N = 10000 # Numero de amostragens N
    p = 0.99
    counterSamples = 0
    bestInliers = set()
    homography = None
    epslon = 1.0
    epslonUpdate = None
    
    # Adaptative RANSAC
    while (N > counterSamples):

        # Escolhe randomicamente 4 correspondencias para estimar a solucao
        pairs = [pointsMap[i] for i in np.random.choice(len(pointsMap), 4)]

        H = HomographyGenerator(pairs)
        inliers = {(c[0], c[1], c[2], c[3])
                   for c in pointsMap if dist(c, H) < 500}

        epslonUpdate = 1 - (len(inliers)/len(pointsMap))
        
        if (epslonUpdate < epslon):
            epslon = epslonUpdate
            novoN = updateN(epslon, p)

        if len(inliers) > len(bestInliers):
            bestInliers = inliers
            homography = H

        if novoN < N:
            break
        N = novoN
        counterSamples+=1
        
    return homography, bestInliers


