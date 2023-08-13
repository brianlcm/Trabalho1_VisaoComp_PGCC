import numpy as np

def calculaHomografia(pares):
    A = []
    for x1, y1, x2, y2 in pares:
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

def atualizaN(epslon):
    p = 0.99
    novoN = np.log(1-p)/np.log(1-(1-epslon)**4)
    return novoN

def dist(pair, H):
    # points in homogeneous coordinates
    p1 = np.array([pair[0], pair[1], 1])
    p2 = np.array([pair[2], pair[3], 1])

    p2_estimate = np.dot(H, np.transpose(p1))
    p2_estimate = (1 / p2_estimate[2]) * p2_estimate

    return np.linalg.norm(np.transpose(p2) - p2_estimate)

def runRANSAC(mapaPontos):
    N = 10000
    contador_amostras = 0
    bestInliers = set()
    homography = None
    epslon = 1
    novo_epslon = None
    
    while (N > contador_amostras):

        pares = [mapaPontos[i] for i in np.random.choice(len(mapaPontos), 4)]

        H = calculaHomografia(pares)
        inliers = {(c[0], c[1], c[2], c[3])
                   for c in mapaPontos if dist(c, H) < 500}

        novo_epslon = 1 - (len(inliers)/len(mapaPontos))
        
        if (novo_epslon < epslon):
            epslon = novo_epslon
            novoN = atualizaN(epslon)

        if len(inliers) > len(bestInliers):
            bestInliers = inliers
            homography = H

        if novoN < N:
            break
        N = novoN
        contador_amostras+=1
        
    return homography, bestInliers


