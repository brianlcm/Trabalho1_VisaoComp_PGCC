import numpy as np

# Funcao para calcular a homografia
def HomographyGenerator(pairs):
    A = []
    
    # Monta uma matriz de acordo com as equacoes da homografia
    for x1, y1, x2, y2 in pairs:
        A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
    A = np.array(A)

    # Resolve o problem de mínimos quadrados com o Singular Value Decomposition (SVD)
    U, S, V = np.linalg.svd(A)

    # V has shape (9, 9) for any number of input pairs. V[-1] is the eigenvector
    # of (A^T)A with the smalles eigenvalue. Reshape into 3x3 matrix.
    H = np.reshape(V[-1], (3, 3))

    # Normalization
    H = (1 / H.item(8)) * H
    
    return H

# Funcao usada para atualizar o valor de N no loop do RANSAC
def updateN(epslon, p):

    N = np.log(1-p)/np.log(1-(1-epslon)**4)

    return round(N)

# Funcao utilizada para verificar se uma amostra e uma solucao candidata de acordo com um inline threshold
def dist(pair, H):

    # points in homogeneous coordinates
    p1 = np.array([pair[0], pair[1], 1])
    p2 = np.array([pair[2], pair[3], 1])

    # Calcula o produto escalar de dois arrays
    p2_estimate = np.dot(H, np.transpose(p1))
    p2_estimate = (1 / p2_estimate[2]) * p2_estimate

    # Retorna a norma do vetor e utiliza a funcao transpose do numpy para calcular um array com eixos transpostos
    return np.linalg.norm(np.transpose(p2) - p2_estimate)

# Funcao principal que executa o RANSAC
def run_RANSAC(pointsMap):

    # Define os parametros do algoritmo do RANSAC
    N = 1000 # Numero de amostragens N
    p = 0.99 # Valor de p recomendado
    inlierThreshold = np.sqrt(6) # valor do inlier threshold recomendado
    counterSamplings = 0 # Contador de iteracoes
    bestInliers = set() # Variavel para salvar a melhor solucao encontrada
    homography = None
    epsilon = 1.0 # Epsilon inicial (pode ser alterado)
    epsilonUpdate = 0.0
    
    # Adaptative RANSAC
    while (N > counterSamplings):

        # Escolhe randomicamente 4 correspondencias para estimar a solucao
        pairs = [pointsMap[i] for i in np.random.choice(len(pointsMap), 4)]

        # Chama a funcao para calcular a homografia
        H = HomographyGenerator(pairs)

        # Seleciona os inliers de acordo com uma distancia d para a qual uma amostra concorda com a solucao candidata
        inliers = {(c[0], c[1], c[2], c[3])
                   for c in pointsMap if dist(c, H) <= inlierThreshold}

        # Minimum number of inliers to be accepted as valid set
        minInliers = (1-epsilon)*len(pointsMap)

        # Atualizacao do epslon
        epsilonUpdate = 1 - (len(inliers)/len(pointsMap))

        # Verifica se o valor de N e epslon deve ser atualizado
        if (epsilonUpdate < epsilon):
            epsilon = epsilonUpdate
            N_Update = updateN(epsilon, p)
            N = N_Update       
        
        # Se os inliers encontrados passam por essas condicoes, entao e a nova solucao otima
        if ((len(inliers) > len(bestInliers)) and (len(inliers) > minInliers)):
            bestInliers = inliers
            homography = H
        
        # Atualiza o numero de amostragens
        counterSamplings += 1
        
    return homography, bestInliers


