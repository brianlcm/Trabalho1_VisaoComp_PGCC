import numpy as np

# Function to calculate homography
def HomographyGenerator(pairs, T1, T2):
    
    A = []

    # Construct a matrix according to the homography equations
    for x1, y1, x2, y2 in pairs:
        A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
    A = np.array(A)

    # Solves the least squares problem with Singular Value Decomposition (SVD)
    _, _, V = np.linalg.svd(A)

    # V has shape (9, 9) for any number of input pairs. V[-1] is the eigenvector
    # of (A^T)A with the smalles eigenvalue. Reshape into 3x3 matrix.
    H = np.reshape(V[-1], (3, 3))

    H = desnormalizePoints(H, T1, T2)

    # Normalizes the points
    #H = (1 / H.item(8)) * H

    return H

# Performs normalization of points
def normalizePoints(pointsMap):

    # Get the size of the set of matches
    pointsMapSize = len(pointsMap)

    # Calculates the average of the set
    mean = np.mean(pointsMap, 0)

    # 
    s = np.linalg.norm((pointsMap-mean), axis=1).sum() / (pointsMapSize * np.sqrt(2))

    # Compute a similarity transformation T, moves original points to
    # new set of points, such that the new centroid is the origin,
    # and the average distance from origin is square root of 2
    T = np.array([[s, 0, mean[0]],
                  [0, s, mean[1]],
                  [0, 0, 1]])
    T = np.linalg.inv(T)
    pointsMap = np.dot(T, np.concatenate((pointsMap.T, np.ones((1, pointsMap.shape[0])))))
    pointsMap = pointsMap[0:2].T

    return pointsMap, T

def desnormalizePoints(HNormalized, T1, T2):
    # Denormalization: denormalize the homography back
    H = np.dot(np.dot(np.linalg.pinv(T2), HNormalized), T1)
    H = H/H[-1, -1]
    return H

# Function used to update the value of N in the RANSAC loop
def updateN(epsilon, p):

    N = np.log(1-p)/np.log(1-(1-epsilon)**4)

    return round(N)

# Function used to check if a sample is a candidate solution according to an inline threshold
def dist(pair, H):

    # Points in homogeneous coordinates
    p1 = np.array([pair[0], pair[1], 1])
    p2 = np.array([pair[2], pair[3], 1])

    # Calculates the scalar product of two arrays
    p2_estimate = np.dot(H, np.transpose(p1))
    p2_estimate = (1 / p2_estimate[2]) * p2_estimate

    # Returns the norm of the vector and uses numpy's transpose function to calculate an array with transposed axes
    return np.linalg.norm(np.transpose(p2) - p2_estimate)

# Main function that executes RANSAC
def run_RANSAC(pointsMapImg1, pointsMapImg2):

    # Defines the parameters of the RANSAC algorithm
    N = 1000 # Number of samplings N
    p = 0.99 # Recommended p-value
    inlierThreshold = np.sqrt(6) # Recommended inlier threshold value
    counterSamplings = 0 # Iteration counter
    bestInliers = set() # Variable to save the best solution found
    homography = None
    epsilon = 1.0 # Initial epsilon (can be changed)
    epsilonUpdate = 0.0

    print(f'Running RANSAC...')

    TotalPointsMap = np.concatenate([pointsMapImg1,pointsMapImg2], axis=1)

    # Normalizes the points of each image
    img1Normalized, T1 = normalizePoints(pointsMapImg1)
    img2Normalized, T2 = normalizePoints(pointsMapImg2)

    TotalPointsMapNorm = np.concatenate([img1Normalized,img2Normalized], axis=1)
    
    # Adaptative RANSAC
    while (counterSamplings < 150):

        # Randomly choose 4 matches to estimate the solution
        pairs = [TotalPointsMapNorm[i] for i in np.random.choice(len(TotalPointsMapNorm), 4)]

        # Invokes the function to calculate the homography
        H = HomographyGenerator(pairs, T1, T2)

        # Selects inliers according to a distance 'inlierThreshold'
        inliers = {(c[0], c[1], c[2], c[3])
                   for c in TotalPointsMap if dist(c, H) <= inlierThreshold}

        # Minimum number of inliers to be accepted as valid set
        minInliers = (1-epsilon)*len(TotalPointsMap)

        # Epslon update
        epsilonUpdate = 1 - (len(inliers)/len(TotalPointsMap))

        # Checks whether the value of N and epslon should be updated
        if (epsilonUpdate < epsilon):
            epsilon = epsilonUpdate
            N_Update = updateN(epsilon, p)
            N = N_Update       
        
        # If the found inliers pass these conditions, then it is the new optimal solution
        if ((len(inliers) > len(bestInliers)) and (len(inliers) > minInliers)):
            bestInliers = inliers
            homography = H
        
        # Update the number of samples
        counterSamplings += 1
        
        # Print results
        print(f'\x1b[2K\r└──> iteration {counterSamplings}/{N}', end='')
        
        print(f'\nNummber of matches: {len(TotalPointsMap)}')
        print(f'Number of inliers: {len(bestInliers)}')
        print(f'Minimum inliers: {minInliers}')
        
    
    return homography, bestInliers


