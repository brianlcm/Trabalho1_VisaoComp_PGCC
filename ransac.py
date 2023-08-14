import numpy as np

# Function to calculate homography
def HomographyGenerator(pairs):
    
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

    # Normalizes the points
    H = (1 / H.item(8)) * H

    return H

# Function used to update the value of N in the RANSAC loop
def updateN(epslon, p):

    N = np.log(1-p)/np.log(1-(1-epslon)**4)

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
def run_RANSAC(pointsMap):

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
    
    # Adaptative RANSAC
    while (N > counterSamplings):

        # Randomly choose 4 matches to estimate the solution
        pairs = [pointsMap[i] for i in np.random.choice(len(pointsMap), 4)]

        # Invokes the function to calculate the homography
        H = HomographyGenerator(pairs)

        # Selects inliers according to a distance 'inlierThreshold'
        inliers = {(c[0], c[1], c[2], c[3])
                   for c in pointsMap if dist(c, H) <= inlierThreshold}

        # Minimum number of inliers to be accepted as valid set
        minInliers = (1-epsilon)*len(pointsMap)

        # Epslon update
        epsilonUpdate = 1 - (len(inliers)/len(pointsMap))

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
        
        print(f'\nNummber of matches: {len(pointsMap)}')
        print(f'Number of inliers: {len(bestInliers)}')
        print(f'Minimum inliers: {minInliers}')
        
    return homography, bestInliers


