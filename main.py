# Importing required libraries and files
import cv2
import numpy as np
import matching_images
import ransac
import util
import generate_panorama

# Main project function
def main():
    
    # Defines the image pair that will be used to create the panoramic image
    pair_imgs = 'A' # A, B, C, D

    # Reads the image pair from the 'input_images' folder
    imgInput1 = cv2.imread('input_images\\pair_images_{}\\input1.jpg'.format(pair_imgs))
    imgInput2 = cv2.imread('input_images\\pair_images_{}\\input2.jpg'.format(pair_imgs))

    # Invokes the function to match the images and return the point map of each image
    pointsMapImg1, pointsMapImg2 = matching_images.run_matching_images(imgInput1, imgInput2, pair_imgs)
    
    # Invokes the do function to compute the homography from RANSAC
    homography, inliers = ransac.run_RANSAC(pointsMapImg1, pointsMapImg2)

    # Draw the inliers on the two images after using RANSAC
    util.draw_inlines(inliers, imgInput1, imgInput2, pair_imgs)
    
    # Transforms set of inliers into numpy array
    inliers = np.array(list(inliers))

    # Take the points of the inliers of each image separately
    inliersImg1 = inliers[:,:2]
    inliersImg2 = inliers[:,[-2,-1]]

    # Take the points of the inliers of each image separately
    normInliersImg1, T1 = ransac.normalizePoints(inliersImg1)
    normInliersImg2, T2 = ransac.normalizePoints(inliersImg2)

    # Concatenates normalized point arrays
    TotalInliersNorm = np.concatenate([normInliersImg1, normInliersImg2], axis=1)

    # Estimates H using the inliers of the best candidate solution
    homography = ransac.HomographyGenerator(TotalInliersNorm, T1, T2)
    
    # Invokes the function to generate the panoramic image
    panoramicImage = generate_panorama.fit_transform_homography(homography, imgInput1, imgInput2)

    # Saves final panoramic image
    cv2.imwrite('results\\pair_images_{}\\panoramic_image.png'.format(pair_imgs), panoramicImage)
    
if __name__ == '__main__':
    main()