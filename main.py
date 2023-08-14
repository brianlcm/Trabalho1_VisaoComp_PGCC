# Importing required libraries and files
import cv2
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

    # Invokes the function to match the images and return the point map.
    pointsMap = matching_images.run_matching_images(imgInput1, imgInput2, pair_imgs)
    
    # Invokes the do function to compute the homography from RANSAC
    homography, inliers = ransac.run_RANSAC(pointsMap)

    # Draw the inliers on the two images after using RANSAC
    util.draw_inlines(inliers, imgInput1, imgInput2, pair_imgs)

    # Estimates H using the inliers of the best candidate solution
    homography = ransac.HomographyGenerator(inliers)
    
    # Invokes the function to generate the panoramic image
    panoramicImage = generate_panorama.fit_transform_homography(homography, imgInput1, imgInput2)

    # Saves final panoramic image
    cv2.imwrite('results\\pair_images_{}\\panoramic_image.png'.format(pair_imgs), panoramicImage)
    
if __name__ == '__main__':
    main()