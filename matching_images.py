import cv2
import numpy as np
import os

# Function that uses the ORB descriptor to calculate the points of interest and compute the descriptors
def ORB_descriptor(img1_gray, img2_gray):

    # Invokes the function to create ORB object from OpenCV
    orb = cv2.ORB_create()

    # Detects and computes interest points and descriptors in each image
    img1_keypoints, img1_descriptors = orb.detectAndCompute(img1_gray, None)
    img2_keypoints, img2_descriptors = orb.detectAndCompute(img2_gray, None)

    return img1_keypoints, img1_descriptors, img2_keypoints, img2_descriptors

# Function that uses the SIFT descriptor to calculate the points of interest and compute the descriptors
def SIFT_descriptor(img1_gray, img2_gray):

    # Invokes the function to create ORB object from OpenCV
    sift = cv2.SIFT_create()

    # Detects and computes interest points and descriptors in each image
    img1_keypoints, img1_descriptors = sift.detectAndCompute(img1_gray, None)
    img2_keypoints, img2_descriptors = sift.detectAndCompute(img2_gray, None)

    return img1_keypoints, img1_descriptors, img2_keypoints, img2_descriptors

# Function that performs matching between images using OpenCV's BFMatcher algorithm
def run_matching_images(img1, img2, pair_imgs):

    # Convert images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Invokes the function to detect and compute the interest points and descriptors in each image (ORB or SIFT)
    img1_keypoints, img1_descriptors, img2_keypoints, img2_descriptors = ORB_descriptor(img1_gray, img2_gray)

    # Save images with points of interest
    cv2.imwrite('results\\pair_images_{}\\keypoints_img1.png'.format(pair_imgs),
                cv2.drawKeypoints(img1_gray, img1_keypoints, img1_gray))
    cv2.imwrite('results\\pair_images_{}\\keypoints_img2.png'.format(pair_imgs),
                cv2.drawKeypoints(img2_gray, img2_keypoints, img2_gray))
    
    # Invokes the OpenCV's BFMatcher function to force matching between the two images
    matcher = cv2.BFMatcher(cv2.NORM_L2, True)
    matches = matcher.match(img1_descriptors,img2_descriptors)

    # Creates array with the mapping of the correspondence points found between the two images
    pointsMapImg1  = np.array([
            [img1_keypoints[match.queryIdx].pt[0],
            img1_keypoints[match.queryIdx].pt[1]] for match in matches
        ])
    
    pointsMapImg2  = np.array([
            [img2_keypoints[match.trainIdx].pt[0],
            img2_keypoints[match.trainIdx].pt[1]] for match in matches
        ])
    
    # Saves images with all matching points
    matched_image = cv2.drawMatches(img1, img1_keypoints, img2, img2_keypoints, matches, None, flags=2)
    cv2.imwrite('results\\pair_images_{}\\matches_BFMatcher.png'.format(pair_imgs), matched_image)

    return pointsMapImg1, pointsMapImg2