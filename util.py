import cv2
import numpy as np

# Function to draw lines to indicate correspondence
def draw_inlines(inliers, img_1, img_2, pair_imgs):

        # Define some parameters to draw the inliers
        lineColor = [0, 255, 0] # The color of the line has been set to green
        ptColor = [0, 0, 0] # The color of the point has been set to black
        lineThickness = 2 # Line thickness
        radius = 5 # radius of circles
        circThickness = 2 # Thickness of circles

        # Shape function returns a tuple of the number of rows, columns, and channels
        _, w, _ = img_1.shape 

        # Use numpy's hstack function to stack the arrays of the two images
        imgStack = np.hstack((img_1, img_2))

        # Loop to go through the inliers and draw the circles and lines in the two images
        for x1, y1, x2, y2 in inliers:
            x1_d = int(round(x1))
            y1_d = int(round(y1))

            x2_d = int(round(x2) + w)
            y2_d = int(round(y2))

            # Draw circles on images
            cv2.circle(imgStack, (x1_d, y1_d), radius=radius, color=ptColor,
                       thickness=circThickness, lineType=cv2.LINE_AA)

            cv2.circle(imgStack, (x2_d, y2_d), radius=radius, color=ptColor,
                       thickness=circThickness, lineType=cv2.LINE_AA)

            # Draw lines on images
            cv2.line(imgStack, (x1_d, y1_d), (x2_d, y2_d), color=lineColor,
                     thickness=lineThickness)

        # Save the image with the markings
        cv2.imwrite('results\\pair_images_{}\\matches_RANSAC.png'.format(pair_imgs), imgStack)