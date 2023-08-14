import numpy as np
from itertools import product

# Main function for the creation of the panoramic image
def fit_transform_homography(homography, img1, img2):

    # Print message on screen
    print('Generating the panoramic image...')
    
    # Take the dimensions of the image
    img_width, img_height = len(img1[0]), len(img1)

    # Create new canvas with zeros np function which returns a new array of given shape and type, filled with zeros
    warpedImg = np.zeros((img_height, img_width * 2, 3), dtype=np.uint8)

    # Scroll x and y of a new canvas twice the width of the original image
    for x, y in product(range(img_width * 2), range(img_height)):

        # Invokes the compute the map point function
        mapped_point = compute_map_point(x, y, homography)
        
        # If the origin point is in the image on the left
        if x < img_width:
            # If the mapped point is inside the image on the left
            if verify_point(mapped_point, img_height, img_width):
                # Merge the color value on the left and the interpolated value of the image on the right
                r, g, b = pixel_interpolation(mapped_point[0], mapped_point[1], img2)
                r_left, g_left, b_left = img1[y][x]

                if r == -1:
                    warpedImg[y][x] = img1[y][x]
                    continue

                threshold = 50
                if r_left - r > threshold or g_left - g > threshold or b_left - b > threshold:
                    warpedImg[y][x] = [r, g, b]
                else:
                    warpedImg[y][x] = [(r + r_left) / 2, (g + g_left) / 2, (b + b_left) / 2]
            
            # Take left pixel
            else:
                warpedImg[y][x] = img1[y][x]  
        
        # Then origin point is outside the left image
        else:  
            if verify_point(mapped_point, img_height, img_width):
                # Take right pixel
                r, g, b = pixel_interpolation(mapped_point[0], mapped_point[1], img2)
                warpedImg[y][x] = [r, g, b]

                if r == -1:
                    warpedImg[y][x] = [0, 0, 0]

    return warpedImg

# Computes the mapping of the points and returns a matrix
def compute_map_point(x, y, homography):
    
    # Calculates the product of the homography arrays and the x and y points.
    p_mapped = np.dot(homography, np.array([x, y, 1]))
    matrix_mapPoint = (1 / p_mapped[-1]) * p_mapped

    return matrix_mapPoint

# Calculates the pixel value of the source image
def pixel_interpolation(x, y, rgb_sourceImg):

    imgWidth, imgHeight = len(rgb_sourceImg[0]), len(rgb_sourceImg)
    r = bilinear_interpolation(x, y, rgb_sourceImg[:, :, 0], imgWidth, imgHeight)
    g = bilinear_interpolation(x, y, rgb_sourceImg[:, :, 1], imgWidth, imgHeight)
    b = bilinear_interpolation(x, y, rgb_sourceImg[:, :, 2], imgWidth, imgHeight)
    return r, g, b

# Computes bilinear interpolation
def bilinear_interpolation(loc_x, loc_y, pixels, imgWidth, imgHeight):
    
    # Check image boundaries
    if loc_x < 0 or loc_y < 0 or loc_x > imgWidth - 1 or loc_y > imgHeight - 1:
        return -1

    interpolated_value = 0.0

    x = int(loc_x)
    y = int(loc_y)
    a = loc_x - x
    b = loc_y - y

    # Calculates the interpolation value
    interpolated_value += (1.0 - a) * (1.0 - b) * pixels[y][x]
    interpolated_value += a * b * pixels[min(y + 1, imgHeight - 1)][min(x + 1, imgWidth - 1)]
    interpolated_value += (1.0 - a) * b * pixels[min(y + 1, imgHeight - 1)][x]
    interpolated_value += a * (1.0 - b) * pixels[y][min(x + 1, imgWidth - 1)]

    return interpolated_value

# Check if a point is within the image
def verify_point(point, image_height, image_width):

    # Returns true if the point is inside the image and returns otherwise
    if (0 <= point[0] < image_width) and (0 <= point[1] < image_height):
        return True
    else:
        return False