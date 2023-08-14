import numpy as np
from itertools import product

# Funcao principal para a criacao da imagem panoramica
def fit_transform_homography(homography, img1, img2):

    # Pega as dimensoes da imagem
    image_width, image_height = len(img1[0]), len(img1)

    # create new canvas with zeros np function which returns a new array of given shape and type, filled with zeros
    warped_image = np.zeros((image_height, image_width * 2, 3), dtype=np.uint8)

    # loop through x and y of a new canvas twice the width of the original image
    for x, y in product(range(image_width * 2), range(image_height)):

        # Calls the Compute the map point function
        mapped_point = compute_map_point(x, y, homography)
        
        # if source point is in left image
        if x < image_width:
            # mapped point is within left image
            if verify_point(mapped_point, image_height, image_width):
                # Blend color value from left and interpolated value from right image
                r, g, b = interpolation_pixel(mapped_point[0], mapped_point[1], img2)
                r_left, g_left, b_left = img1[y][x]

                if r == -1:
                    warped_image[y][x] = img1[y][x]
                    continue

                threshold = 50
                if r_left - r > threshold or g_left - g > threshold or b_left - b > threshold:
                    warped_image[y][x] = [r, g, b]
                else:
                    warped_image[y][x] = [(r + r_left) / 2, (g + g_left) / 2, (b + b_left) / 2]

            else:
                warped_image[y][x] = img1[y][x]  # take left pixel

        else:  # source point is outside left image
            if verify_point(mapped_point, image_height, image_width):
                # take right pixel
                r, g, b = interpolation_pixel(mapped_point[0], mapped_point[1], img2)
                warped_image[y][x] = [r, g, b]

                if r == -1:
                    warped_image[y][x] = [0, 0, 0]

    return warped_image

def compute_map_point(x: int, y: int, homography: np.ndarray) -> np.ndarray:
    """
    Compute the map point
    Parameters
    ----------
    x: int
        x coordinate
    y: int
        y coordinate
    homography: np.ndarray
        homography matrix
    Returns
    -------
    np.ndarray
        the matrix that contains the map point
    """
    p_prime = np.dot(homography, np.array([x, y, 1]))
    return (1 / p_prime[-1]) * p_prime

def interpolation_pixel(x: float, y: float, rgb_source_image: np.ndarray):
    # get the pixel value from the source image
    image_width, image_height = len(rgb_source_image[0]), len(rgb_source_image)
    r = compute_bilinear_interpolation(x, y, rgb_source_image[:, :, 0], image_width, image_height)
    g = compute_bilinear_interpolation(x, y, rgb_source_image[:, :, 1], image_width, image_height)
    b = compute_bilinear_interpolation(x, y, rgb_source_image[:, :, 2], image_width, image_height)
    return r, g, b


def compute_bilinear_interpolation(location_x, location_y, pixel_array, image_width, image_height):
    if location_x < 0 or location_y < 0 or location_x > image_width - 1 or location_y > image_height - 1:
        return -1

    interpolated_value = 0.0

    x = int(location_x)
    y = int(location_y)
    a = location_x - x
    b = location_y - y

    interpolated_value += (1.0 - a) * (1.0 - b) * pixel_array[y][x]
    interpolated_value += a * b * pixel_array[min(y + 1, image_height - 1)][min(x + 1, image_width - 1)]
    interpolated_value += (1.0 - a) * b * pixel_array[min(y + 1, image_height - 1)][x]
    interpolated_value += a * (1.0 - b) * pixel_array[y][min(x + 1, image_width - 1)]

    return interpolated_value

# Check if a point is within the image
def verify_point(point, image_height, image_width):

    # Returns true if the point is inside the image and returns otherwise
    if (0 <= point[0] < image_width) and (0 <= point[1] < image_height):
        return True
    else:
        return False