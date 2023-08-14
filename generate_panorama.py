import numpy as np
from itertools import product

# Funcao principal para a criacao da imagem panoramica
def fit_transform_homography(homography, img1, img2):

    # Printa mensagem na tela
    print('Generating the panoramic image...')
    
    # Pega as dimensoes da imagem
    img_width, img_height = len(img1[0]), len(img1)

    # create new canvas with zeros np function which returns a new array of given shape and type, filled with zeros
    warpedImg = np.zeros((img_height, img_width * 2, 3), dtype=np.uint8)

    # loop through x and y of a new canvas twice the width of the original image
    for x, y in product(range(img_width * 2), range(img_height)):

        # Calls the compute the map point function
        mapped_point = compute_map_point(x, y, homography)
        
        # if source point is in left image
        if x < img_width:
            # mapped point is within left image
            if verify_point(mapped_point, img_height, img_width):
                # Blend color value from left and interpolated value from right image
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

            else:
                warpedImg[y][x] = img1[y][x]  # take left pixel

        else:  # source point is outside left image
            if verify_point(mapped_point, img_height, img_width):
                # take right pixel
                r, g, b = pixel_interpolation(mapped_point[0], mapped_point[1], img2)
                warpedImg[y][x] = [r, g, b]

                if r == -1:
                    warpedImg[y][x] = [0, 0, 0]

    return warpedImg

# Computa o mapeamento dos pontos e retorna uma matrix
def compute_map_point(x, y, homography):
    
    # Calcula o produto dos arrays da homografia e dos pontos x e y
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
def bilinear_interpolation(location_x, location_y, pixel_array, imgWidth, imgHeight):
    
    if location_x < 0 or location_y < 0 or location_x > imgWidth - 1 or location_y > imgHeight - 1:
        return -1

    interpolated_value = 0.0

    x = int(location_x)
    y = int(location_y)
    a = location_x - x
    b = location_y - y

    interpolated_value += (1.0 - a) * (1.0 - b) * pixel_array[y][x]
    interpolated_value += a * b * pixel_array[min(y + 1, imgHeight - 1)][min(x + 1, imgWidth - 1)]
    interpolated_value += (1.0 - a) * b * pixel_array[min(y + 1, imgHeight - 1)][x]
    interpolated_value += a * (1.0 - b) * pixel_array[y][min(x + 1, imgWidth - 1)]

    return interpolated_value

# Check if a point is within the image
def verify_point(point, image_height, image_width):

    # Returns true if the point is inside the image and returns otherwise
    if (0 <= point[0] < image_width) and (0 <= point[1] < image_height):
        return True
    else:
        return False