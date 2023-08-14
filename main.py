# Importando bibliotecas e arquivos necessarios
import cv2
import matching_images
import ransac
import util
import generate_panorama

def main():

    # Faz a leitura do par de imagens a partir da pasta 'images'.
    imgInput1 = cv2.imread('input_images\\input1A.jpg')
    imgInput2 = cv2.imread('input_images\\input1B.jpg')

    # Chama a funcao para fazer a correspondencia entre as imagens e retornar o mapa de pontos
    pointsMap = matching_images.run_matching_images(imgInput1, imgInput2)

    # Chama a funcao do para computar a homografia a partir do RANSAC
    homography, inliers = ransac.run_RANSAC(pointsMap)

    # Desenha os inliers nas duas imagens apos o uso do RANSAC
    util.draw_inlines(inliers, imgInput1, imgInput2)

    # Estima H utilizando os inliers da melhor solucao candidata
    homography = ransac.HomographyGenerator(inliers)
    
    panoramicImage = generate_panorama.fit_transform_homography(homography, imgInput1, imgInput2)

    cv2.imwrite('results\\final-2.png', panoramicImage)
    
if __name__ == '__main__':
    main()