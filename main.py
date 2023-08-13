# Importando bibliotecas e arquivos necessarios

import cv2
import matching_images
import ransac

def main():
    # Faz a leitura do par de imagens a partir da pasta 'images'.
    imgInput1 = cv2.imread('images\\1.jpg')
    imgInput2 = cv2.imread('images\\2.jpg')

    # Chama a funcao para fazer a correspondencia entre as imagens e retornar o mapa de pontos
    pointsMap = matching_images.pointsMap(imgInput1, imgInput2)

    # Chama a funcao do para computar a homografia a partir do RANSAC
    homography, inliers = ransac.runRANSAC(pointsMap)
    homography = ransac.HomographyGenerator(inliers)
    
if __name__ == '__main__':
    main()