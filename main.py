# Importando bibliotecas e arquivos necessarios

import cv2
import matching_images
import ransac
import util

def main():
    # Faz a leitura do par de imagens a partir da pasta 'images'.
    imgInput1 = cv2.imread('images\\1.jpg')
    imgInput2 = cv2.imread('images\\2.jpg')

    # Chama a funcao para fazer a correspondencia entre as imagens e retornar o mapa de pontos
    pointsMap = matching_images.pointsMap(imgInput1, imgInput2)

    # Chama a funcao do para computar a homografia a partir do RANSAC
    homography, inliers = ransac.runRANSAC(pointsMap)

    #cv2.imwrite('images\\matches_RANSAC.png', util.drawMatches(imgInput1, imgInput2, pointsMap, inliers))
    util.draw_lines(inliers, imgInput1, imgInput2,
                                  line_color=[0, 255, 0], pt_color=[0, 0, 0])

    # Estima H utilizando os inliers da melhor solucao candidata
    homography = ransac.HomographyGenerator(inliers)
    
if __name__ == '__main__':
    main()