# Importando bibliotecas e arquivos necessarios
import cv2
import matching_images
import ransac
import util
import generate_panorama

def main():
    
    # Define o par de imagens que sera usado para criar a imagem panoramica
    pair_imgs = 'A' # A, B, C

    # Faz a leitura do par de imagens a partir da pasta 'images'.
    
    imgInput1 = cv2.imread('input_images\\pair_images_{}\\input1.jpg'.format(pair_imgs))
    imgInput2 = cv2.imread('input_images\\pair_images_{}\\input2.jpg'.format(pair_imgs))

    # Chama a funcao para fazer a correspondencia entre as imagens e retornar o mapa de pontos
    pointsMap = matching_images.run_matching_images(imgInput1, imgInput2, pair_imgs)

    # Chama a funcao do para computar a homografia a partir do RANSAC
    homography, inliers = ransac.run_RANSAC(pointsMap)

    # Desenha os inliers nas duas imagens apos o uso do RANSAC
    util.draw_inlines(inliers, imgInput1, imgInput2, pair_imgs)

    # Estima H utilizando os inliers da melhor solucao candidata
    homography = ransac.HomographyGenerator(inliers)
    
    panoramicImage = generate_panorama.fit_transform_homography(homography, imgInput1, imgInput2)

    cv2.imwrite('results\\pair_images_{}\\panoramic_image.png'.format(pair_imgs), panoramicImage)
    
if __name__ == '__main__':
    main()