import cv2
import matching_images
import ransac

def main():
    # Faz a leitura do par de imagens a partir da pasta 'images'.
    img1 = cv2.imread('images\\1.jpg')
    img2 = cv2.imread('images\\2.jpg')

    # Chama a funcao para fazer a correspondencia entre as imagens e retornar o mapa de pontos
    mapaPontos = matching_images.pointMaps(img1, img2)

    # Chama a funcao do para computar a homografia a partir do RANSAC
    homografia, inliers = ransac.RANSAC(mapaPontos)
    homografia = ransac.calculaHomografia(inliers)
    
if __name__ == '__main__':
    main()