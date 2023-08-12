import cv2
import matching_images

def main():

    img1 = cv2.imread('imagens/imagem1.jpg')
    img2 = cv2.imread('imagens/imagem2.jpg')
    mapaPontos = matching_images.returna_mapaPontos()
