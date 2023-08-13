import cv2
import numpy as np
import os
import util

def pointMaps(img1, img2):

    # Chama a funcao para criar objeto SIFT a partir do OpenCV
    sift = cv2.SIFT_create()

    # Detecta e computa os pontos de interesse e descritores em cada imagem
    img1_keypoints, img1_descriptors = sift.detectAndCompute(img1, None)
    img2_keypoints, img2_descriptors = sift.detectAndCompute(img2, None)

    # Salva as imagens com os pontos de interesse
    cv2.imwrite('images\\keypoints-1.png',
                cv2.drawKeypoints(img1, img1_keypoints, img1))
    cv2.imwrite('images\\keypoints-2.png',
                cv2.drawKeypoints(img2, img2_keypoints, img2))
    
    # Chama a funcao BFMatcher do OpenCV para forcar a matching entre as duas imagens 
    matcher = cv2.BFMatcher(cv2.NORM_L2, True)
    matches = matcher.match(img1_descriptors,img2_descriptors)

    # Cria array com o mapeamento dos pontos de correspondencia encontrados entre as duas imagens
    point_map = np.array([
            [img1_keypoints[match.queryIdx].pt[0],
            img1_keypoints[match.queryIdx].pt[1],
            img2_keypoints[match.trainIdx].pt[0],
            img2_keypoints[match.trainIdx].pt[1]] for match in matches
        ])

    # Salva a imagens com todos os pontos de correspondencia
    matched_image = cv2.drawMatches(img1, img1_keypoints, img2, img2_keypoints, matches, None, flags=2)
    cv2.imwrite('images\\Matches.png', matched_image)
    
    return point_map