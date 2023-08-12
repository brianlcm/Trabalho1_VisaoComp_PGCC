import cv2
import numpy as np

def returna_mapaPontos(img1, img2):

    orb = cv2.ORB_create()

    img1_keypoints, img1_descriptors = orb.detectAndCompute(img1, None)
    img2_keypoints, img2_descriptors = orb.detectAndCompute(img2, None)

    cv2.imwrite('imagens/keypoints-1.png',
                    cv2.drawKeypoints(img1, img1_keypoints, img1))
    cv2.imwrite('imagens/keypoints-2.png',
                    cv2.drawKeypoints(img2, img2_keypoints, img2))

    matcher = cv2.BFMatcher(cv2.NORM_L2, True)
    matches = matcher.match(img1_descriptors,img2_descriptors)

    point_map = np.array([
            [img1_keypoints[match.queryIdx].pt[0],
            img1_keypoints[match.queryIdx].pt[1],
            img2_keypoints[match.trainIdx].pt[0],
            img2_keypoints[match.trainIdx].pt[1]] for match in matches
        ])

    cv2.imwrite('imagens/matches.png', cv2.drawMatches(img1, img1_keypoints,
    img2, img2_keypoints, point_map))

    return point_map