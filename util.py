import cv2
import numpy as np

# Function to draw lines to indicate correspondence
def draw_inlines(inliers, img_1, img_2):

        # Define alguns parametros para desenhar os inliers
        lineColor = [0, 255, 0] # A cor da linha foi definida como verde
        ptColor = [0, 0, 0] # A cor do ponto foi definida como preto
        lineThickness = 2
        radius = 5
        circThickness = 2

        # Shape function returns a tuple of the number of rows, columns, and channels
        h, w, _ = img_1.shape 

        # Utiliza funcao hstack do numpy para empilhar os arrays das duas imagens
        imgStack = np.hstack((img_1, img_2))

        # Loop para passar percorrer os inliers e desenhar os circulos e linhas nas duas imagens
        for x1, y1, x2, y2 in inliers:
            x1_d = int(round(x1))
            y1_d = int(round(y1))

            x2_d = int(round(x2) + w)
            y2_d = int(round(y2))

            cv2.circle(imgStack, (x1_d, y1_d), radius=radius, color=ptColor,
                       thickness=circThickness, lineType=cv2.LINE_AA)

            cv2.circle(imgStack, (x2_d, y2_d), radius=radius, color=ptColor,
                       thickness=circThickness, lineType=cv2.LINE_AA)

            cv2.line(imgStack, (x1_d, y1_d), (x2_d, y2_d), color=lineColor,
                     thickness=lineThickness)

        # Salva a imagem com as marcacoes
        cv2.imwrite('results\\matches_RANSAC.png', imgStack)