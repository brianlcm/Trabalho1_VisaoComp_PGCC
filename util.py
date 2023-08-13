import cv2
import numpy as np

def draw_lines(corr_pts, img_1, img_2, line_color, pt_color):
        """
        Function to draw lines to indicate correspondence
        :param corr_pts: nd array of points from ing_1 to img2 [x1, y1, x2, y2] rows
        :param img_1: RGB ndarray for image 1
        :param img_2: RGB ndarray for image 2
        :param save_path: Full path to save result image
        :param line_color: color of line. 3 tuple RGB
        :param pt_color: color of point marking coorresponding points, 3 tuple of RGB
        :return:
        """
        _line_thickness = 2
        _radius = 5
        _circ_thickness = 2
        h, w, _ = img_1.shape

        img_stack = np.hstack((img_1, img_2))

        for x1, y1, x2, y2 in corr_pts:
            x1_d = int(round(x1))
            y1_d = int(round(y1))

            x2_d = int(round(x2) + w)
            y2_d = int(round(y2))

            cv2.circle(img_stack, (x1_d, y1_d), radius=_radius, color=pt_color,
                       thickness=_circ_thickness, lineType=cv2.LINE_AA)

            cv2.circle(img_stack, (x2_d, y2_d), radius=_radius, color=pt_color,
                       thickness=_circ_thickness, lineType=cv2.LINE_AA)

            cv2.line(img_stack, (x1_d, y1_d), (x2_d, y2_d), color=line_color,
                     thickness=_line_thickness)

        cv2.imwrite('images\\matches_RANSAC.png', img_stack)