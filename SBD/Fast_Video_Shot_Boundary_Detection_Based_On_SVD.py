import cv2
import numpy as np

class SVD():

    def convert_RBG_to_HSV(self, R,G,B):

        V = np.max([R, G, B])

        if V != 0:
            S = (V - np.min([R, G, B])/V)*255
        else:
            S = 0

        if V == R:
            H = 30*(G - B) / S
        elif V==G:
            H = 60 + 30*(B - R) /S
        else:
            H = 120 + 30 * (R - G) / S

        if H < 0:
            H += 180

        return [H, S, V]

    def get_pixel_diff(self, frame1, frame2):

        return np.sum(np.abs(frame1 - frame2))