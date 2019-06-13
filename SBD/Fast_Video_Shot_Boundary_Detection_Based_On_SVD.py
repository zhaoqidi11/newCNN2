import cv2
import numpy as np
from copy import deepcopy

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

    def process_invalid_frame(self, ret, frame, index, sign, i_video):

        temp_i = index + sign
        while ret is False:
            i_video.set(1, temp_i)
            ret, frame = i_video.read()
            temp_i += sign
        return frame

    def get_valid_frame(self, i_video, index, sign):
        i_video.set(1, index)
        ret, frame = i_video.read()
        if ret:
            return frame
        else:
            return self.process_invalid_frame(ret, frame, index, sign, i_video)

    def candidate_segments(self, video_path):

        i_Video = cv2.VideoCapture(video_path)

        video_number = i_Video.get(7)

        length = 21

        group_number = (video_number - length) / (length - 1) + 1

        rest_first = -1

        rest_last = -1

        if length + (group_number - 1) * (length - 1) < video_number:

            rest_first = length + (group_number - 1) * (length - 1) - 1

            rest_last = video_number - 1

        d_length = []

        d_length_frame = {}

        first_frame_of_group = self.get_valid_frame(i_Video, i * (length - 1), 1)

        for i in range(group_number-1):

            last_frame_of_group = self.get_valid_frame(i_Video, (i + 1) * (length - 1), -1)

            d_length.append(self.get_pixel_diff(first_frame_of_group, last_frame_of_group))

            d_length_frame[i * (length - 1)]  = deepcopy(first_frame_of_group)

            first_frame_of_group = deepcopy(last_frame_of_group)

        d_length_frame[(group_number-1)*(length=1)] = first_frame_of_group

        if rest_first != -1:

            d_length_frame[rest_first] = self.get_valid_frame(i_Video, rest_first, 1)

            d_length_frame[rest_last] = self.get_valid_frame(i_Video, rest_last, -1)

            d_length.append(self.get_pixel_diff(d_length_frame[rest_first] , d_length_frame[rest_last]))

        else:

            d_length_frame[video_number-1] = self.get_valid_frame(i_Video, video_number-1, -1)

            d_length.append(self.get_pixel_diff( d_length_frame[(group_number-1)*(length - 1)], d_length_frame[video_number-1]))


        GroupNumber = 10

        global_mean = np.mean(d_length)

        NoOfGroup = group_number / GroupNumber

        Tl = []

        for i in range(NoOfGroup):


