import cv2
import numpy as np
from copy import deepcopy
import math
import os
from glob2 import glob
import time

class SVD():
    # convert RGB to HSV
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
    # process the invalid frame in a segment
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

            group_number += 1

        d_length = []

        # save the frame
        frame = {}

        all_segments = []

        first_frame_of_group = self.get_valid_frame(i_Video, 0, 1)

        for i in range(group_number-1):

            last_frame_of_group = self.get_valid_frame(i_Video, (i + 1) * (length - 1), -1)

            d_length.append(self.get_pixel_diff(first_frame_of_group, last_frame_of_group))

            all_segments.append([i*(length-1), (i+1)*(length-1)])

            frame[i * (length - 1)]  = deepcopy(first_frame_of_group)

            first_frame_of_group = deepcopy(last_frame_of_group)

        frame[(group_number-1)*(length-1)] = first_frame_of_group

        if rest_first != -1:

            # frame[rest_first] = self.get_valid_frame(i_Video, rest_first, 1)

            frame[rest_last] = self.get_valid_frame(i_Video, rest_last, -1)

            d_length.append(self.get_pixel_diff(frame[rest_first] , frame[rest_last]))

            all_segments.append([rest_first, rest_last])

        else:

            frame[video_number-1] = self.get_valid_frame(i_Video, video_number-1, -1)

            d_length.append(self.get_pixel_diff(frame[length + (group_number - 1) * (length - 1) - 1], frame[video_number-1]))

            all_segments.append([length + (group_number - 1) * (length - 1) - 1, video_number - 1])

        candidate_segments = []

        GroupNumber = 10

        global_mean = np.mean(d_length)

        NoOfGroup = group_number / GroupNumber

        Tl = []

        a = 0.7

        for i in range(NoOfGroup):

            Now_Group = d_length[i*GroupNumber:i+GroupNumber]

            local_mean = np.mean(Now_Group)

            Tl.append(local_mean + a*(1+math.log(global_mean/local_mean))*np.std(local_mean))

            for j in Now_Group:

                if j > Tl[-1]:

                    candidate_segments.append([0])

    def get_labels_TRECViD(self, label_path):

        hard_truth = []
        gra_truth = []

        with open(label_path) as f:

            xmlfile = f.readlines()

        for i in range(len(xmlfile)):
            if 'CUT' in xmlfile[i]:
                hard_truth.append([int(xmlfile[i].split('"')[-4]), int(xmlfile[i].split('"')[-2])])
            elif 'DIS' in xmlfile[i] or 'OTH' in xmlfile[i]:
                gra_truth.append([int(xmlfile[i].split('"')[-4]), int(xmlfile[i].split('"')[-2])])

        return [hard_truth, gra_truth]

    def sbd_on_trecvid(self):

        current_dir = os.path.join(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1]))

        path_to_frames = '/home/DSBD_Test/segments'

        videos = glob(os.sep.join([path_to_frames, '*']))

        labels_path = '/home/t2007ref'

        for i in videos:

            # if cmp(i.split(os.sep)[-1], '4.mp4') != 0:
            #     continue

            print 'Now', i.split(os.sep)[-1], ' is analyasing...'

            begin_time = time.time()

            self.candidate_segments(i)

            end_time = time.time()

            print 'the cost of time is ', str(end_time - begin_time), '\n'

if __name__ == '__main__':

    test1 =SVD()
    test1.sbd_on_trecvid()