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

    def get_valid_frame(self, frame_path, index, sign):

        frame = cv2.imread(frame_path[index])

        while type(frame) is  None:

            index += sign

            frame = cv2.imread(frame_path[index + sign])

        else:

            return frame

    def candidate_segments(self, video_path):

        frames_path = glob(os.path.join(video_path, '*.jpg'))

        frames_path_suffix = frames_path[0].split(os.sep)[:-1]

        video_number = len(frames_path)

        frames_path = [os.path.join(os.sep.join(frames_path_suffix),(str(i)+'.jpg')) for i in range(video_number)]

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

        all_segments_flag = []

        first_frame_of_group = self.get_valid_frame(frames_path, 0, 1)

        frame[0] = deepcopy(first_frame_of_group)

        for i in range(group_number-1):

            last_frame_of_group = self.get_valid_frame(frames_path, (i + 1) * (length - 1), -1)

            frame[(i+1)*(length-1)] = deepcopy(last_frame_of_group)

            d_length.append(self.get_pixel_diff(first_frame_of_group, last_frame_of_group))

            all_segments.append([i*(length-1), (i+1)*(length-1)])

            all_segments_flag.append(0)

            first_frame_of_group = deepcopy(last_frame_of_group)

        if rest_first != -1:

            # frame[rest_first] = self.get_valid_frame(i_Video, rest_first, 1)

            frame[rest_last] = self.get_valid_frame(frames_path, rest_last, -1)

            d_length.append(self.get_pixel_diff(frame[rest_first], frame[rest_last]))

            all_segments.append([rest_first, rest_last])

        else:

            frame[video_number-1] = self.get_valid_frame(frames_path, video_number-1, -1)

            d_length.append(self.get_pixel_diff(frame[length + (group_number - 1) * (length - 1) - 1], frame[video_number-1]))

            all_segments.append([length + (group_number - 1) * (length - 1) - 1, video_number - 1])

        all_segments_flag.append(0)


        GroupNumber = 10

        global_mean = np.mean(d_length)

        NoOfGroup = group_number / GroupNumber

        Tl = []

        a = 0.7

        for i in range(NoOfGroup-1):

            Now_Group = d_length[i*GroupNumber:i*GroupNumber + GroupNumber]

            local_mean = np.mean(Now_Group)

            Tl.append(local_mean + a*(1+math.log(global_mean/local_mean))*np.std(local_mean))

            for j in range(len(Now_Group)):

                if Now_Group[j] > Tl[-1]:

                    all_segments_flag[i*GroupNumber+j] = 1

        local_mean = np.mean(d_length[(NoOfGroup-1)*GroupNumber:len(d_length)])

        Tl.append(local_mean + a*(1+math.log(global_mean/local_mean))*np.std(local_mean))

        for j in range(len(d_length[(NoOfGroup-1)*GroupNumber:len(d_length)])):

            if d_length[(NoOfGroup-1)*GroupNumber + j] > Tl[-1]:

                all_segments_flag[(NoOfGroup-1)*GroupNumber + j] = 1

        for i in range(1, len(d_length)-1):

            if ((d_length[i] > 3*d_length[i-1]) or (d_length[i] > 3*d_length[i+1])) and (d_length[i] > 0.8*global_mean):

                all_segments_flag[i] = 1

        gra_segments = []

        second_segments = []

        second_segments_flag = []

        for i in range(len(all_segments_flag)):

            if all_segments_flag[i] == 1:

                middle_frame = self.get_valid_frame(frames_path, all_segments[i][0] + (length-1)/2,1)

                d_f = self.get_pixel_diff(frame[all_segments[i][0]], middle_frame)

                d_b = self.get_pixel_diff(frame[all_segments[i][1]], middle_frame)

                if d_f / d_b > 1.2:

                    second_segments.append([all_segments[i][0], all_segments[i][0] + (length-1)/2])

                    frame[all_segments[i][0] + (length-1)/2] = deepcopy(middle_frame)

                elif d_b / d_f > 1.2:

                    second_segments.append([all_segments[i][0] + (length-1)/2, all_segments[i][1]])

                    frame[all_segments[i][0] + (length-1)/2] = deepcopy(middle_frame)

                elif d_f / d_length[i] < 0.3 and d_b / d_length[i] < 0.3:

                    continue

                else:

                    gra_segments.append(all_segments[i])

        print 'a'




        print 'a'





        print 'a'

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

        path_to_frames = '/home/t2007'

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