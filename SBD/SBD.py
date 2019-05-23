from glob2 import glob
import sys

# if you want to import cv2, you must add path to sys.path
sys.path.append('/home/opencv-3.4.3/build/lib')

import cv2
# test cv2
# print cv2.__version__

import math
import copy
import os
import time

class SBD():

    def if_overlap(self, begin1, end1, begin2, end2):

        if begin1 > begin2:
            begin1, end1, begin2, end2 = begin2, end2, begin1, end1

        return end1 >= begin2
    def get_frame_hist(self, frame, bins_number):

        B_frame_hist = cv2.calcHist([frame], channels=[0], mask=None, ranges=[0.0, 255.0], histSize=[bins_number])
        G_frame_hist = cv2.calcHist([frame], channels=[1], mask=None, ranges=[0.0, 255.0], histSize=[bins_number])
        R_frame_hist = cv2.calcHist([frame], channels=[2], mask=None, ranges=[0.0, 255.0], histSize=[bins_number])

        return[B_frame_hist, G_frame_hist, R_frame_hist]

    def get_hist_chi_squa_diff(self, frame1, frame2, allpixels):
        bins_number = 64

        [B_frame1_hist, G_frame1_hist, R_frame1_hist] = self.get_frame_hist(frame1, bins_number)
        [B_frame2_hist, G_frame2_hist, R_frame2_hist] = self.get_frame_hist(frame2, bins_number)

        return (cv2.compareHist(B_frame1_hist, B_frame2_hist, method=cv2.HISTCMP_CHISQR) + \
               cv2.compareHist(G_frame1_hist, G_frame2_hist, method=cv2.HISTCMP_CHISQR) + \
               cv2.compareHist(R_frame1_hist, R_frame2_hist, method=cv2.HISTCMP_CHISQR))/allpixels


    def check_candidate_segments(self, candidate_segments, hard_cut_truth, gra_truth):

        missed_hard_cut = []

        missed_gra_cut = []

        h_i = 0
        g_i = 0

        miss_h = 0
        miss_g = 0

        i = 0
        while i < len(candidate_segments):


            if h_i < len(hard_cut_truth) and self.if_overlap(candidate_segments[i][0], candidate_segments[i][1], hard_cut_truth[h_i][0], hard_cut_truth[h_i][1]):
                h_i += 1
                i += 1
                continue
            elif h_i < len(hard_cut_truth) and candidate_segments[i][0] > hard_cut_truth[h_i][1]:
                miss_h += 1
                h_i += 1



            if g_i < len(gra_truth) and self.if_overlap(candidate_segments[i][0], candidate_segments[i][1], gra_truth[g_i][0], gra_truth[g_i][1]):
                g_i += 1
                i += 1
                continue
            elif g_i < len(gra_truth) and candidate_segments[i][0] > gra_truth[g_i][1]:
                miss_g += 1
                g_i += 1

            i += 1

        return [miss_h, miss_g]




    def get_candadite_segments(self, VideoPath):

        group_length = 10
        second_group_length = 5

        i_video = cv2.VideoCapture(VideoPath)

        # get width of this video
        wid = int(i_video.get(3))
        # get height of this video
        hei = int(i_video.get(4))
        # It save the number of frames in this video
        number_of_frames = int(i_video.get(7))

        # get the frame no. of one frame in this video (be used to normalize)
        AllPixels = wid * hei

        number_of_frames_in_group = int(math.ceil(number_of_frames / float(group_length)))

        diff_group_10 = []

        diff_group_5 = []

        i_video.set(1, 0)
        ret_group_first_frame, group_first_frame = i_video.read()

        # the diff between the 0 th and 10 th([0,10], [10,20], [20,30], ...) is larger than threshold
        candidate_segments_10 = []
        # the diff between the 5 th and 15 th([5,15], [15,25], [25,35], ...) is larger than threshold
        candidate_segments_5 = []

        for i in range(1, number_of_frames_in_group):

            i_video.set(1, i * group_length)

            ret_group_last_frame, group_last_frame = i_video.read()

            d = self.get_hist_chi_squa_diff(group_last_frame, group_first_frame, AllPixels)

            if d > 0.5:

                candidate_segments_10.append([(i - 1) * group_length, i * group_length])

                diff_group_10.append(d)

            group_first_frame = copy.deepcopy(group_last_frame)

        if number_of_frames % group_length == number_of_frames % second_group_length:
            group_5_15_number = number_of_frames_in_group - 1
        else:
            group_5_15_number = number_of_frames_in_group

        i_video.set(1, second_group_length)

        ret_group_first_frame, first_frame_5_to_15 = i_video.read()

        for i in range(1, group_5_15_number):

            if i * group_length + second_group_length >= number_of_frames:

                break

            i_video.set(1, i * group_length + second_group_length)

            ret_group_last_frame, Frames10_5_2 = i_video.read()

            d = self.get_hist_chi_squa_diff(first_frame_5_to_15, Frames10_5_2, AllPixels)

            if d > 0.5:

                candidate_segments_5.append([(i - 1) * group_length + second_group_length,

                                           i * group_length + second_group_length])
                diff_group_5.append(d)

            first_frame_5_to_15 = copy.deepcopy(Frames10_5_2)

        all_candidate_segments = []

        i = 0
        j = 0
        while i < len(candidate_segments_10) or j < len(candidate_segments_5):

            if candidate_segments_10[i][1] < candidate_segments_5[j][0]:
                all_candidate_segments.append(candidate_segments_10[i])
                i += 1
            elif candidate_segments_10[i][0] > candidate_segments_5[j][1]:
                all_candidate_segments.append(candidate_segments_5[j])
                j += 1
            else:
                if diff_group_10[i] > diff_group_5[j]:
                    all_candidate_segments.append(candidate_segments_10[i])
                else:
                    all_candidate_segments.append(candidate_segments_5[j])
                i += 1
                j += 1
            if i == len(candidate_segments_10) and j < len(candidate_segments_5):
                all_candidate_segments.extend(candidate_segments_5[j:])
                break
            elif j == len(candidate_segments_5) and i < len(candidate_segments_10):
                all_candidate_segments.extend(candidate_segments_10[i:])
                break

        return all_candidate_segments

    def get_labels_rai(self, label_path):

        hard_truth = []
        gra_truth = []

        with open(label_path) as f:
            all_lines = f.readlines()

        ground_truth = [[int(all_lines[0].strip().split('\t')[1])]]

        for i in range(1, len(all_lines) - 1):
            ground_truth[-1].extend([int(all_lines[i].strip().split('\t')[0])])
            ground_truth.append([int(all_lines[i].strip().split('\t')[1])])
        ground_truth[-1].extend([int(all_lines[-1].strip().split('\t')[0])])

        for i in ground_truth:
            if i[1]-i[0] == 1:
                hard_truth.append(i)
            else:
                gra_truth.append(i)

        return [hard_truth, gra_truth]

    def sbd_on_rai(self):


        current_dir = os.path.join(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1]))

        os.chdir('./videos')

        videos = glob('*.mp4')

        for i in videos:
            # if cmp(i, '8.mp4') == -1:
            #     continue
            print 'Now', i, ' is analyasing...'
            begin_time = time.time()


            all_candidate_segments = self.get_candadite_segments(i)

            [hard_truth, gra_truth] = self.get_labels_rai('./annotations/gt_' + i.split('.')[0] + '.txt')

            [missed_hard, missed_gra] = self.check_candidate_segments(all_candidate_segments, hard_truth, gra_truth)

            print 'missed hard no. is ', str(missed_hard), 'missed gra no. is ', str(missed_gra), '\n'

            print 'hard Recall is', str(float(len(hard_truth) - missed_hard) / len(hard_truth)), 'gra Recall is', str(float(len(gra_truth) - missed_gra) / len(gra_truth)), '\n'
            end_time = time.time()

            print 'the cost of time is ', str(end_time - begin_time), '\n'



if  __name__ == '__main__':

    test1 = SBD()
    test1.sbd_on_rai()