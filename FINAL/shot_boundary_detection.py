
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
import shutil
import numpy as np
from read_binary_blob import read_binary_blob
from copy import deepcopy

class SBD():


    def if_overlap(self, begin1, end1, begin2, end2, if_include = False):

        if begin1 > begin2:
            begin1, end1, begin2, end2 = begin2, end2, begin1, end1

        return end1 >= begin2 if if_include else end1 > begin2

    def get_pixel_diff(self, frame1, frame2, all_pixels):

        return np.sum(np.power(np.abs(frame1 - frame2), 2)) / float(all_pixels)

    def get_frame_hist(self, frame, bins_number):

        B_frame_hist = cv2.calcHist([frame], channels=[0], mask=None, ranges=[0.0, 255.0], histSize=[bins_number])
        G_frame_hist = cv2.calcHist([frame], channels=[1], mask=None, ranges=[0.0, 255.0], histSize=[bins_number])
        R_frame_hist = cv2.calcHist([frame], channels=[2], mask=None, ranges=[0.0, 255.0], histSize=[bins_number])

        return[B_frame_hist, G_frame_hist, R_frame_hist]

    def get_frame_hist_diff(self, frame1, frame2, allpixels, diff_method ='chi_squa', bins_number = 64):

        [B_frame1_hist, G_frame1_hist, R_frame1_hist] = self.get_frame_hist(frame1, bins_number)
        [B_frame2_hist, G_frame2_hist, R_frame2_hist] = self.get_frame_hist(frame2, bins_number)

        if cmp(diff_method, 'manh'):

            return np.sum(np.abs(B_frame1_hist - B_frame2_hist) + np.abs(G_frame1_hist - G_frame2_hist) + np.abs(R_frame1_hist - R_frame2_hist)) / allpixels

        elif cmp(diff_method, 'chi_squa'):

            return (cv2.compareHist(B_frame1_hist, B_frame2_hist, method=cv2.HISTCMP_CHISQR) + \
                    cv2.compareHist(G_frame1_hist, G_frame2_hist, method=cv2.HISTCMP_CHISQR) + \
                    cv2.compareHist(R_frame1_hist, R_frame2_hist, method=cv2.HISTCMP_CHISQR)) / allpixels

    def mkdir(self, path):

        if os.path.exists(path):
            shutil.rmtree(path)
            os.mkdir(path)
        else:
            os.mkdir(path)

    def generate_images_sequence(self, tmp_folder_path, video_path, candidate_segments):

        i_video = cv2.VideoCapture(video_path)

        for i in candidate_segments:

            index = i[0]

            save_path = os.sep.join([tmp_folder_path, str(index+1).zfill(6)])

            os.mkdir(save_path)

            while index < i[1]:

                cv2.imwrite(os.sep.join([save_path, '.'.join([str(index+1).zfill(6), 'jpg'])]), self.get_valid_frame(i_video, index, 1))

                index += 1

            cv2.imwrite(os.sep.join([save_path, '.'.join([str(i[1]+1).zfill(6), 'jpg'])]), self.get_valid_frame(i_video, i[1], -1))

    def create_tmp_images(self, current_dir, video_path, segments):

        tmp_folder_path = os.sep.join([current_dir, 'tmp'])

        self.mkdir(tmp_folder_path)

        # save images extracted from video

        temp_folder_path = os.sep.join([tmp_folder_path, 'images'])

        temp_out_folder_path = os.sep.join([tmp_folder_path, 'features'])

        temp_folder_list_path = os.sep.join([tmp_folder_path, 'to_be_extracted.list'])

        temp_out_folder_list_path = os.sep.join([tmp_folder_path, 'output.list'])

        self.mkdir(temp_folder_path)

        self.mkdir(temp_out_folder_path)

        if os.path.exists(temp_folder_list_path):
            os.remove(temp_folder_list_path)
        if os.path.exists(temp_out_folder_list_path):
            os.remove(temp_out_folder_list_path)

        temp_list = []
        temp_out_list = []

        for s in segments:

            temp_list.append(' '.join([video_path, str(s[0]), '0']) + '\n')
            temp_out_list.append(os.sep.join([tmp_folder_path, str(s[0]).zfill(6)]) + '\n')

        with open(temp_folder_list_path, 'w') as f:
            f.writelines(temp_list)

        with open(temp_out_folder_list_path, 'w') as f:
            f.writelines(temp_out_list)

        return tmp_folder_path, temp_folder_list_path, temp_out_folder_list_path

    def extract_features(self, video_path, current_dir, candidate_segments):

        tmp_folder_path, temp_folder_list_path, temp_out_folder_list_path = self.create_tmp_images(current_dir, video_path, candidate_segments)

        extract_tools_path = '/home/newC3D2/C3D/C3D-v1.1/build/tools/extract_image_features'

        model_path = 'ModelFiles'

        model_file = os.path.join(current_dir, model_path, 'PreResNet18.prototxt')

        model_weight = os.path.join(current_dir, model_path, 'PreResNet18_iter_35000.caffemodel')

        gpu_id = '1'

        batch_size = '24'

        batch_num = str(int(math.ceil(float(len(candidate_segments)) / int(batch_size))))

        feature1 = 'fc8-new'

        feature2 = 'prob'

        shell = ' '.join(['GLOG_logtostderr=1', extract_tools_path, model_file, model_weight, gpu_id, batch_size, batch_num, temp_out_folder_list_path, feature1, feature2])

        os.system(shell)

        return tmp_folder_path

    def detect_gra(self, candidate_segment):

        if len(candidate_segment) == 0:

            return 0

        gra_cut = candidate_segment[0]

        for i in candidate_segment[1:]:

            print 'a'

    def detect_hard(self, candidate_segment, video_path):

        d = []

        i_Video = cv2.VideoCapture(video_path)

        frame_first = self.get_valid_frame(i_Video, candidate_segment[0], 1)


        for i in range(candidate_segment[0]+1,candidate_segment[1]):

            frame_next = self.get_valid_frame(i_Video, i, -1)

            d.append(0.5 * self.get_frame_hist_diff(frame_first, frame_next, frame_first.shape[0] * frame_first.shape[1]) + 0.5 * self.get_pixel_diff(frame_first, frame_next, frame_first.shape[0] * frame_first.shape[1]))

            frame_first = copy.deepcopy(frame_next)


        return [candidate_segment[0]+np.argmax(d), candidate_segment[0]+np.argmax(d)+1]

    def get_location(self, candidate_segments, video_path):


        hard_cut = []


        for i in range(len(candidate_segments)):

            hard_cut.append(self.detect_hard(candidate_segments[i], video_path))


        return hard_cut







    def process_invalid_frame(self, ret, frame, index, sign, i_video):

        temp_i = index + sign

        num = i_video.get(7)

        while ret is False and temp_i<num:
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

    def remove_elements_from_list(self, to_be_removed_elements, list):

        new_list = []

        if len(to_be_removed_elements) == 0:

            return list


        if to_be_removed_elements[0] > 0 :

            new_list = list[0:to_be_removed_elements[0]]

        begin = to_be_removed_elements[0] + 1

        for i in to_be_removed_elements[1:]:

            new_list.extend(list[begin:i])

            begin = i + 1

        if begin < len(list):

            new_list.extend(list[begin:])

        return new_list






    def remove_invalid_segments(self, candidate_segments, video_path):

        temporal_window = 8

        invalid_index = []

        i_Video = cv2.VideoCapture(video_path)

        for i in range(len(candidate_segments)):

            first_frame = self.get_valid_frame(i_Video, candidate_segments[i][0], 1)

            last_frame = self.get_valid_frame(i_Video, candidate_segments[i][1], -1)

            if first_frame is None or last_frame is None:

                continue

            # print self.get_hist_manh_diff(first_frame, last_frame, first_frame.shape[1] * first_frame.shape[0]), '\n'

            # if 0.5*self.get_pixel_diff(first_frame, last_frame, first_frame.shape[1] * first_frame.shape[0]) + \
            #         0.5*self.get_hist_chi_squa_diff(first_frame, last_frame, first_frame.shape[1] * first_frame.shape[0]) < 200:

            if self.get_hist_chi_squa_diff(first_frame, last_frame, first_frame.shape[1] * first_frame.shape[0]) < 2:

            # if self.get_hist_manh_diff(cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV), cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV), first_frame.shape[1] * first_frame.shape[0]) <0.5:

            # if self.get_hist_manh_diff(first_frame, last_frame, first_frame.shape[1] * first_frame.shape[0]) < 1:

                invalid_index.append(i)

        return self.remove_elements_from_list(invalid_index, candidate_segments)

    def remove_invalid_segments2(self, candidate_segments, video_path):

        temporal_window = 8

        invalid_index = []

        i_Video = cv2.VideoCapture(video_path)

        for i in range(len(candidate_segments)):

            first_frame = self.get_valid_frame(i_Video, candidate_segments[i][0], 1)

            last_frame = self.get_valid_frame(i_Video, candidate_segments[i][1], -1)

            # print self.get_hist_manh_diff(first_frame, last_frame, first_frame.shape[1] * first_frame.shape[0]), '\n'

            # if 0.5*self.get_pixel_diff(first_frame, last_frame, first_frame.shape[1] * first_frame.shape[0]) + \
            #         0.5*self.get_hist_chi_squa_diff(first_frame, last_frame, first_frame.shape[1] * first_frame.shape[0]) < 200:

            if self.get_hist_chi_squa_diff(first_frame, last_frame, first_frame.shape[1] * first_frame.shape[0]) < 0.5:

            # if self.get_hist_manh_diff(cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV), cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV), first_frame.shape[1] * first_frame.shape[0]) <0.5:

            # if self.get_hist_manh_diff(first_frame, last_frame, first_frame.shape[1] * first_frame.shape[0]) < 1:

                invalid_index.append(i)

        return self.remove_elements_from_list(invalid_index, candidate_segments)

    def calculate_all_segments_index(self, video_path):

        i_video = cv2.VideoCapture(video_path)

        number_of_frames_in_video = int(i_video.get(7))

        temporal_window = 8

        candidate_segments = []

        group_number = 1 + (number_of_frames_in_video - 2 * temporal_window) / temporal_window

        for i in range(group_number):

            candidate_segments.append([i * temporal_window, i * temporal_window + 2 * temporal_window - 1])

        return candidate_segments

    def recall_pre_f1(self, a, b, c):
        recall = float(a)/b if b!=0 else 0
        precison = float(a)/c if c!=0 else 0
        f1 = float(2*recall*precison) / (recall + precison) if (recall + precison)!=0 else 0

        return precison, recall, f1



    def get_union_cut(self, set1, set2):

        cnt = 0
        # tmp_set = []
        for s1 in set1:
            for s2 in set2:
                if self.if_overlap(s1[0], s1[1], s2[0], s2[1]): # and s2[1] - s2[0] > 1:
                    # tmp_set.append(s1)
                    cnt += 1
                    break

        return cnt

    def eval(self, cut, cut_truth, gra, gra_truth):

        cut_correct = self.get_union_cut(cut_truth, cut)
        gra_correct = self.get_union_cut(gra_truth, gra)

        all_correct = self.get_union_cut(cut_truth+gra_truth, cut+gra)

        # return self.recall_pre_f1(gra_correct, len(gra), len(gra_truth)), self.recall_pre_f1(cut_correct, len(cut), len(cut_truth)), self.recall_pre_f1(all_correct, len(cut_truth+gra_truth),len(cut+gra))
        return cut_correct, gra_correct, all_correct






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

    def get_hard_and_gra_segments(self, tmp_folder_path):

        length = 15

        tmp_have_been_extracted_feature_list = os.sep.join([tmp_folder_path, 'output.list'])

        with open(tmp_have_been_extracted_feature_list, 'r') as f:

            all_segments = f.readlines()

        all_segments = [i.strip() for i in all_segments]

        suffix = '.prob'

        hard_segments = []

        gra_segments = []

        threshold = 0.7

        for i in all_segments:

            (s, prob) = read_binary_blob(i + suffix)

            if np.argmax(prob) == 1 and max(prob) > threshold:

                gra_segments.append([int(i.split(os.sep)[-1]), int(i.split(os.sep)[-1]) + length])

            elif np.argmax(prob) == 2 and max(prob) > threshold:

                if len(hard_segments) > 0 and self.if_overlap(hard_segments[-1][0], hard_segments[-1][1], int(i.split(os.sep)[-1]), int(i.split(os.sep)[-1]) + length):

                    hard_segments[-1][0] = int(i.split(os.sep)[-1])

                else:

                    hard_segments.append([int(i.split(os.sep)[-1]), int(i.split(os.sep)[-1]) + length])

        return [hard_segments, gra_segments]

    def sbd_on_rai(self):

        current_dir = os.path.join(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1]))

        video_path = 'videos'

        videos = glob(os.sep.join([video_path, '*.mp4']))

        log_file_path = 'logs'

        label_path = os.path.join(current_dir, video_path, 'annotations', 'gt_')

        cut_correct = 0

        gra_correct = 0

        all_correct = 0

        cut_t = 0

        gra_t = 0

        all_t = 0

        cut_n = 0

        gra_n = 0

        all_n = 0

        for video in videos:

            # if cmp(video.split(os.sep)[-1], '9.mp4') != 0:
            #
            #     continue

            print 'Now', video.split(os.sep)[-1], ' is analyasing...'

            begin_time = time.time()

            # get labels
            [hard_truth, gra_truth] = self.get_labels_rai(
                label_path + video.split(os.sep)[-1].split('.')[0] + '.txt')

            # get all candidate segments (to be sent to CNN)

            segments = self.calculate_all_segments_index(video)

            tmp_folder_path = self.extract_features(video, current_dir, segments)

            [res_hard, gra_segments] = self.get_hard_and_gra_segments(tmp_folder_path)

            gra_segments = self.remove_invalid_segments(gra_segments, video)

            res_gra = [deepcopy(gra_segments[0])]

            for gra in gra_segments[1:]:

                if self.if_overlap(res_gra[-1][0], res_gra[-1][1], gra[0], gra[1]):

                    res_gra[-1][1] = gra[1]

                else:

                    res_gra.append(deepcopy(gra))

            res_hard = self.get_location(res_hard, video)

            cut_t += len(hard_truth)
            gra_t += len(gra_truth)
            all_t += len(hard_truth) + len(gra_truth)

            cut_correct_n, gra_correct_n, all_correct_n = self.eval(res_hard, hard_truth, res_gra, gra_truth)

            cut_correct += cut_correct_n
            gra_correct += gra_correct_n
            all_correct += cut_correct_n+gra_correct_n

            cut_n += len(res_hard)
            gra_n += len(res_gra)
            all_n += len(res_gra+res_hard)

            result = ['hard_prob_thresh: 0.7, gra_prob_thresh: 0.7, chi_squr_gra_thresh: 2,\n', str(video) + '\n',
                      str(gra_correct_n), '\t', str(len(res_gra)), '\t', str(len(gra_truth)), '\n', str(cut_correct_n), '\t', str(len(res_hard)),
                      '\t', str(len(hard_truth)), '\n']

            with open (log_file_path, 'a') as f:
                f.writelines(result)

            end_time = time.time()

            print 'the cost of time is ', str(end_time - begin_time), '\n'

        result = ['All: ', str(self.recall_pre_f1(all_correct, all_t, all_n))]

        with open(log_file_path, 'a') as f:

            f.writelines(result)



if  __name__ == '__main__':

    test1 = SBD()
    test1.sbd_on_rai()

