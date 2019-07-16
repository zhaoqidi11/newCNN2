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

    def get_frame_hist(self, frame, bins_number):

        B_frame_hist = cv2.calcHist([frame], channels=[0], mask=None, ranges=[0.0, 255.0], histSize=[bins_number])
        G_frame_hist = cv2.calcHist([frame], channels=[1], mask=None, ranges=[0.0, 255.0], histSize=[bins_number])
        R_frame_hist = cv2.calcHist([frame], channels=[2], mask=None, ranges=[0.0, 255.0], histSize=[bins_number])

        return[B_frame_hist, G_frame_hist, R_frame_hist]


    # Get the Manhattan Distance
    def get_hist_manh_diff(self, frame1, frame2, allpixels):

        bins_number = 32

        [B_frame1_hist, G_frame1_hist, R_frame1_hist] = self.get_frame_hist(frame1, bins_number)
        [B_frame2_hist, G_frame2_hist, R_frame2_hist] = self.get_frame_hist(frame2, bins_number)

        return np.sum(np.abs(B_frame1_hist - B_frame2_hist) + np.abs(G_frame1_hist - G_frame2_hist) + np.abs(R_frame1_hist - R_frame2_hist)) / allpixels


    def get_hist_chi_squa_diff(self, frame1, frame2, allpixels):

        bins_number = 64

        [B_frame1_hist, G_frame1_hist, R_frame1_hist] = self.get_frame_hist(frame1, bins_number)
        [B_frame2_hist, G_frame2_hist, R_frame2_hist] = self.get_frame_hist(frame2, bins_number)

        return (cv2.compareHist(B_frame1_hist, B_frame2_hist, method=cv2.HISTCMP_CHISQR) + \
               cv2.compareHist(G_frame1_hist, G_frame2_hist, method=cv2.HISTCMP_CHISQR) + \
               cv2.compareHist(R_frame1_hist, R_frame2_hist, method=cv2.HISTCMP_CHISQR))/allpixels



    def if_overlap(self, begin1, end1, begin2, end2):

        if begin1 > begin2:
            begin1, end1, begin2, end2 = begin2, end2, begin1, end1

        return end1 >= begin2

    def if_overlap2(self, begin1, end1, begin2, end2):

        if begin1 > begin2:
            begin1, end1, begin2, end2 = begin2, end2, begin1, end1

        return end1 > begin2

    def get_pixel_diff(self, frame1, frame2, all_pixels):

        return np.sum(np.power(np.abs(frame1 - frame2), 2)) / float(all_pixels)


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

    def check_candidate_segments2(self, candidate_segments, hard_truth, gra_truth):

        miss_hard = []
        miss_gra = []

        for i in range(len(hard_truth)):
            for j in range(len(candidate_segments)):
                if candidate_segments[j][1] < hard_truth[i][0]:
                    continue
                if self.if_overlap(candidate_segments[j][0], candidate_segments[j][1], hard_truth[i][0], hard_truth[i][1]):
                    break
                if candidate_segments[j][0] > hard_truth[i][1]:
                    miss_hard.append(hard_truth[i])
                    break

        for i in range(len(gra_truth)):
            for j in range(len(candidate_segments)):
                if candidate_segments[j][1] < gra_truth[i][0]:
                    continue
                if self.if_overlap(candidate_segments[j][0], candidate_segments[j][1], gra_truth[i][0], gra_truth[i][1]):
                    break
                if candidate_segments[j][0] > gra_truth[i][1]:
                    miss_gra.append(gra_truth[i])
                    break

        return [len(miss_hard), len(miss_gra)]

    def check_candidate_segments3(self, candidate_segments, hard_truth, gra_truth):


       candidate_segments_label = []


       for i in range(len(candidate_segments)):

           for j in range(len(hard_truth)):

               if candidate_segments[i][0] > hard_truth[j][1] and j < len(hard_truth)-1:
                   continue
               elif self.if_overlap(candidate_segments[i][0], candidate_segments[i][1], hard_truth[j][0], hard_truth[j][1]):
                   candidate_segments_label.append(2)
                   break
               elif candidate_segments[i][1] < hard_truth[j][0]:
                   candidate_segments_label.append(0)
                   break

               if j == len(hard_truth)-1:
                   candidate_segments_label.append(0)

       for i in range(len(candidate_segments)):

           for j in range(len(gra_truth)):

               if candidate_segments[i][0] > gra_truth[j][1]:
                   continue
               elif self.if_overlap(candidate_segments[i][0], candidate_segments[i][1], gra_truth[j][0],
                                    gra_truth[j][1]):
                   candidate_segments_label[i] = 1
                   break
               elif candidate_segments[i][1] < gra_truth[j][0]:
                   break



       candidate_segments_label_save = [str(i)+'\n' for i in candidate_segments_label]

       with open('./label.txt', 'w') as f:
           f.writelines(candidate_segments_label_save)

       return candidate_segments_label









    def generate_images_sequence(self, tmp_folder_path, video_path, C3D_segments):


        for i in C3D_segments:

            index = i[0]

            save_path = os.sep.join([tmp_folder_path, str(index+1).zfill(6)])

            os.mkdir(save_path)

            while index < i[1]:

                cv2.imwrite(os.sep.join([save_path, '.'.join([str(index+1).zfill(6), 'jpg'])]), self.get_valid_frame(video_path, index))

                index += 1

            cv2.imwrite(os.sep.join([save_path, '.'.join([str(i[1]+1).zfill(6), 'jpg'])]), self.get_valid_frame(video_path, i[1]))



    def generate_candidate_segments_to_C3D(self, candidate_segments, num_of_frames_in_video, number):

        length = (candidate_segments[0][1] - candidate_segments[0][0]) / 2

        C3D_segments_begin = {}

        for i in candidate_segments:

            for j in range(-1,number-1):

                if i[0] + j * length < 0:

                    C3D_segments_begin[0] = 16
                    C3D_segments_begin[8] = 24
                    C3D_segments_begin[16] = 32
                    break

                elif i[1] + length > num_of_frames_in_video:

                    C3D_segments_begin[num_of_frames_in_video-32] = num_of_frames_in_video -16
                    C3D_segments_begin[num_of_frames_in_video-24] = num_of_frames_in_video - 8
                    C3D_segments_begin[num_of_frames_in_video-16] = num_of_frames_in_video
                    break
                elif not C3D_segments_begin.has_key(i[0] + j*length):
                    C3D_segments_begin[i[0] + j*length] = i[0] + (j+2)*length
                else:
                    continue

        return [[i, i+16] for i in sorted(C3D_segments_begin)]



    def mkdir(self, path):

        if os.path.exists(path):
            shutil.rmtree(path)
            os.mkdir(path)
        else:
            os.mkdir(path)

    def get_candidate_segments_image(self, current_dir, video_path, candidate_segments, number_of_frames_in_video):

        temp_folder_path = os.sep.join([current_dir , 'tmp_video_images'])

        temp_out_folder_path = os.sep.join([current_dir , 'tmp_out_video_images'])

        temp_folder_list_path = os.sep.join([current_dir , 'tmp_list.prefix'])

        temp_out_folder_list_path = os.sep.join([current_dir , 'tmp_output_list.list'])

        self.mkdir(temp_folder_path)
        self.mkdir(temp_out_folder_path)

        if os.path.exists(temp_folder_list_path):
            os.remove(temp_folder_list_path)
        if os.path.exists(temp_out_folder_list_path):
            os.remove(temp_out_folder_list_path)

        # C3D_segments = self.generate_candidate_segments_to_C3D(candidate_segments, number_of_frames_in_video, 3)

        self.generate_images_sequence(temp_folder_path, video_path, candidate_segments)

        temp_list = []
        temp_out_list = []

        for i in candidate_segments:

            temp_list.append(' '.join([os.sep.join([temp_folder_path, str(i[0]+1).zfill(6), '']), str(i[0]+1), '0']) + '\n')
            temp_out_list.append(os.sep.join([temp_out_folder_path, str(i[0]+1).zfill(6),str(i[0]+1).zfill(6)]) + '\n')

        with open(temp_folder_list_path, 'w') as f:
            f.writelines(temp_list)

        with open(temp_out_folder_list_path, 'w') as f:
            f.writelines(temp_out_list)

        return candidate_segments

    def detect_gra(self, candidate_segment):

        if len(candidate_segment) == 0:

            return 0

        gra_cut = candidate_segment[0]

        for i in candidate_segment[1:]:

            print 'a'

    def detect_hard(self, candidate_segment, video_path, all_pixels):

        d = []

        frame_first = self.get_valid_frame(video_path, candidate_segment[0])


        for i in range(candidate_segment[0]+1,candidate_segment[1]):

            frame_next = self.get_valid_frame(video_path, i)

            d.append(0.5 * self.get_hist_chi_squa_diff(frame_first, frame_next, all_pixels) + 0.5 * self.get_pixel_diff(frame_first, frame_next, all_pixels))

            frame_first = copy.deepcopy(frame_next)


        return [candidate_segment[0]+np.argmax(d), candidate_segment[0]+np.argmax(d)+1]

    def valid_candidate_segments(self, candidate_segments, candidate_segments_label):

        new_candidate_segments = []

        new_candidate_segments_label = []

        begin = 0

        while candidate_segments_label[begin] == 0:

            begin += 1

        old_candidate_segment = candidate_segments[begin]

        old_candidate_label = candidate_segments_label[begin]

        new_candidate_segments.append(copy.deepcopy(candidate_segments[begin]))

        new_candidate_segments_label.append(candidate_segments_label[begin])

        for i in range(begin, len(candidate_segments)):

            if candidate_segments_label[i] == 0:

                continue

            elif candidate_segments_label[i] != old_candidate_label:

                new_candidate_segments.append(candidate_segments[i])

                new_candidate_segments_label.append(copy.deepcopy(candidate_segments_label[i]))

            elif self.if_overlap2(candidate_segments[i][0], candidate_segments[i][1], old_candidate_segment[0], old_candidate_segment[1]):

                if old_candidate_label == 1:

                    new_candidate_segments[-1] = [np.min([new_candidate_segments[-1][0], old_candidate_segment[0]]), np.max([new_candidate_segments[-1][1], old_candidate_segment[1]])]

                else:

                    new_candidate_segments[-1] = [np.max([new_candidate_segments[-1][0], old_candidate_segment[0]]), np.min([new_candidate_segments[-1][1], old_candidate_segment[1]])]

            else:

                new_candidate_segments.append(copy.deepcopy(candidate_segments[i]))

                new_candidate_segments_label.append(candidate_segments_label[i])

            old_candidate_segment = candidate_segments[i]

            old_candidate_label = candidate_segments_label[i]


        return [new_candidate_segments, new_candidate_segments_label]







    def get_location(self, candidate_segments, candidate_segments_label, video_path, all_pixels, hard_truth):

        [new_candidate_segments, new_candidate_segments_label] = self.valid_candidate_segments(candidate_segments, candidate_segments_label)

        hard_cut = []

        gra_cut = []



        for i in range(len(new_candidate_segments)):

            if new_candidate_segments_label[i] == 1:

                print "This is a gradual segment\n"

            else:

                hard_cut.append(self.detect_hard(new_candidate_segments[i], video_path, all_pixels))


        return hard_cut







    def process_invalid_frame(self, ret, frame, index, sign, i_video):

        temp_i = index + sign
        while ret is False:
            i_video.set(1, temp_i)
            ret, frame = i_video.read()
            temp_i += sign
        return frame

    def get_valid_frame(self, frames_path, index):

        suffix = 'jpg'

        frame = cv2.imread(os.sep.join([frames_path, '.'.join([str(index), suffix])]))

        return frame


    def get_candidate_segments_label(self, candidate_segments, hard_truth, gra_truth):

        candidate_segments_label = []

        for i in range(len(candidate_segments)):

            for j in range(len(hard_truth)):

                if candidate_segments[i][0] > hard_truth[j][1] and j < len(hard_truth) - 1:
                    continue
                elif self.if_overlap(candidate_segments[i][0], candidate_segments[i][1], hard_truth[j][0],
                                     hard_truth[j][1]):
                    candidate_segments_label.append(2)
                    break
                elif candidate_segments[i][1] < hard_truth[j][0]:
                    candidate_segments_label.append(0)
                    break

                if j == len(hard_truth) - 1:
                    candidate_segments_label.append(0)

        for i in range(len(candidate_segments)):

            for j in range(len(gra_truth)):

                if candidate_segments[i][0] > gra_truth[j][1]:
                    continue
                elif self.if_overlap(candidate_segments[i][0], candidate_segments[i][1], gra_truth[j][0],
                                     gra_truth[j][1]):
                    candidate_segments_label[i] = 1
                    break
                elif candidate_segments[i][1] < gra_truth[j][0]:
                    break


        return candidate_segments_label

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

        for i in range(len(candidate_segments)):

            first_frame = cv2.imread(os.path.join(video_path, str(candidate_segments[i][0]/temporal_window + 1).zfill(6), str(candidate_segments[i][0]+1).zfill(6)+'.jpg'))

            last_frame = cv2.imread(os.path.join(video_path, str(candidate_segments[i][0]/temporal_window + 1).zfill(6), str(candidate_segments[i][0]+2*temporal_window).zfill(6)+'.jpg'))

            # print self.get_hist_manh_diff(first_frame, last_frame, first_frame.shape[1] * first_frame.shape[0]), '\n'

            # if 0.5*self.get_pixel_diff(first_frame, last_frame, first_frame.shape[1] * first_frame.shape[0]) + \
            #         0.5*self.get_hist_chi_squa_diff(first_frame, last_frame, first_frame.shape[1] * first_frame.shape[0]) < 100:

            if self.get_hist_chi_squa_diff(first_frame, last_frame, first_frame.shape[1] * first_frame.shape[0]) < 0.5:

            # if self.get_hist_manh_diff(first_frame, last_frame, first_frame.shape[1] * first_frame.shape[0]) < 1:

            # if self.get_hist_manh_diff(cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV), cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV), first_frame.shape[1] * first_frame.shape[0]) < 2:

                invalid_index.append(i)

        return self.remove_elements_from_list(invalid_index, candidate_segments)

    def remove_invalid_segments2(self, candidate_segments, video_path):

        temporal_window = 8

        invalid_index = []

        for i in range(len(candidate_segments)):

            first_frame = cv2.imread(os.path.join(video_path, str(candidate_segments[i][0]/temporal_window + 1).zfill(6), str(candidate_segments[i][0]+1).zfill(6)+'.jpg'))

            last_frame = cv2.imread(os.path.join(video_path, str(candidate_segments[i][0]/temporal_window + 1).zfill(6), str(candidate_segments[i][0]+2*temporal_window).zfill(6)+'.jpg'))

            # print self.get_hist_manh_diff(first_frame, last_frame, first_frame.shape[1] * first_frame.shape[0]), '\n'

            # if 0.5*self.get_pixel_diff(first_frame, last_frame, first_frame.shape[1] * first_frame.shape[0]) + \
            #         0.5*self.get_hist_chi_squa_diff(first_frame, last_frame, first_frame.shape[1] * first_frame.shape[0]) < 100:

            # if self.get_hist_chi_squa_diff(first_frame, last_frame, first_frame.shape[1] * first_frame.shape[0]) < 5:

            if self.get_hist_manh_diff(first_frame, last_frame, first_frame.shape[1] * first_frame.shape[0]) < 2:

            # if self.get_hist_manh_diff(cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV), cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV), first_frame.shape[1] * first_frame.shape[0]) < 2:

                invalid_index.append(i)

        return self.remove_elements_from_list(invalid_index, candidate_segments)






    def extract_features(self, VideoPath):

        suffix = '*.jpg'

        all_group = range(1,len(glob(os.sep.join([VideoPath, '*']))) + 1)

        temporal_window = 8

        tmp_folder = '/home/CNN2/SBD/tmp2'

        tmp_feature_folder = os.sep.join([tmp_folder, 'feature'])

        tmp_to_be_extracted_feature_list = os.sep.join([tmp_folder, 'to_be_extracted.list'])

        tmp_have_been_extracted_feature_list = os.sep.join([tmp_folder, 'output.list'])

        self.mkdir(tmp_folder)

        self.mkdir(tmp_feature_folder)

        all_group_list = [' '.join([os.path.join(VideoPath, str(i).zfill(6)), str((i-1)*temporal_window + 1).zfill(6), '0\n']) for i in all_group]

        all_group_output = [os.path.join(tmp_feature_folder, str((i-1)*temporal_window + 1).zfill(6))+'\n' for i in all_group]


        with open(tmp_to_be_extracted_feature_list, 'w') as f:

            f.writelines(all_group_list)

        with open(tmp_have_been_extracted_feature_list, 'w') as f:

            f.writelines(all_group_output)



        extract_image_features = '/home/newC3D/C3D/C3D-v1.1/build/tools/extract_image_features'

        model_file = 'feature_extract_img.prototxt'

        caffemodel = '/home/C3D/C3D-v1.1/latest_result/models/train_group1/train_group_1_iter_200000.caffemodel'

        gpu_id = '0'

        batch_size = '4'

        batch_num = str(int(math.ceil(float(len(all_group)) / int(batch_size))))

        feature1 = 'fc8-new'

        feature2 = 'prob'

        shell = ' '.join(['GLOG_logtostderr=1', extract_image_features, model_file, caffemodel, gpu_id, batch_size, batch_num, tmp_have_been_extracted_feature_list, feature1, feature2])

        os.system(shell)

    def get_candidate_segments(self):

        length = 15

        tmp_folder = '/home/CNN2/SBD/tmp2'

        tmp_have_been_extracted_feature_list = os.sep.join([tmp_folder, 'output.list'])

        with open(tmp_have_been_extracted_feature_list, 'r') as f:

            all_segments = f.readlines()

        all_segments = [i.strip() for i in all_segments]

        suffix = '.prob'

        hard_segments = []

        gra_segments = []

        prob_list = {}

        for i in all_segments:

            (s, prob) = read_binary_blob(i + suffix)

            if np.argmax(prob) == 1 and max(prob) > 0.8:

                # print prob,'\n'

                # if len(gra_segments) > 0 and (self.if_overlap(gra_segments[-1][0], gra_segments[-1][1],
                #                                               int(i.split(os.sep)[-1]), int(i.split(os.sep)[-1]) + length) or int(i.split(os.sep)[-1])-gra_segments[-1][1] == 1):
                #
                #     gra_segments[-1][1] = int(i.split(os.sep)[-1]) + length
                #
                # else:
                #
                #     gra_segments.append([int(i.split(os.sep)[-1]), int(i.split(os.sep)[-1]) + length])

                gra_segments.append([int(i.split(os.sep)[-1]), int(i.split(os.sep)[-1]) + length])

            elif np.argmax(prob) == 2 and max(prob) > 0.8:

                # print i, max(prob),'\n'
                #
                if len(hard_segments) > 0 and self.if_overlap(hard_segments[-1][0], hard_segments[-1][1], int(i.split(os.sep)[-1]), int(i.split(os.sep)[-1]) + length):

                    hard_segments[-1][0] = int(i.split(os.sep)[-1])

                else:

                    hard_segments.append([int(i.split(os.sep)[-1]), int(i.split(os.sep)[-1]) + length])

        hard_segments = [[i[0]-1, i[1]-1] for i in hard_segments]
        gra_segments = [[i[0]-1, i[1]-1] for i in gra_segments]

        return [hard_segments, gra_segments]


    def recall_pre_f1(self, a, b, c):
        recall = float(a)/b if b!=0 else 0
        precison = float(a)/c if c!=0 else 0
        f1 = float(2*recall*precison) / (recall + precison)

        return precison, recall, f1



    def get_union_cut(self, set1, set2):

        cnt = 0
        for s1 in set1:
            for s2 in set2:
                if self.if_overlap(s1[0], s1[1], s2[0], s2[1]):
                    cnt += 1
                    break

        return cnt

    def eval(self, cut, cut_truth, gra, gra_truth):

        cut_correct = self.get_union_cut(cut, cut_truth)
        gra_correct = self.get_union_cut(gra, gra_truth)

        all_correct = self.get_union_cut(cut+gra, cut_truth+gra_truth)

        # return self.recall_pre_f1(gra_correct, len(gra), len(gra_truth)), self.recall_pre_f1(cut_correct, len(cut), len(cut_truth)), self.recall_pre_f1(all_correct, len(cut_truth+gra_truth),len(cut+gra))
        return cut_correct, gra_correct, all_correct

    def get_labels_TRECViD(self, label_path):

        hard_truth = []
        gra_truth = []

        with open(label_path) as f:

            xmlfile = f.readlines()

        for i in range(len(xmlfile)):
            if 'CUT' in xmlfile[i]:
                hard_truth.append([int(xmlfile[i].split('"')[-4]), int(xmlfile[i].split('"')[-2])])
            elif 'DIS' in xmlfile[i] or 'OTH' in xmlfile[i] or 'FOI' in xmlfile[i]:
                gra_truth.append([int(xmlfile[i].split('"')[-4]), int(xmlfile[i].split('"')[-2])])

        return [hard_truth, gra_truth]



    def sbd_on_trecvid(self):

        current_dir = os.path.join(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1]))

        path_to_frames = '/home/DSBD_Test/segments'

        videos = glob(os.sep.join([path_to_frames, '*']))

        labels_path = '/home/t2007ref'

        cut_correct = 0

        gra_correct = 0

        all_correct = 0

        cut_t = 0

        gra_t = 0

        all_t = 0

        cut_n = 0

        gra_n = 0

        all_n = 0

        for i in videos:

            # if cmp(i.split(os.sep)[-1], 'BG_34901') != 0:
            #     continue

            print 'Now', i.split(os.sep)[-1], ' is analyasing...'

            begin_time = time.time()

            self.extract_features(i)

            [hard_segments, gra_segments] = self.get_candidate_segments()

            hard_segments = self.remove_invalid_segments(hard_segments, i)

            gra_segments = self.remove_invalid_segments2(gra_segments, i)

            [hard_truth, gra_truth] = self.get_labels_TRECViD(os.sep.join([labels_path, 'ref_' + i.split(os.sep)[-1] + '.xml']))

            new_gra_segments = [deepcopy(gra_segments[0])]

            for gra in gra_segments[1:]:

                if self.if_overlap(new_gra_segments[-1][0], new_gra_segments[-1][1], gra[0], gra[1]):

                    new_gra_segments[-1][1] = gra[1]

                else:

                    new_gra_segments.append(deepcopy(gra))

            new_hard_segments = [deepcopy(hard_segments[0])]

            for hard in hard_segments[1:]:

                if self.if_overlap(new_hard_segments[-1][0], new_hard_segments[-1][1], hard[0], hard[1]):

                    new_hard_segments[-1][1] = hard[1]

                else:

                    new_hard_segments.append(deepcopy(hard))



            # self.eval(new_hard_segments, hard_truth)
            #
            # self.eval(new_gra_segments, gra_truth)

            # print self.eval(new_hard_segments, hard_truth, new_gra_segments, gra_truth)

            cut_t += len(hard_truth)
            gra_t += len(gra_truth)
            all_t += len(hard_truth) + len(gra_truth)

            cut_correct_n, gra_correct_n, all_correct_n = self.eval(new_hard_segments, hard_truth, new_gra_segments, gra_truth)

            cut_correct += cut_correct_n
            gra_correct += gra_correct_n
            all_correct += all_correct_n

            cut_n += len(new_hard_segments)
            gra_n += len(new_gra_segments)
            all_n += len(new_gra_segments+new_hard_segments)

            result = [str(i)+'\n',str(len(hard_truth))+' ' + str(cut_correct_n) + ' '+str(len(new_hard_segments)-cut_correct_n)+' '+str(len(hard_truth) - cut_correct_n)+'\n',
                      str(len(gra_truth))+' ' + str(gra_correct_n) + ' '+str(len(new_gra_segments)-gra_correct_n)+' '+str(len(gra_truth) - gra_correct_n)+'\n',
                      str(self.recall_pre_f1(gra_correct_n, len(gra_truth), len(new_gra_segments)))+'\n',
                      str(self.recall_pre_f1(cut_correct_n, len(hard_truth), len(new_hard_segments)))+'\n',
                      str(self.recall_pre_f1(all_correct_n, len(new_hard_segments+new_gra_segments),len(hard_truth+gra_truth)))+'\n']
            with open ('/home/TRECVID_Test_log.log', 'a') as f:
                f.writelines(result)

            end_time = time.time()

            print 'the cost of time is ', str(end_time - begin_time), '\n'

        with open('/home/TRECVID_Test_log.log', 'a') as f:
            f.writelines(['all_results:\n',str(self.recall_pre_f1(gra_correct, gra_t, gra_n))+'\n',str(self.recall_pre_f1(cut_correct, cut_t, cut_n))+'\n', str(self.recall_pre_f1(all_correct, all_t, all_n))])

if  __name__ == '__main__':

    test1 = SBD()
    test1.sbd_on_trecvid()
# test1.get_candidate_segments()