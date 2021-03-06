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

class SBD():

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

        i_video = cv2.VideoCapture(video_path)

        for i in C3D_segments:

            index = i[0]

            save_path = os.sep.join([tmp_folder_path, str(index+1).zfill(6)])

            os.mkdir(save_path)

            while index < i[1]:

                cv2.imwrite(os.sep.join([save_path, '.'.join([str(index+1).zfill(6), 'jpg'])]), self.get_valid_frame(i_video, index, 1))

                index += 1

            cv2.imwrite(os.sep.join([save_path, '.'.join([str(i[1]+1).zfill(6), 'jpg'])]), self.get_valid_frame(i_video, i[1], -1))



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

        i_Video = cv2.VideoCapture(video_path)

        frame_first = self.get_valid_frame(i_Video, candidate_segment[0], 1)


        for i in range(candidate_segment[0]+1,candidate_segment[1]):

            frame_next = self.get_valid_frame(i_Video, i, -1)

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

    def get_valid_frame(self, i_video, index, sign):
        i_video.set(1, index)
        ret, frame = i_video.read()
        if ret:
            return frame
        else:
            return self.process_invalid_frame(ret, frame, index, sign, i_video)


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



    # Get candidate segments
    def get_candidate_segments(self, VideoPath):

        group_length = 16
        second_group_length = 8

        i_video = cv2.VideoCapture(VideoPath)

        # get width of this video
        wid = int(i_video.get(3))
        # get height of this video
        hei = int(i_video.get(4))
        # It save the number of frames in this video
        number_of_frames_in_video = int(i_video.get(7))

        # get the frame no. of one frame in this video (be used to normalize)
        all_pixels = wid * hei

        group_number = (number_of_frames_in_video - group_length) / (group_length - 1) + 1


        diff_group_10 = []

        diff_group_5 = []


        # the diff between the 0 th and 10 th([0,10], [10,20], [20,30], ...) is larger than threshold
        candidate_segments_10 = []
        # the diff between the 5 th and 15 th([5,15], [15,25], [25,35], ...) is larger than threshold
        candidate_segments_5 = []


        group_first_frame = self.get_valid_frame(i_video, group_length - 1, -1)

        d = self.get_hist_chi_squa_diff(self.get_valid_frame(i_video, 0, 1), group_first_frame, all_pixels)

        if d > 0.5:

            candidate_segments_10.append([0, group_length - 1])

            diff_group_10.append(d)



        for i in range(1, group_number):

            group_last_frame = self.get_valid_frame(i_video, group_length-1 + i*(group_length-1), -1)

            d = self.get_hist_chi_squa_diff(group_last_frame, group_first_frame, all_pixels)

            if d > 0.5:

                candidate_segments_10.append([group_length-1 + (i-1)*(group_length-1), group_length-1 + i*(group_length-1)])

                diff_group_10.append(d)

            group_first_frame = copy.deepcopy(group_last_frame)


        group_5_15_number = (number_of_frames_in_video - second_group_length - group_length) / (group_length) + 1




        first_frame_5_to_15 = self.get_valid_frame(i_video, second_group_length + group_length - 1, 1)

        d = self.get_hist_chi_squa_diff(first_frame_5_to_15, self.get_valid_frame(i_video, second_group_length, 1), all_pixels)

        if d > 0.5:

            candidate_segments_5.append([second_group_length, second_group_length + group_length - 1])

            diff_group_5.append(d)


        for i in range(1, group_5_15_number):

            Frames10_5_2 = self.get_valid_frame(i_video, second_group_length + group_length-1 + i*(group_length-1), -1)

            d = self.get_hist_chi_squa_diff(first_frame_5_to_15, Frames10_5_2, all_pixels)

            if d > 0.5:

                candidate_segments_5.append([second_group_length + group_length-1 + (i-1)*(group_length-1),

                                           second_group_length + group_length-1 + i*(group_length-1)])
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

        if all_candidate_segments[-1][1] >= number_of_frames_in_video:

            all_candidate_segments[-1][0] = number_of_frames_in_video - (second_group_length + 1)

            all_candidate_segments[-1][1] = number_of_frames_in_video - 1

        return [all_candidate_segments, number_of_frames_in_video, all_pixels]

    def eval(self, cut, truth):

        count = 0

        for i in cut:

            for j in range(len(truth)):

                if self.if_overlap(i[0],i[1], truth[j][0], truth[j][1]):

                    count += 1
                    break

                if j == len(truth) - 1:

                    print i

        return count




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

        path_to_video = 'videos'

        videos = glob(os.sep.join([path_to_video, '*.mp4']))


        for i in videos:

            if cmp(i.split(os.sep)[-1], '1.mp4') != 0:

                continue

            print 'Now', i.split(os.sep)[-1], ' is analyasing...'

            begin_time = time.time()


            [all_candidate_segments, number_of_frames_in_video, all_pixels] = self.get_candidate_segments(i)



            # hard_cut = self.get_location(all_candidate_segments, self.get_candidate_segments_label(all_candidate_segments, hard_truth
                                                                                        # , gra_truth), i, all_pixels, hard_truth)
            # correct_number = self.eval(hard_cut, hard_truth)





            C3D_segments = self.get_candidate_segments_image(current_dir, i, all_candidate_segments, number_of_frames_in_video)


            self.check_candidate_segments3(all_candidate_segments, hard_truth, gra_truth)

            #
            # [missed_hard, missed_gra] = self.check_candidate_segments2(all_candidate_segments, hard_truth, gra_truth)
            #
            # print 'missed hard no. is ', str(missed_hard), 'missed gra no. is ', str(missed_gra), '\n'
            #
            # print 'hard Recall is', str(float(len(hard_truth) - missed_hard) / len(hard_truth)), 'gra Recall is', str(float(len(gra_truth) - missed_gra) / len(gra_truth)), '\n'

            end_time = time.time()

            print 'the cost of time is ', str(end_time - begin_time), '\n'



if  __name__ == '__main__':

    test1 = SBD()
    test1.sbd_on_rai()