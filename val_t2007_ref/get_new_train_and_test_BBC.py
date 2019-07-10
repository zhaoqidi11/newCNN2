import sys

sys.path.append('/home/opencv-3.4.3/build/lib')

import os

from glob2 import glob

import random

import cv2


def if_overlap_hard(begin1, end1, begin2, end2):

    if begin1 > begin2:
        begin1, end1, begin2, end2 = begin2, end2, begin1, end1

    return end1 > begin2


def if_overlap_segment(begin1, end1, begin2, end2):
    if begin1 > begin2:
        begin1, end1, begin2, end2 = begin2, end2, begin1, end1

    return end1 - begin2 >= 2


def get_labels_rai(label_path):
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
        if i[1] - i[0] == 1:
            hard_truth.append(i)
        else:
            gra_truth.append(i)

    return hard_truth, gra_truth

def get_all_ref(folder_path):

    all_videos = glob(os.path.join(folder_path, '*.mp4'))

    all_hard_truth_num = 0

    all_gra_truth_num = 0

    all_hard_segments_num = 0

    all_gra_segments_num = 0

    all_normal_segments_num = 0

    all_write_segments_num = 0

    annotations_folder_path = '/home/BBC/annotations/'

    for i in all_videos:

        hard_truth, gra_truth = get_labels_rai(annotations_folder_path + i.split(os.sep)[-1].split('.')[0] + '.txt')

        all_hard_truth_num += len(hard_truth)

        all_gra_truth_num += len(gra_truth)

        i_video = cv2.VideoCapture(i)

        frame_num = i_video.get(7)

        normal_segments = []

        hard_segments = []

        gra_segments = []

        end = 15

        hard_truth_index = 0

        gra_truth_index = 0

        flag_hard = False

        flag_gra = False

        while end < frame_num:

            if hard_truth_index < len(hard_truth) and if_overlap_hard(end - 15, end, hard_truth[hard_truth_index][0],
                                                                      hard_truth[hard_truth_index][1]):

                hard_segments.append([end - 15, end])

                flag_hard = True

            elif gra_truth_index < len(gra_truth) and if_overlap_segment(end - 15, end, gra_truth[gra_truth_index][0],
                                                                         gra_truth[gra_truth_index][1]):

                gra_segments.append([end - 15, end])

                flag_gra = True

            else:

                normal_segments.append([end - 15, end])

            end += 8

            if hard_truth_index < len(hard_truth) and flag_hard == True and end - 15 >= hard_truth[hard_truth_index][1]:

                flag_hard = False

                hard_truth_index += 1

            if gra_truth_index < len(gra_truth) and flag_gra == True and end - 15 - gra_truth[gra_truth_index][1] > -5:

                flag_gra = False

                gra_truth_index += 1

        write_hard = [str(line_hard[0])+'\t'+str(line_hard[1])+'\t2\n' for line_hard in hard_segments if random.random() < 0.2]

        write_gra = [str(line_gra[0])+'\t'+str(line_gra[1])+'\t1\n' for line_gra in gra_segments]

        write_normal = [str(line_normal[0])+'\t'+str(line_normal[1])+'\t0\n' for line_normal in normal_segments if random.random() < 0.03]

        all_write_segments_num += len(write_normal)

        write = []

        write.extend(write_hard)
        write.extend(write_gra)
        write.extend(write_normal)

        with open('/home/BBC/ref'+ i.split(os.sep)[-1].split('.')[0] + '.txt', 'w') as f:

            f.writelines(write)

        print 'a'


        all_hard_segments_num += len(hard_segments)

        all_gra_segments_num += len(gra_segments)

        all_normal_segments_num += len(normal_segments)

    print 'a'


    # for i in all_videos:
    #
    #     if cmp(i.split(os.sep)[-1], 'bbc_1') == 0:
    #
    #         continue
    #
    #     all_xml_in_this_folder = glob(os.path.join(i, '*.xml'))
    #
    #     for xml in all_xml_in_this_folder:
    #
    #         with open(xml) as f:
    #
    #             lines = f.readlines()
    #
    #         frame_num = int(lines[1].split('"')[-2])
    #
    #         hard_truth, gra_truth = get_labels_TRECViD(xml)
    #
    #         all_hard_truth_num += len(hard_truth)
    #
    #         all_gra_truth_num += len(gra_truth)
    #
    #         normal_segments = []
    #
    #         hard_segments = []
    #
    #         gra_segments = []
    #
    #         end = 15
    #
    #         hard_truth_index = 0
    #
    #         gra_truth_index = 0
    #
    #         flag_hard = False
    #
    #         flag_gra = False
    #
    #
    #         while end < frame_num:
    #
    #             if hard_truth_index < len(hard_truth) and if_overlap_hard(end-15, end, hard_truth[hard_truth_index][0], hard_truth[hard_truth_index][1]):
    #
    #                 hard_segments.append([end-15, end])
    #
    #                 flag_hard = True
    #
    #             elif gra_truth_index < len(gra_truth) and if_overlap_segment(end-15, end, gra_truth[gra_truth_index][0], gra_truth[gra_truth_index][1]):
    #
    #                 gra_segments.append([end-15, end])
    #
    #                 flag_gra = True
    #
    #             else:
    #
    #                 normal_segments.append([end-15,end])
    #
    #             end += 8
    #
    #             if hard_truth_index < len(hard_truth) and flag_hard == True and end - 15 >= hard_truth[hard_truth_index][1]:
    #
    #                 flag_hard = False
    #
    #                 hard_truth_index += 1
    #
    #
    #             if gra_truth_index < len(gra_truth) and flag_gra == True and end - 15 - gra_truth[gra_truth_index][1] > -5:
    #
    #                 flag_gra = False
    #
    #                 gra_truth_index += 1
    #
    #         write_hard = [str(line_hard[0])+'\t'+str(line_hard[1])+'\t2\n' for line_hard in hard_segments]
    #
    #         write_gra = [str(line_gra[0])+'\t'+str(line_gra[1])+'\t1\n' for line_gra in gra_segments]
    #
    #         write_normal = [str(line_normal[0])+'\t'+str(line_normal[1])+'\t0\n' for line_normal in normal_segments if random.random() < 0.05]
    #
    #         all_write_segments_num += len(write_normal)
    #
    #         write = []
    #
    #         write.extend(write_hard)
    #         write.extend(write_gra)
    #         write.extend(write_normal)
    #
    #         with open('/home/C3D/ref2_test/'+ xml.split('.')[0].split(os.sep)[-1] + '.txt', 'w') as f:
    #
    #             f.writelines(write)
    #
    #
    #
    #
    #         all_hard_segments_num += len(hard_segments)
    #
    #         all_gra_segments_num += len(gra_segments)
    #
    #         all_normal_segments_num += len(normal_segments)
    #
    #
    #         print 'a'
    # print'a'






if __name__ == '__main__':

    get_all_ref('/home/BBC')

