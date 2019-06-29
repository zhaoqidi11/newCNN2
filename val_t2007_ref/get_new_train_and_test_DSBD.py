import os

from glob2 import glob

import random

def if_overlap_hard(begin1, end1, begin2, end2):

    if begin1 > begin2:
        begin1, end1, begin2, end2 = begin2, end2, begin1, end1

    return end1 > begin2


def if_overlap_segment(begin1, end1, begin2, end2):
    if begin1 > begin2:
        begin1, end1, begin2, end2 = begin2, end2, begin1, end1

    return end1 - begin2 >= 5


def get_labels_TRECViD(label_path):
    hard_truth = []
    gra_truth = []

    with open(label_path) as f:

        xmlfile = f.readlines()

    for i in range(len(xmlfile)):
        if 'CUT' in xmlfile[i]:
            hard_truth.append([int(xmlfile[i].split('"')[-4]), int(xmlfile[i].split('"')[-2])])
        elif 'DIS' in xmlfile[i] or 'OTH' in xmlfile[i] or 'FOI' in xmlfile[i]:
            gra_truth.append([int(xmlfile[i].split('"')[-4]), int(xmlfile[i].split('"')[-2])])

    return hard_truth, gra_truth

def get_all_ref(folder_path):

    all_ref_folder = glob(os.path.join(folder_path, '*'))

    all_hard_truth_num = 0

    all_gra_truth_num = 0

    all_hard_segments_num = 0

    all_gra_segments_num = 0

    all_normal_segments_num = 0

    all_write_segments_num = 0

    for i in all_ref_folder:

        if cmp(i.split(os.sep)[-1], '2005') == 0:

            continue

        all_xml_in_this_folder = glob(os.path.join(i, '*.xml'))

        for xml in all_xml_in_this_folder:

            with open(xml) as f:

                lines = f.readlines()

            frame_num = int(lines[1].split('"')[-2])

            hard_truth, gra_truth = get_labels_TRECViD(xml)

            all_hard_truth_num += len(hard_truth)

            all_gra_truth_num += len(gra_truth)

            normal_segments = []

            hard_segments = []

            gra_segments = []

            end = 15

            hard_truth_index = 0

            gra_truth_index = 0

            flag_hard = False

            flag_gra = False


            while end < frame_num:

                if hard_truth_index < len(hard_truth) and if_overlap_hard(end-15, end, hard_truth[hard_truth_index][0], hard_truth[hard_truth_index][1]):

                    hard_segments.append([end-15, end])

                    flag_hard = True

                elif gra_truth_index < len(gra_truth) and if_overlap_segment(end-15, end, gra_truth[gra_truth_index][0], gra_truth[gra_truth_index][1]):

                    gra_segments.append([end-15, end])

                    flag_gra = True

                else:

                    normal_segments.append([end-15,end])

                end += 8

                if hard_truth_index < len(hard_truth) and flag_hard == True and end - 15 >= hard_truth[hard_truth_index][1]:

                    flag_hard = False

                    hard_truth_index += 1


                if gra_truth_index < len(gra_truth) and flag_gra == True and end - 15 - gra_truth[gra_truth_index][1] > -5:

                    flag_gra = False

                    gra_truth_index += 1

            write_hard = [str(line_hard[0])+'\t'+str(line_hard[1])+'\t2\n' for line_hard in hard_segments]

            write_gra = [str(line_gra[0])+'\t'+str(line_gra[1])+'\t1\n' for line_gra in gra_segments]

            write_normal = [str(line_normal[0])+'\t'+str(line_normal[1])+'\t0\n' for line_normal in normal_segments if random.random() < 0.05]

            all_write_segments_num += len(write_normal)

            write = []

            write.extend(write_hard)
            write.extend(write_gra)
            write.extend(write_normal)

            with open('/home/C3D/ref2_test/'+ xml.split('.')[0].split(os.sep)[-1] + '.txt', 'w') as f:

                f.writelines(write)




            all_hard_segments_num += len(hard_segments)

            all_gra_segments_num += len(gra_segments)

            all_normal_segments_num += len(normal_segments)


            print 'a'
    print'a'






if __name__ == '__main__':

    get_all_ref('/home/C3D/ref')

