import os
from copy import deepcopy

def if_overlap(begin1, end1, begin2, end2):
    if begin1 > begin2:
        begin1, end1, begin2, end2 = begin2, end2, begin1, end1

    return end1 > begin2

def get_labels_TRECViD(label_path):

    with open(label_path) as f:

        xmlfile = f.readlines()

    truth = []

    for i in range(len(xmlfile))[2:len(xmlfile)-1]:

            truth.append([int(xmlfile[i].split('"')[-4]), int(xmlfile[i].split('"')[-2])])

    return truth

if __name__ == '__main__':

    id = 38150

    with open('/home/C3D/C3D-v1.1/examples/c3d_ucf101_finetuning/new_test_list.txt') as f:

        lines = f.readlines()

    label_path = '/home/t2007ref/ref_BG_' + str(id) +'.xml'

    truth = get_labels_TRECViD(label_path)

    segments_begin = {}

    extra_segments = []

    prefix =lines[0].strip().split(os.sep)[:4]

    for i in lines:

        if cmp(i.strip().split(' ')[0].split(os.sep)[4], 'BG_' + str(id)) == 0:

            segments_begin[int(i.strip().split(' ')[-2]) - 1] = int(i.strip().split(' ')[-1])

        else:

            extra_segments.append(i)


    segments_begin = sorted(segments_begin.items(), key=lambda item:item[0])

    all_segments = []

    for i in range(1, len(segments_begin)+1):

        prefix_back = deepcopy(prefix)

        prefix_back.extend(['BG_'+str(id), str(i).zfill(6), ''])

        all_segments.append(os.sep.join(prefix_back) + ' '+ str(segments_begin[i-1][0]+1)+' '+ str(segments_begin[i-1][1]) + '\n')

    # with open('/home/BG_'+str(id)+'.txt', 'w') as f:
    #     f.writelines(all_segments)
    #
    # with open('/home/extra.txt', 'w') as f:
    #     f.writelines(extra_segments)


    print 'a'

    index = 0

    for i in range(len(segments_begin)):

        for j in range(len(truth)):

            if if_overlap(segments_begin[i][0], segments_begin[i][0]+15, truth[j][0], truth[j][1]):

                if truth[j][1]-truth[j][0] == 1:

                    if segments_begin[i][1] == 2:

                        break

                    else:

                        print segments_begin[i], '\n'

                        all_segments[i] = ' '.join(all_segments[i].strip().split(' ')[:-1]) + ' 2\n'

                        break

                elif segments_begin[i][1] == 1:

                    break

                else:

                    all_segments[i] = ' '.join(all_segments[i].strip().split(' ')[:-1]) + ' 1\n'

                    print segments_begin[i], '\n'

                    break

            if j == len(truth) - 1:

                if segments_begin[i][1] != 0:

                    all_segments[i] = ' '.join(all_segments[i].strip().split(' ')[:-1]) + ' 0\n'

                    print segments_begin[i]

    with open('/home/fixed_BG_'+str(id)+'.txt', 'w') as f:
        f.writelines(all_segments)

    print 'a'