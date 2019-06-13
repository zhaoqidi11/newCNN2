import os


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

    id = 2408

    with open('/home/C3D/C3D-v1.1/examples/c3d_ucf101_finetuning/new_test_list.txt') as f:

        lines = f.readlines()

    label_path = '/home/t2007ref/ref_BG_' + str(id) +'.xml'

    truth = get_labels_TRECViD(label_path)

    segments_begin = {}

    for i in lines:

        if cmp(i.strip().split(' ')[0].split(os.sep)[4], 'BG_' + str(id)) == 0:

            segments_begin[int(i.strip().split(' ')[-2]) - 1] = int(i.strip().split(' ')[-1])

    segments_begin = sorted(segments_begin.items(), key=lambda item:item[0])

    index = 0

    for i in segments_begin:

        for j in truth:

            if if_overlap(i[0], i[0]+15, j[0], j[1]):

                if j[1]-j[0] == 1:

                    if i[1] == 2:

                        continue
                elif i[1] == 1:

                    continue

                else:

                    print i, '\n'

    print 'a'