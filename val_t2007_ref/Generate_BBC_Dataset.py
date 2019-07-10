import sys
sys.path.append('/home/opencv-3.4.3/build/lib')
import cv2
from glob2 import glob
import os


def process_invalid_frame(ret, frame, index, sign, i_video):
    temp_i = index + sign
    while ret is False:
        i_video.set(1, temp_i)
        ret, frame = i_video.read()
        temp_i += sign
    return frame


def get_valid_frame(i_video, index, sign):
    i_video.set(1, index)
    ret, frame = i_video.read()
    if ret:
        return frame
    else:
        return process_invalid_frame(ret, frame, index, sign, i_video)

generate_BBC_dir= '/home/BBC/BBC_dataset'

all_videos_path = '/home/BBC'

all_videos = glob(os.path.join(all_videos_path, '*.mp4'))

txt_line = []

for video in all_videos:

    # sub_dir = os.path.join(generate_BBC_dir, video.split(os.sep)[-1].split('.')[0])
    #
    # os.mkdir(sub_dir)
    #
    # i_video = cv2.VideoCapture(video)
    #
    with open(all_videos_path+'/ref'+video.split(os.sep)[-1].split('.')[0]+'.txt') as f:

        all_lines = f.readlines()

    for i in all_lines:
    #
    #     sub_sub_dir = os.path.join(sub_dir, str(int(i.strip().split('\t')[0])+1).zfill(6))
    #
    #     os.mkdir(sub_sub_dir)
    #
    #     for j in range(int(i.strip().split('\t')[0]), int(i.strip().split('\t')[1])):
    #
    #         frame = get_valid_frame(i_video, j, 1)
    #
    #         cv2.imwrite(os.path.join(sub_sub_dir, str(j+1).zfill(6)+'.jpg'), frame)
    #
    #     frame = get_valid_frame(i_video, int(i.strip().split('\t')[1]), -1)
    #
    #     cv2.imwrite(os.path.join(sub_sub_dir, str(int(i.strip().split('\t')[1])+1).zfill(6)+'.jpg'), frame)

        txt_line.append(os.path.join(os.path.join(generate_BBC_dir, video.split(os.sep)[-1].split('.')[0]), \
                                                  str(int(i.strip().split('\t')[0])+1).zfill(6))+' '+ \
                                     str(int(i.strip().split('\t')[0])+1).zfill(6)+' '+i.strip().split('\t')[-1]+'\n')


with open('/home/BBC/ref.txt', 'w') as f:

    f.writelines(txt_line)

print 'a'