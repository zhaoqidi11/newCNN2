import sys
sys.path.append('/home/opencv-3.4.3/build/lib')

import cv2

def get_candidate_segments2(video_path):

    i_video = cv2.VideoCapture(video_path)

    number_of_frames_in_video = int(i_video.get(7))

    temporal_window = 8

    candidate_segments = []

    group_number = 1 + (number_of_frames_in_video - 2 * temporal_window) / temporal_window

    for i in range(group_number):
        candidate_segments.append([i * temporal_window, i * temporal_window + 2 * temporal_window - 1])

    return candidate_segments

def get_rai_files():

    get_candidate_segments2('/home/CNN2/SBD/videos/1.mp4')