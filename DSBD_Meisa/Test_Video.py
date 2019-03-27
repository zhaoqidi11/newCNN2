# Test the correctness of the index obtained by OpenCV
if __name__ == '__main__':
    from glob import glob
    import sys
    sys.path.append('/usr/local/lib/python2.7/site-packages/')
    import cv2
    import os
    os.chdir('/media/user02/Volume/TRECVID2007VIDEO/')
    AllMpgFiles = glob('*.mpg')

    for i in AllMpgFiles:
        i_Video = cv2.VideoCapture(i)
        print int(i_Video.get(7))
        print 'a'
