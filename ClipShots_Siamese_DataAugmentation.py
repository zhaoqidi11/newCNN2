class ClipShots_Siamese_DataAugmentation():
    from glob import glob
    import cv2
    import os

    Path = '/media/user02/New Volume/newImage/newImage/'
    os.chdir(Path)
    AllImages = glob('*.jpg')

    Img = []

    DesPath = '/media/user02/New Volume/newImage/newImage_V/'
    for i in range(len(AllImages)):
        cv2.imwrite(DesPath + AllImages[i].split('.jpg')[0] + '_V.jpg', cv2.flip(cv2.imread(AllImages[i]), 0))