# Generate the test dataset of the DeepSBD


if __name__ == '__main__':
    AllTest = []
    import os
    import shutil

    tv2007prefix = '/media/user02/Volume/TRECVid2007/'
    OUTPUTtv2007prefix = '/media/user02/Volume/DSBD_Test/segments/'

    with open('/media/user02/Volume/DSBD/data/test_list.txt') as f:
        AllTest = f.readlines()

    for i in AllTest:
        if cmp(i.split('/')[5], 'tv2007') == 0:
            os.makedirs(OUTPUTtv2007prefix + i.split('/')[-3] + '/' + i.split('/')[-2])
            for j in range(16):
                shutil.copyfile(
                    tv2007prefix + i.split('/')[-3] + '/' + i.split('/')[-3] + '_' + str(int(i.split(' ')[-2]) - 1 + j).zfill(5) + '.jpeg',
                    OUTPUTtv2007prefix + i.split('/')[-3] + '/' + i.split('/')[-2] + '/' + str(int(i.split(' ')[-2]) + j).zfill(6) + '.jpg'
                )