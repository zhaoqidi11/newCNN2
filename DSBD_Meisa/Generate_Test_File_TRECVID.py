# Generate the test dataset of the DeepSBD


if __name__ == '__main__':
    AllTest = []
    import os
    import shutil

    tv2007prefix = '/home/t2007/t2007/'
    OUTPUTtv2007prefix = '/home/DSBD_Test/segments/'

    with open('/home/new_test_list.txt') as f:
        AllTest = f.readlines()

    for i in AllTest:
        os.makedirs(OUTPUTtv2007prefix + i.split('/')[-3] + '/' + i.split('/')[-2])
        for j in range(16):
            shutil.copyfile(
                tv2007prefix + i.split('/')[-3] + '/' + i.split('/')[-3] + '_' + str(int(i.split(' ')[-2]) - 1 + j).zfill(5) + '.jpeg',
                OUTPUTtv2007prefix + i.split('/')[-3] + '/' + i.split('/')[-2] + '/' + str(int(i.split(' ')[-2]) + j).zfill(6) + '.jpg'
            )