# fix the t2007 folder
from glob2 import glob

import os

if __name__ == '__main__':

    all_folders = glob('/home/t2007/*')

    for i in all_folders:

        all_jpeg = glob(os.sep.join([i, '*']))

        for j in all_jpeg:

            os.rename(j, os.sep.join([str(i), str(int(j.split(os.sep)[-1].split('_')[-1].split('.')[-2]))+ '.jpg']))
