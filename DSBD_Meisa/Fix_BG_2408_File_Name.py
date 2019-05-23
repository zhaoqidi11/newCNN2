if __name__ == '__main__':
    import glob2
    import os

    os.chdir('/home/t2007/t2007/BG_2408/')
    for file in glob2.iglob('*.jpeg'):
        os.rename(file, 'BG_2408_' + file.split('.')[0].split('_')[-1].zfill(5) + '.jpeg')

    # os.chdir('/home/')
    #
    # os.rename('0111.txt', '.'.join(['0111.txt'.split('.')[0].zfill(5), 'txt']))