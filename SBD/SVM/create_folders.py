if __name__ == '__main__':
    import os

    with open('/home/CNN2/SBD/tmp_output_list.list') as f:

        tmp_output_list = f.readlines()

    for i in tmp_output_list:

        os.mkdir(os.sep.join(i.split(os.sep)[:-1]))