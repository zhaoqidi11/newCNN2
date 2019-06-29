if __name__ == '__main__':

    with open('/home/fixed_test.txt') as f:

        all_lines = f.readlines()


    type_0_num = 0
    type_1_num = 1
    type_2_num = 2

    for i in all_lines:

        type_ = int(i.strip().split(' ')[-1])

        if type_ == 0:

            type_0_num += 1

        elif type_ == 1:

            type_1_num += 1

        elif type_ == 2:

            type_2_num += 1

    print 'a'

