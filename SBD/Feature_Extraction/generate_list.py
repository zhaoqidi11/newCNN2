# Generate the output list to be used in C3D

if __name__ == '__main__':

    with open('../Test_DSBD/new_test_list.txt') as f:
        train_list = f.readlines()

    generated_list = []

    for i in train_list:
        generated_list.append(i.strip().split(' ')[0] + i.strip().split(' ')[1].zfill(6) + '\n')

    with open('../Test_DSBD/test.prefix', 'w') as f:
        f.writelines(generated_list)