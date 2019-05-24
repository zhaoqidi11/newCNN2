import os

if __name__ == '__main__':

    output_prefix_path = '/home/CNN2/SBD/Test_DSBD/output.prefix'

    with open(output_prefix_path) as f:

        all_lines = f.readlines()

    for i in all_lines:

        if not os.path.exists(os.sep.join(i.strip().split(os.sep)[:-1])):

            os.makedirs(os.sep.join(i.strip().split(os.sep)[:-1]))