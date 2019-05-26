# To load SVM model
# from sklearn.externals import joblib

# notice this "import"!
# the first "read_binary_blob" is the function "read_binary_blob"
# the second "read_binary_blob" is "read_binary_blob.py"
from read_binary_blob import read_binary_blob

import numpy as np
if __name__ == '__main__':

    # Load SVM model
    #clf = joblib.load('train_model.pkl')

    # Load test dataset

    with open('/home/CNN2/SBD/tmp_output_list.list') as f:

        all_test = f.readlines()

    with open('/home/CNN2/SBD/label.txt') as f:

        all_test_labels = f.readlines()

    # prefix = 'fc8-new'

    # prefix = 'pool5'
    prefix = 'prob'
    # prefix = 'res5b'
    predict_label = []

    for i in all_test:

        (s1, test_x_labels) = read_binary_blob('.'.join([i.strip(), prefix]))

        predict_label.append(np.argmax(test_x_labels))

    false_number = 0
    for i in range(len(predict_label)):

        if predict_label[i] != int(all_test_labels[i].strip()):

            false_number += 1

    print 'false number is ', int(false_number)

    print 'accuracy is ', float(len(all_test_labels) - false_number) / len(all_test_labels)

    # 0.9156136674488349
    print 'a'