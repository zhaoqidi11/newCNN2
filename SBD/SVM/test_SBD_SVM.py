# To load SVM model
from sklearn.externals import joblib

# notice this "import"!
# the first "read_binary_blob" is the function "read_binary_blob"
# the second "read_binary_blob" is "read_binary_blob.py"
from read_binary_blob import read_binary_blob

import numpy as np
if __name__ == '__main__':

    # Load SVM model
    clf = joblib.load('train_model.pkl')

    # Load test dataset

    with open('/home/CNN2/SBD/tmp_output_list.list') as f:

        all_test = f.readlines()

    with open('/home/CNN2/SBD/label.txt') as f:

        all_test_labels = f.readlines()

    prefix = 'fc8-new'

    # prefix = 'pool5'
    # prefix = 'prob'
    # prefix = 'res5b'

    (s1, test_x) = read_binary_blob('.'.join([all_test[0].strip(), prefix]))
    test_x_labels = np.array([int(all_test_labels[0].strip())])

    vector_length = test_x.shape[0]

    test_x = test_x.reshape(1, vector_length)

    for i in range(1, len(all_test)):
        test_x = np.concatenate(
            (test_x, read_binary_blob('.'.join([all_test[i].strip(), prefix]))[1].reshape(1, vector_length)), axis=0)

        test_x_labels = np.append(test_x_labels, np.array([int(all_test_labels[i].strip())]))


    # y_hat = clf.predict(test_x)

    predict =  clf.predict(test_x)
    score = clf.score(test_x, test_x_labels)
    # 0.9156136674488349
    print 'a'