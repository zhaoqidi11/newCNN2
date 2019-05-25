from sklearn import svm
import numpy as np

# notice this "import"!
# the first "read_binary_blob" is the function "read_binary_blob"
# the second "read_binary_blob" is "read_binary_blob.py"
from read_binary_blob import read_binary_blob

# To save SVM model
from sklearn.externals import joblib

if __name__ == '__main__':

    with open('../Test_DSBD/train.prefix') as f:

        all_train = f.readlines()

    with open('../Test_DSBD/new_train_list.txt') as f:

        all_train_labels = f.readlines()

    prefix = 'fc8-new'

    # prefix = 'pool5'
    # prefix = 'prob'
    # prefix = 'res5b'

    (s1, train_x) = read_binary_blob('.'.join([all_train[0].strip(), prefix]))
    train_x_labels = np.array([int(all_train_labels[0].strip().split(' ')[-1])])

    vector_length = train_x.shape[0]

    train_x = train_x.reshape(1, vector_length)

    for i in range(1, len(all_train)):

        train_x = np.concatenate((train_x, read_binary_blob('.'.join([all_train[i].strip(), prefix]))[1].reshape(1, vector_length)), axis = 0)

        train_x_labels = np.append(train_x_labels, np.array([int(all_train_labels[i].strip().split(' ')[-1])]))

    clf = svm.SVC(C=0.8, kernel='linear', decision_function_shape='ovr')

    clf.fit(train_x, train_x_labels)

    # The path to save
    path = '/home/train_model.pkl'

    joblib.dump(clf, path)

