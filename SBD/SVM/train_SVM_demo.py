from sklearn import svm
import numpy as np
if __name__ == '__main__':

    x = np.array([[2,3], [4,5]])

    x_test = np.array([[1,1], [2,2]])

    y = np.array([1,0])

    clf = svm.SVC(kernel='linear')

    clf.fit(x, y)

    result = clf.predict(x_test)

    print result
