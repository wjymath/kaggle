import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE

def load_train_data():
    data = np.genfromtxt("train.csv", delimiter=',', skip_header=1)
    return data


def load_test_data():
    data = np.genfromtxt("test.csv", delimiter=',', skip_header=1)
    return data


if __name__ == "__main__":
    data = load_train_data()
    test_data = load_test_data()
    x = data[:, 2:]
    y = data[:, 1]
    print(x.shape)
    print(y.shape)


    test_x = test_data[:, 1:]
    no = test_data[:, 0]
    no = no.astype(int)

    for i in range(10):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)
        func = LogisticRegression(random_state=0, C=2, solver='liblinear', class_weight='balanced', penalty='l1', warm_start=False)
        # func = SGDClassifier(random_state=0, loss='log', class_weight='balanced', penalty='l1', l1_ratio=0.5, validation_fraction=0.5,
        #                           warm_start=False, max_iter=1000, tol=1e-4)
        # func = svm.NuSVC(kernel='rbf', gamma='scale')
        # func = svm.LinearSVC(loss='hinge', class_weight='balanced', max_iter=10000)
        clf = RFE(func, n_features_to_select=100)
        clf.fit(x_train, y_train)
        print(str(clf.score(x_test, y_test)))
        if clf.score(x_test, y_test) > 0.85:
            break
    print(str(clf.score(x, y)))
    pre_y = clf.predict(test_x)
    pre_y = clf.predict_proba(test_x)[:,1]

    result = zip(no, pre_y)
    file_out = open("submission.csv", 'w')
    file_out.write('id,target\n')
    for item in result:
        file_out.write(str(item[0]) + ',' + str(item[1]) + "\n")
    file_out.close()