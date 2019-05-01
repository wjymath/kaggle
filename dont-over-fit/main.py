import numpy as np
from sklearn.linear_model import LogisticRegression


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

    test_x = test_data[:, 1:]
    no = test_data[:, 0]
    no = no.astype(int)
    print(no[:3])
    # 要求强正则性
    clf = LogisticRegression(random_state=10, C=0.25, solver='liblinear', class_weight='balanced', penalty='l1', verbose=1).fit(x, y)
    pre_y = clf.predict_proba(test_x)[:,1]
    # pre_y = pre_y.astype(int)
    result = zip(no, pre_y)
    file_out = open("submission.csv", 'w')
    file_out.write('id,target\n')
    for item in result:
        file_out.write(str(item[0]) + ',' + str(item[1]) + "\n")
    file_out.close()