import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


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

    iter = 5
    for i in range(5):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)
        clf = LogisticRegression(random_state=0, solver='liblinear', class_weight='balanced', penalty='l1', verbose=1,
                                 warm_start=True).fit(x_train, y_train)
    pre_y = clf.predict(test_x)
    pre_y = pre_y.astype(int)
    result = zip(no, pre_y)
    file_out = open("submission.csv", 'w')
    file_out.write('id,target\n')
    for item in result:
        file_out.write(str(item[0]) + ',' + str(item[1]) + "\n")
    file_out.close()