import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split  # 数据集的分割函数
from sklearn.ensemble import GradientBoostingClassifier

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

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    sample_dict = {1: 0.36, 0: 0.64}

    clf = GradientBoostingClassifier(n_estimators=100, loss='exponential', max_depth=1, random_state=0, subsample=0.7).fit(x_train, y_train)
    clf.fit(x_train, y_train)
    print(str(100) + "\t" + str(0.7) + "\t" + str(clf.score(x_test, y_test)))
    pre_y = clf.predict(test_x)
    pre_y = pre_y.astype(int)
    result = zip(no, pre_y)
    file_out = open("submission.csv", 'w')
    file_out.write('id,target\n')
    for item in result:
        file_out.write(str(item[0]) + ',' + str(item[1]) + "\n")
    file_out.close()