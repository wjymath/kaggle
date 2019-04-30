import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split  # 数据集的分割函数
from sklearn import metrics


def load_train_data():
    data = np.genfromtxt("train.csv", delimiter=',', skip_header=1)
    return data


def load_test_data():
    data = np.genfromtxt("test.csv", delimiter=',', skip_header=1)
    return data


class HiddenLayer:
    def __init__(self, x, num):
        row = x.shape[0]
        columns = x.shape[1]
        rnd = np.random.RandomState(5)
        self.w = rnd.uniform(-1, 1, (columns, num))
        self.b = np.zeros([row, num], dtype=float)
        for i in range(num):
            rand_b = rnd.uniform(-0.5, 0.5)
        self.b = np.tile(rand_b, (row, 1))
        self.h = self.sigmoid(np.dot(x, self.w) + self.b)
        self.H_ = np.linalg.pinv(self.h)
        # print(self.H_.shape)

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def regressor_train(self, T):
        C = 2
        I = len(T)
        sub_former = np.dot(np.transpose(self.h), self.h) + I / C
        all_m = np.dot(np.linalg.pinv(sub_former), np.transpose(self.h))
        T = T.reshape(-1, 1)
        self.beta = np.dot(all_m, T)
        return self.beta

    def classifisor_train(self, T):
        en_one = OneHotEncoder()
        T = en_one.fit_transform(T.reshape(-1, 1)).toarray()  # 独热编码之后一定要用toarray()转换成正常的数组
        C = 3
        I = len(T)
#        print(np.dot(np.transpose(self.h), self.h))
        sub_former = np.dot(np.transpose(self.h), self.h) + I / C
        all_m = np.dot(np.linalg.pinv(sub_former), np.transpose(self.h))
        self.beta = np.dot(all_m, T)
        return self.beta

    def regressor_test(self, test_x):
        b_row = test_x.shape[0]
        h = self.sigmoid(np.dot(test_x, self.w) + self.b[:b_row, :])
        result = np.dot(h, self.beta)
        return result

    def classifisor_test(self, test_x):
        print(test_x.shape)
        b_row = test_x.shape[0]
        print(np.dot(test_x, self.w).shape)
        h = self.sigmoid(np.dot(test_x, self.w) + np.tile(self.b[0,:], (b_row, 1)))
        result = np.dot(h, self.beta)
        result = [item.tolist().index(max(item.tolist())) for item in result]
        return result


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
    a = HiddenLayer(x_train, 60)
    a.classifisor_train(y_train)

    result = a.classifisor_test(x_test)

    print(result)
    print(y_test)
    print(metrics.accuracy_score(y_test, result))

    print(test_x.shape)
    print(x_test.shape)
    print(x_train.shape)
    result = a.classifisor_test(test_x)
    pre_y = np.array(result).astype(int)
    result = zip(no, pre_y)
    file_out = open("submission.csv", 'w')
    file_out.write('id,target\n')
    for item in result:
        file_out.write(str(item[0]) + ',' + str(item[1]) + "\n")
    file_out.close()