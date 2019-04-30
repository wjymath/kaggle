import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split  # 数据集的分割函数
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC

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

    # lasso
    model_aic = LassoLarsIC(criterion='aic')
    model_aic.fit(x, y)
    alpha_aic_ = model_aic.alpha_
    print("alpha_aic_ is " + str(alpha_aic_))

    model_bic = LassoLarsIC(criterion='bic')
    model_bic.fit(x, y)
    alpha_bic_ = model_bic.alpha_
    print("alpha_bic_ is " + str(alpha_bic_))

    # model_cv = LassoCV(cv=20)
    # model_cv.fit(x, y)
    # alpha_cv_ = model_cv.alpha_
    # print("alpha_cv_ is " + str(alpha_cv_))
    #
    # model_lar = LassoLarsCV(cv=20)
    # model_lar.fit(x, y)
    # alpha_lar_ = model_lar.alpha_
    # print("alpha_lar_ is " + str(alpha_lar_))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    sample_dict = {1: 0.36, 0: 0.64}
    reg = linear_model.Lasso(alpha=alpha_bic_)
    reg.fit(x_train, y_train)
    pre_y = reg.predict(test_x)
    pre_y = pre_y.astype(int)
    result = zip(no, pre_y)
    file_out = open("submission.csv", 'w')
    file_out.write('id,target\n')
    for item in result:
        file_out.write(str(item[0]) + ',' + str(item[1]) + "\n")
    file_out.close()