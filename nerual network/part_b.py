import scipy.io as sio
import numpy as np
from sklearn.svm import LinearSVC


if __name__ == "__main__":
    data = sio.loadmat('ExtYaleB10.mat')
    train_sample = data['train']
    test_sample = data['test']
    x_train_full = np.column_stack(train_sample[0, i][:, :, j].reshape(192 * 168, 1) for i in range(10) for j in range(50))
    x_test_full = np.column_stack(test_sample[0, i][:, :, j].reshape(192 * 168, 1) for i in range(10) for j in range(14))
    I = np.identity(10)
    y_train = np.column_stack(I[:, i] for i in range(10) for j in range(50))
    y_test = np.column_stack(I[:, i] for i in range(10) for j in range(14))

    y_train_1 = np.zeros(500, )
    for i in range(10):
        for j in range(50):
            y_train_1[i * 50 + j] = i

    y_test_1 = np.zeros(140, )
    for i in range(10):
        for j in range(14):
            y_test_1[i * 14 + j] = i

    clf = LinearSVC()
    clf.fit(x_train_full.T, y_train_1)
    print(clf.score((x_test_full).T, y_test_1))


