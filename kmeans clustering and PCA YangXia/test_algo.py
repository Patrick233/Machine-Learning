import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import kmeans as kmeans
import spectral_clustering as spectral
import pca as pca


def main():
    r = 5  # number of random initial
    d = 2  # reduced dimension
    k = 2  # number of cluster
    N = 200  # sample size
    part = input("Input part A/B \n")
    number = input("Input number 1~6 \n")

    if (part == 'A'):
        data = sio.loadmat('HW3_Data/dataset1.mat')
        Y = data['Y']
        if (number == 1):
            plt.close()
            x1 = Y[0, :]
            y1 = Y[1, :]
            plt.scatter(x1, y1)
            plt.show()
            plt.close()
        if (number == 2):
            plt.close()
            x1 = Y[0, :]
            y1 = Y[2, :]
            plt.scatter(x1, y1)
            plt.show()
        if (number == 3):
            plt.close()
            u,y_reduced = pca.pca(Y, 2)
            x = y_reduced[0,:]
            y = y_reduced[1,:]
            plt.scatter(np.asarray(x), np.asarray(y))
            plt.show()
        if (number == 4):
            plt.close()
            result = kmeans.k_means(np.matrix(Y), k, r)
            x1 = []
            y1 = []
            x2 = []
            y2 = []
            U,y_2d = pca.pca(Y, d)
            for i in range(N):
                if (result[i] == 0):
                    x1.append(y_2d[0, i])
                    y1.append(y_2d[1, i])
                else:
                    x2.append(y_2d[0, i])
                    y2.append(y_2d[1, i])

            plt.scatter(x1, y1, color='red')
            plt.scatter(x2, y2, color='blue')
            plt.show()
        if (number == 5):
            plt.close()
            U, y_2d = pca.pca(Y, d)
            result = kmeans.k_means(y_2d, k, r)

            x1 = []
            y1 = []
            x2 = []
            y2 = []

            for i in range(200):
                if (result[i] == 0):
                    x1.append(y_2d[0, i])
                    y1.append(y_2d[1, i])
                else:
                    x2.append(y_2d[0, i])
                    y2.append(y_2d[1, i])

            plt.scatter(x1, y1, color='red')
            plt.scatter(x2, y2, color='blue')
            plt.show()

    if (part == 'B'):
        data = sio.loadmat('HW3_Data/dataset2.mat')
        Y = data['Y']
        if (number == 1):
            x1 = Y[0, :]
            y1 = Y[1, :]
            plt.scatter(x1, y1)
            plt.show()

        if (number == 2):

            x1 = Y[0, :]
            y1 = Y[2, :]
            plt.scatter(x1, y1)
            plt.show()
        if (number == 3):

            u,y_reduced = pca.pca(Y, 2)
            x = y_reduced[0,:]
            y = y_reduced[1,:]
            plt.scatter(np.asarray(x), np.asarray(y))
            plt.show()

        if (number == 4):

            U, y_2d = pca.pca(Y, d)
            result = kmeans.k_means(np.matrix(y_2d), k, r)
            x1 = []
            y1 = []
            x2 = []
            y2 = []

            for i in range(N):
                if (result[i] == 0):
                    x1.append(y_2d[0, i])
                    y1.append(y_2d[1, i])
                else:
                    x2.append(y_2d[0, i])
                    y2.append(y_2d[1, i])

            plt.scatter(x1, y1, color='red')
            plt.scatter(x2, y2, color='blue')
            plt.show()

        if (number == 5):
            kernel = pca.get_kernel(Y)
            u = pca.kernel_pca(kernel, d)
            y_reduced = np.matrix(kernel * u)
            result = kmeans.k_means(y_reduced.T, k, r)

            x1 = []
            y1 = []
            x2 = []
            y2 = []

            #U, y_2d = pca.pca(Y, d)
            y_2d = y_reduced.T
            for i in range(N):
                if (result[i] == 0):
                    x1.append(y_2d[0, i])
                    y1.append(y_2d[1, i])
                else:
                    x2.append(y_2d[0, i])
                    y2.append(y_2d[1, i])

            plt.scatter(x1, y1, color='red')
            plt.scatter(x2, y2, color='blue')
            plt.show()

        if(number == 6):
            W = np.matrix(spectral.get_w_matrix(Y, 5, 1))
            result = spectral.spectral_cluster(W, 2)

            x1 = []
            y1 = []
            x2 = []
            y2 = []

            U, x = pca.pca(Y, 2)
            y_2d = x
            for i in range(200):
                if (result[i] == 0):
                    x1.append(y_2d[0, i])
                    y1.append(y_2d[1, i])
                else:
                    x2.append(y_2d[0, i])
                    y2.append(y_2d[1, i])

            plt.scatter(x1, y1, color='red')
            plt.scatter(x2, y2, color='blue')
            plt.show()



if __name__ == "__main__":
    main()
