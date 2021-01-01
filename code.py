import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.utils.extmath import randomized_svd

""" Generate random index """
def getRandomIndex(n, x):
    index = np.random.choice(np.arange(n), size=x, replace=False)
    return index

if __name__ == "__main__":
    """ Read file """
    origin_file = []
    n = 0
    p = 0
    with open('ratings.csv') as myFile:
        lines =  csv.reader(myFile)
        start = True
        for line in lines:
            if(start):
                start = False
            else:
                uid = int(line[0])
                mid = int(line[1])
                if(uid>n):
                    n = uid
                if(mid>p):
                    p = mid
                insert = [uid, mid, float(line[2])]
                origin_file.append(insert)
    # print(origin_file)
    # print(n)
    # print(p)

    """ Generate matrix M """
    M = np.zeros((n+1, p+1))
    # print(M.shape)
    omega = []
    for record in origin_file:
        M[record[0]][record[1]] = record[2]
        omega.append([record[0],record[1],record[2]])
    print(M)
    print("Shape: ", M.shape)
    print()
    # print(omega)

    """ Divide the training set and the test set """
    omega_test_index = np.array(getRandomIndex(len(omega)-1, round(0.1*len(omega))))
    omrga_train_index = np.delete(np.arange(len(omega)-1), omega_test_index)
    M_test = np.zeros((n+1, p+1))
    M_train = np.zeros((n + 1, p + 1))
    size = 0
    for i in range(0,len(omega)):
        if i in omega_test_index:
            M_test[omega[i][0]][omega[i][1]] = omega[i][2]
        if i in omrga_train_index:
            M_train[omega[i][0]][omega[i][1]] = omega[i][2]
    print("Number of nozero entries in test: ", np.count_nonzero(M_test))
    print("Number of nozero entries in training: ", np.count_nonzero(M_train))

    """ Learning """
    rank = np.linalg.matrix_rank(M_train)
    print("rank：", rank)
    r = 6
    u_0 = np.zeros((1, r))
    v_0 = np.zeros((1, r))
    # u = np.random.uniform(0, 5, (n, r))
    # v = np.random.uniform(0, 5, (p, r))
    u = np.random.rand(n, r)
    v = np.random.rand(p, r)
    u = np.r_[u_0, u]
    v = np.r_[v_0, v]
    print("Shape of u: ", u.shape)
    print("Shape of v: ", v.shape)

    # lamda = 1
    lamdas = [10 ** (-6), 10 ** (-3), 0.1, 0.5, 2, 5, 10, 20, 50, 100, 500, 1000]
    RMSEs= []
    for lamda in lamdas:
        eta = 0.05
        time = 20
        objectives = []
        times = []
        for t in range(1, time):
            for i in range(0, n+1):
                for j in range(0, p+1):
                    if M_train[i,j] > 0:
                        first_in = M_train[i,j]-np.dot(u[i,:],np.transpose(v[j,:]))
                        for rr in range(r):
                            first_u = first_in * (-1) * v[j][rr]
                            gradient_u = first_u + lamda * u[i][rr]
                            first_v = first_in * (-1) * u[i][rr]
                            gradient_v = first_v + lamda * v[j][rr]
                            u[i][rr] = u[i][rr] - eta * gradient_u
                            v[j][rr] = v[j][rr] - eta * gradient_v
            print("t: ",t)
            print("learning rate: ", eta)
            uF = np.linalg.norm(u) ** 2
            vF = np.linalg.norm(v) ** 2
            print("lamda: ",lamda)
            nom = np.linalg.norm(M_train-np.dot(u, np.transpose(v))) ** 2
            print("nom: ",nom)
            min = 1/2 * nom ** 2 + lamda/2 * (uF + vF)
            print("min: ",min)
            times.append(t)
            objectives.append(min)
        plt.figure()
        plt.plot(times, objectives)
        plt.xlabel('Times')
        plt.ylabel('Min[F(U,V)]')
        plt.show()

        """ Evaluation """
        RMSE = (np.linalg.norm(M_test - np.dot(u, np.transpose(v)))**2 / omega_test_index.size) ** (1/2)
        RMSEs.append(RMSE)
        print("RMSE: ",RMSE)
    plt.figure()
    plt.plot(lamdas, RMSEs)
    plt.xlabel('Lamda')
    plt.ylabel('RMSE')
    plt.show()

