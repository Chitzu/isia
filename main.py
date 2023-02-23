from sklearn import datasets, neural_network
import numpy as np

date, etichete = datasets.load_iris(return_X_y=True)
# date, etichete = load_boston(return_X_y = True) returneaza
# o matrice cu 506 linii si 13 coloane in date si un vector
# cu 506 valori reale in etichete
# IMPARTIRE IN TRAIN SI TEST
date_train = date[110:,:]
etichete_train = etichete[110:]

date_test = date[:110,:]
etichete_test = etichete[:110]

nr=[10,50,200]
learning_rate=[1,0.1,0.0001]

errc=9999
# CREARE SI ANTRENARE MLP
for i in range(3):
    regr = neural_network.MLPRegressor(hidden_layer_sizes=(nr[i],), learning_rate_init=learning_rate[i])
    regr.fit(date_train,etichete_train)
    # TESTARE MLP
    predictii = regr.predict(date_test)
    err=0
    # EROARE


    for i in range(len(etichete_test)):
        err+=(predictii[i]-etichete_test[i])*(predictii[i]-etichete_test[i])
    err/=len(etichete_test)

    for i in range(len(etichete_test)):
        if etichete_test[i] == predictii[i]:
            acc = acc + 1
    acc = acc/len(etichete_test)



    if errc>err:
        errc=err
        nrc=nr[i]
print('MSE='+ str(errc) + '     Nr de neuroni=' + str(nrc))