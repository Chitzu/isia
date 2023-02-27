from sklearn import neural_network, metrics
import pandas as pd
import numpy as np


data = pd.read_csv('lymphography.data') # citirea setului de date
new_data = np.array(data)
np.random.shuffle(new_data)

# impartirea datelor in date de train si test
data_train = new_data[:110, :18]
data_test = new_data[110:, :18]

# impartirea etichetelor in train si test
etichete_train = new_data[:110, 18]
etichete_test = new_data[110:, 18]

# 1 strat ascuns , learing rate 0.01

clf = neural_network.MLPClassifier(hidden_layer_sizes=30, learning_rate_init=0.01)
clf.fit(data_train, etichete_train)

predictii = clf.predict(data_test)

acuratete = metrics.accuracy_score(y_true=etichete_test, y_pred=predictii)
print(f"Acuratetea1={acuratete*100}%")

# 2 straturi ascunse , learning rate = 0.01

clf = neural_network.MLPClassifier(hidden_layer_sizes=(30, 15), learning_rate_init=0.01)
clf.fit(data_train, etichete_train)

predictii = clf.predict(data_test)

acuratete = metrics.accuracy_score(y_true=etichete_test, y_pred=predictii)
print(f"Acuratetea2={acuratete*100}%")

# 2 straturi ascunse egale , learning rate = 0.1

clf = neural_network.MLPClassifier(hidden_layer_sizes=(30, 30), learning_rate_init=0.1)
clf.fit(data_train, etichete_train)

predictii = clf.predict(data_test)

acuratete = metrics.accuracy_score(y_true=etichete_test, y_pred=predictii)
print(f"Acuratetea3={acuratete*100}%")