import numpy as np
import time
from mnist import laden

train_x, train_y, test_x, test_y = laden()

train_x = train_x.reshape(60000, 784) / 255
test_x = test_x.reshape(10000, 784) / 255

train_x = train_x[:10000]
train_y = train_y[:10000]
anzahl_tests = 1000
k = 3

start = time.time()
richtig = 0
for i in range(anzahl_tests):
    abstaende = np.sum((train_x - test_x[i]) ** 2, axis=1)
    naechste = np.argsort(abstaende)[:k]
    stimmen = train_y[naechste]
    tipp = np.bincount(stimmen).argmax()
    if tipp == test_y[i]:
        richtig = richtig + 1
    if i % 100 == 0:
        print(i, "/", anzahl_tests)

print("k =", k)
print("genauigkeit:", richtig / anzahl_tests * 100, "%")
print("hat", round(time.time() - start, 1), "sekunden gedauert")
