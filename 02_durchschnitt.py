import numpy as np
import matplotlib.pyplot as plt
from mnist import laden

train_x, train_y, test_x, test_y = laden()

train_x = train_x / 255
test_x = test_x / 255

durchschnitt = []
for z in range(10):
    alle_bilder_von_z = train_x[train_y == z]
    durchschnitt.append(alle_bilder_von_z.mean(axis=0))

plt.figure(figsize=(10, 2))
for z in range(10):
    plt.subplot(1, 10, z + 1)
    plt.imshow(durchschnitt[z], cmap="gray_r")
    plt.title(str(z))
    plt.axis("off")
plt.savefig("bilder/durchschnitt.png")
plt.close()

richtig = 0
for i in range(len(test_x)):
    abstaende = []
    for z in range(10):
        abstand = np.sum((test_x[i] - durchschnitt[z]) ** 2)
        abstaende.append(abstand)
    tipp = np.argmin(abstaende)
    if tipp == test_y[i]:
        richtig = richtig + 1

print("richtig:", richtig, "von", len(test_x))
print("genauigkeit:", richtig / len(test_x) * 100, "%")
