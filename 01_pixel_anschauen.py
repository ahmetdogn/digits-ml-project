import numpy as np
import matplotlib.pyplot as plt
from mnist import laden

train_x, train_y, test_x, test_y = laden()

print("train bilder:", train_x.shape)
print("test bilder:", test_x.shape)
print("ein bild:", train_x[0].shape)
print("kleinster wert:", train_x.min(), "groesster wert:", train_x.max())
print("label vom ersten bild:", train_y[0])
print()

bild = train_x[0]
for zeile in range(28):
    text = ""
    for spalte in range(28):
        if bild[zeile][spalte] > 128:
            text = text + "##"
        elif bild[zeile][spalte] > 0:
            text = text + ".."
        else:
            text = text + "  "
    print(text)

plt.figure(figsize=(7, 7))
plt.imshow(bild, cmap="gray_r")
for zeile in range(28):
    for spalte in range(28):
        wert = bild[zeile][spalte]
        if wert > 0:
            if wert > 128:
                farbe = "white"
            else:
                farbe = "black"
            plt.text(spalte, zeile, str(wert), fontsize=4, ha="center", va="center", color=farbe)
plt.title("das erste bild, label = " + str(train_y[0]))
plt.savefig("bilder/pixel.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 4))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(train_x[i], cmap="gray_r")
    plt.title(str(train_y[i]))
    plt.axis("off")
plt.savefig("bilder/beispiele.png")
plt.close()

anzahl = []
for z in range(10):
    anzahl.append(np.sum(train_y == z))
print()
print("wie oft kommt jede zahl vor:")
for z in range(10):
    print(z, anzahl[z])
