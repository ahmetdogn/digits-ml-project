import numpy as np
import sys
from PIL import Image

if len(sys.argv) < 2:
    print("so benutzen: python 05_eigene_zahl.py bild.png")
    exit()

modell = np.load("modell.npz")
w1 = modell["w1"]
b1 = modell["b1"]
w2 = modell["w2"]
b2 = modell["b2"]

for datei in sys.argv[1:]:
    bild = Image.open(datei).convert("L")
    bild = np.array(bild) / 255
    bild = 1 - bild
    bild[bild < 0.2] = 0

    zeilen = np.where(np.max(bild, axis=1) > 0)[0]
    spalten = np.where(np.max(bild, axis=0) > 0)[0]
    bild = bild[zeilen[0]:zeilen[-1] + 1, spalten[0]:spalten[-1] + 1]

    hoehe, breite = bild.shape
    if hoehe > breite:
        neue_hoehe = 20
        neue_breite = max(1, int(breite * 20 / hoehe))
    else:
        neue_breite = 20
        neue_hoehe = max(1, int(hoehe * 20 / breite))

    klein = Image.fromarray((bild * 255).astype(np.uint8)).resize((neue_breite, neue_hoehe))
    klein = np.array(klein) / 255

    fertig = np.zeros((28, 28))
    oben = (28 - neue_hoehe) // 2
    links = (28 - neue_breite) // 2
    fertig[oben:oben + neue_hoehe, links:links + neue_breite] = klein

    x = fertig.reshape(1, 784)
    h = np.maximum(0, x @ w1 + b1)
    out = h @ w2 + b2
    out = np.exp(out - np.max(out))
    out = out / np.sum(out)

    zahl = np.argmax(out)
    print(datei, "->", zahl, "(" + str(round(out[0][zahl] * 100, 1)) + "% sicher)")
