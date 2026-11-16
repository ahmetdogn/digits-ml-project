import numpy as np
import matplotlib.pyplot as plt
from mnist import laden

train_x, train_y, test_x, test_y = laden()

train_x = train_x.reshape(60000, 784) / 255
test_x = test_x.reshape(10000, 784) / 255

neuronen = 128
lernrate = 0.1
epochen = 10
batch_size = 64

np.random.seed(1)
w1 = np.random.randn(784, neuronen) * np.sqrt(2 / 784)
b1 = np.zeros(neuronen)
w2 = np.random.randn(neuronen, 10) * np.sqrt(2 / neuronen)
b2 = np.zeros(10)


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def one_hot(y):
    ergebnis = np.zeros((len(y), 10))
    ergebnis[np.arange(len(y)), y] = 1
    return ergebnis


def vorhersage(x):
    h = relu(x @ w1 + b1)
    out = softmax(h @ w2 + b2)
    return np.argmax(out, axis=1)


def genauigkeit(x, y):
    return np.mean(vorhersage(x) == y) * 100


loss_liste = []
train_acc_liste = []
test_acc_liste = []

for epoche in range(epochen):
    mischen = np.random.permutation(60000)
    train_x = train_x[mischen]
    train_y = train_y[mischen]

    gesamt_loss = 0
    for i in range(0, 60000, batch_size):
        x = train_x[i:i + batch_size]
        y = one_hot(train_y[i:i + batch_size])
        n = len(x)

        z1 = x @ w1 + b1
        h = relu(z1)
        out = softmax(h @ w2 + b2)

        loss = -np.sum(y * np.log(out + 0.0000001)) / n
        gesamt_loss = gesamt_loss + loss

        d_out = (out - y) / n
        d_w2 = h.T @ d_out
        d_b2 = np.sum(d_out, axis=0)
        d_h = d_out @ w2.T
        d_z1 = d_h * (z1 > 0)
        d_w1 = x.T @ d_z1
        d_b1 = np.sum(d_z1, axis=0)

        w1 = w1 - lernrate * d_w1
        b1 = b1 - lernrate * d_b1
        w2 = w2 - lernrate * d_w2
        b2 = b2 - lernrate * d_b2

    durchschnitt_loss = gesamt_loss / (60000 / batch_size)
    train_acc = genauigkeit(train_x, train_y)
    test_acc = genauigkeit(test_x, test_y)
    loss_liste.append(durchschnitt_loss)
    train_acc_liste.append(train_acc)
    test_acc_liste.append(test_acc)
    print("epoche", epoche + 1, "loss:", round(durchschnitt_loss, 4), "train:", round(train_acc, 2), "% test:", round(test_acc, 2), "%")

np.savez("modell.npz", w1=w1, b1=b1, w2=w2, b2=b2)
print("modell gespeichert")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochen + 1), loss_liste)
plt.xlabel("epoche")
plt.ylabel("loss")
plt.subplot(1, 2, 2)
plt.plot(range(1, epochen + 1), train_acc_liste, label="train")
plt.plot(range(1, epochen + 1), test_acc_liste, label="test")
plt.xlabel("epoche")
plt.ylabel("genauigkeit %")
plt.legend()
plt.savefig("bilder/training.png")
plt.close()

tipps = vorhersage(test_x)
falsche = np.where(tipps != test_y)[0]
print("falsch erkannt:", len(falsche), "von 10000")

plt.figure(figsize=(12, 3))
for i in range(20):
    nr = falsche[i]
    plt.subplot(2, 10, i + 1)
    plt.imshow(test_x[nr].reshape(28, 28), cmap="gray_r")
    plt.title(str(test_y[nr]) + " -> " + str(tipps[nr]), color="red")
    plt.axis("off")
plt.savefig("bilder/falsch.png")
plt.close()

plt.figure(figsize=(10, 5))
for i in range(32):
    plt.subplot(4, 8, i + 1)
    plt.imshow(w1[:, i].reshape(28, 28), cmap="seismic")
    plt.axis("off")
plt.savefig("bilder/gewichte.png")
plt.close()
