## sachen ausprobiert mit dem netz

hab immer nur 3 epochen trainiert damit es schneller geht, sonst alles gleich wie in 04_neuronales_netz.py

normal (128 neuronen, lr 0.1): 96.27%

ohne die pixel durch 255 zu teilen: 11.35%
-> das netz lernt gar nichts, der loss bleibt bei 2.30. das ist ungefähr -log(1/10), also rät es einfach. die zahlen sind zu groß (bis 255) und dann gehen die werte kaputt

gewichte mit np.random.randn() ohne * np.sqrt(2/784): 88.77%
-> lernt schon, aber viel schlechter. hab dann gelesen dass man die gewichte klein machen muss je nachdem wie viele eingaben es gibt (he initialisierung)

lernrate 0.01: 91.1% (zu langsam)
lernrate 1.0: 96.24% (geht überraschend, hatte gedacht das explodiert)

16 neuronen: 92.69%
32 neuronen: 94.67%
512 neuronen: 96.52%
-> mehr neuronen = besser, aber von 128 auf 512 bringt fast nix mehr und dauert länger

## verglichen

durchschnittsbild: 82.03%
knn k=3: 91.9% (10000 train, 1000 test)
netz 10 epochen: 97.59%

## problem mit der 7

meine selbst gemalte 7 wird als 3 erkannt (77.7%). die 3 und die 1 gehen. ich glaub es liegt daran dass meine 7 anders aussieht als die in mnist, oder dass sie nicht richtig in der mitte ist. bei mnist ist die zahl nach dem schwerpunkt zentriert und nicht nach dem rahmen drumherum, das mach ich noch nicht. muss ich noch ausprobieren

## was ich gelernt hab

- ein bild ist nur eine matrix mit zahlen
- immer die daten normalisieren!!
- train genauigkeit ist höher als test genauigkeit (98.78 vs 97.59), das nennt man overfitting
- man darf die testdaten nicht zum trainieren benutzen sonst schummelt man
- matrix multiplikation aus lineare algebra braucht man wirklich überall
