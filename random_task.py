import random

import numpy as np

repetitions = 4
soundscape = ["folla", "traffico", "metro"]
technique = ["nessuna", "volume", "eq 0", "eq positivo", "velocità base"]
frase = list(np.arange(1, 64))

task_list = []

for a in soundscape:
    for b in technique:
        task_list.append([a, b])

task_list.append("no soundscape/adattamento")

idx = list(range(16))
count = 1

for i in range(repetitions):
    random.shuffle(idx)
    for n in idx:
        print(task_list[n], count)
        count += 1
