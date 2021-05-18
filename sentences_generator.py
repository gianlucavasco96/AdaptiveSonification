import pandas as pd
import numpy as np

# instantiate the sentences dictionary
d = {"nomi": ["Sofia", "Marco", "Anna", "Sara", "Chiara", "Maria", "Luca", "Andrea", "Matteo", "Simone"],
     "verbi": ["compra", "vuole", "prende", "dipinge", "vede", "cerca", "trascina", "regala", "possiede", "manda"],
     "numerali": ["due", "poche", "quattro", "cinque", "molte", "sette", "otto", "nove", "dieci", "venti"],
     "oggetti": ["scatole", "matite", "tazze", "pietre", "tavole", "palle", "macchine", "sedie", "bottiglie", "porte"],
     "aggettivi": ["azzurre", "piccole", "normali", "nuove", "belle", "bianche", "grandi", "utili", "nere", "rosse"]}

# create the data frame
df = pd.DataFrame(data=d)

# open the txt file for writing
path = 'D:/Gianluca/Università/Magistrale/Tesi/frasi/sentences.txt'
text_file = open(path, "w")

for i in range(100):
    sentence = ""
    # create a list of indexes for constructing the sentence
    idx = np.zeros(5)
    for j in range(5):
        idx[j] = np.random.randint(0, 10)
        sentence += df[df.keys()[j]][idx[j]]
        if j == 4:
            sentence += "."
            if i < 99:
                text_file.write(sentence + '\n')
            else:
                text_file.write(sentence)
        else:
            sentence += " "
