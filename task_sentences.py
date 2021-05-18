import pandas as pd

path = 'D:/Gianluca/Documenti/Frasi.xlsx'
df = pd.read_excel(path, header=None)[0]

start = 0
stop = start + 16 - 1
trasl = 15
for i in range(start, stop):
    idx = (i % 4) * 4 + start + trasl
    if idx >= start + 16:
        idx -= 16
    print(df[idx])

print()

for i in range(start, stop):
    idx = (i % 4) * 4 + start + trasl
    if idx >= start + 16:
        idx -= 16
    print(idx)
