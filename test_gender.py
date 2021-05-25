import pandas as pd
import numpy as np
from autorank import autorank, plot_stats, create_report, latex_table
from matplotlib import pyplot as plt
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
from functions import printBoxplot, starMatrix, printViolinplot

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# CONTROL VARIABLES
column = 'Parole corrette'
sesso = 'F'
outliers = True
rank_order = 'ascending' if column != 'Parole corrette' else 'descending'

path = 'D:/Gianluca/Università/Magistrale/Tesi/test/'
file_order = 'ordine test.xlsx'
file_data = 'dati test.xlsx'
file_users = 'utenti.xlsx'

order = pd.read_excel(path + file_order)
data = pd.read_excel(path + file_data)
users = pd.read_excel(path + file_users)

df = pd.merge(order, data, how='left', on=['Utente', 'Ripetizione', 'Task'])

# condizione di controllo: nessun adattamento e nessun soundscape
control = df[df['Soundscape'] == 'nessuno']

if not outliers:
    # outlier removal with MUSHRA method: users with less than 90% of score on control condition are discarded
    thresh = 360                                    # 90% of 400
    discard = np.zeros(15)

    for user in range(1, 16):
        score = np.sum(control[control['Utente'] == user]['Parole corrette'])
        if score < thresh:
            discard[user - 1] += 1

    discarded_users = np.where(discard)[0] + 1      # indexes of users that have to be discarded

    filt_order = order.copy()
    filt_data = data.copy()
    filt_users = users.copy()

    for user in discarded_users:
        filt_order = filt_order[(filt_order['Utente'] != user)]
        filt_data = filt_data[(filt_data['Utente'] != user)]
        filt_users = filt_users[(filt_users['Utente'] != user)]

    # join dei dati filtrati
    df = pd.merge(filt_order, filt_data, how='left', on=['Utente', 'Ripetizione', 'Task'])
    df = pd.merge(df, filt_users, how='left', on='Utente')
else:
    df = pd.merge(df, users, how='left', on='Utente')

df_gender = df[df['Sesso'] == sesso]

# nessun adattamento in soundscape
none = df_gender[df_gender['Adattamento'] == 'nessuno']
none = none[none['Soundscape'] != 'nessuno']

# velocità base
slow = df_gender[df_gender['Velocità'] == 'lenta']

# volume
volume = df_gender[df_gender['Adattamento'] == 'volume']

# eq 0 dB
eq_0 = df_gender[df_gender['Adattamento'] == 'eq 0']

# eq 10 dB
eq_10 = df_gender[df_gender['Adattamento'] == 'eq positivo']

# boxplots
labels = ['nessuno', 'velocità base', 'volume', 'eq 0', 'eq 10']
data = [none[column], slow[column], volume[column], eq_0[column], eq_10[column]]

printBoxplot(data, labels, column)
# printViolinplot(df, y=column)

# convert to array
none_score = np.array(none[column])
slow_score = np.array(slow[column])
volume_score = np.array(volume[column])
eq0_score = np.array(eq_0[column])
eq10_score = np.array(eq_10[column])

# Friedman's test
statistic, pvalue = friedmanchisquare(none_score, slow_score, volume_score, eq0_score, eq10_score)
print(statistic, pvalue)

# post hoc test
score_array = np.array([none_score, slow_score, volume_score, eq0_score, eq10_score])
result = posthoc_nemenyi_friedman(score_array.T)
result = result.set_axis(labels, axis='index').set_axis(labels, axis='columns')

starMatrix(result)

# autorank test
d = {'nessuno': none_score, 'velocità base': slow_score, 'volume': volume_score,
     'eq 0': eq0_score, 'eq 10': eq10_score}
df_words = pd.DataFrame(d)

result = autorank(df_words, alpha=0.05, verbose=False, order=rank_order)
print(result)

create_report(result)

latex_table(result)

plot_stats(result)
plt.show()
