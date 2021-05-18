import os
import time

from functions import setControlVariables

# TRAINING OR TEST
test = True

# CONTROL VARIABLES
user = 15
repetition = 4
task = 16

script, speed, sentence, soundscape = setControlVariables(user, repetition, task, test)

print("TEST" if test else "TRAINING")
print(soundscape)
time.sleep(1)
os.system(script + " " + str(speed) + " " + str(sentence) + " " + str(test))
