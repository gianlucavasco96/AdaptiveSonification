from pyo import *
from functions import *

s = Server().boot()

# sonification sound: piano samples, C major scale
sonification_path = 'D:/Gianluca/Università/Magistrale/Tesi/piano.wav'
dur = get_duration(sonification_path)

sf = SfPlayer(sonification_path).out()

t = NewTable(length=dur, chnls=2)

# Retrieves the stereo input
inp = Input([0, 1])

# Table recorder
rec = TableRec(inp, table=t, fadetime=0.05).play()

# Reads the content of the table in loop.
osc = Osc(table=t, freq=t.getRate()).out(dur=dur)

# Display the waveform
sc = Scope([sf, osc])

s.gui(locals())
