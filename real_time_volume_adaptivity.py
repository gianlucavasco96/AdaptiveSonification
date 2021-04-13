from pyo import *
from functions import *

s = Server().boot()

# sonification sound: piano samples, C major scale
sonification_path = 'D:/Gianluca/Università/Magistrale/Tesi/piano.wav'
dur = get_duration(sonification_path)

sf = SfPlayer(sonification_path)                                # soundfile player

# Recorded noise
t = NewTable(length=dur, chnls=2)                               # create an empty table ready for recording
inp = Input([0, 1])                                             # retrieves the stereo input
rec = TableRec(inp, table=t, fadetime=0.05).play()              # table recorder
osc = Osc(table=t, freq=t.getRate())                            # reads the content of the table for the spec. duration

# Follow the amplitude envelope of the input sound
env_sound = Follower(sf)
env_noise = Follower(osc)

# modulation factor
# TODO: capire perché il fattore di modulazione non si modifica sullo slider
gain = Sig(5.0)
# gain.ctrl(map_list=[SLMap(0.01, 10.0, "lin", "gain", 5.0)], title='gain control')
sf.mul = gain * env_noise / env_sound

# playback
sf.out()
osc.out(dur=dur)

# Display the waveform
sc = Scope([sf, osc])

s.gui(locals())
