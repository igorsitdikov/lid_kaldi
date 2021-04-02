#!/usr/bin/env python3

from lid import Model, KaldiRecognizer
import sys
import os
import wave

wf = wave.open(sys.argv[1], "rb")
if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
    print ("Audio file must be WAV format mono PCM.")
    exit (1)

model = Model("lid-107")
rec = KaldiRecognizer(model, wf.getframerate())
data = wf.readframes(-1)
rec.AcceptWaveform(data)
print(rec.Result())

