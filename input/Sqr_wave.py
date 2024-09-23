import torch
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from snntorch import spikegen

def sqr_wave_in(w, period, threshold, num_steps):
    #Define a periodic square wave input.
    square_wave = torch.zeros(num_steps)
    for n in num_steps:
        if n % period < period/2:
            square_wave[n] = w
        else:
            square_wave[n] = -w

    return square_wave

print(sqr_wave_in(1, 5, 0.3, 100))