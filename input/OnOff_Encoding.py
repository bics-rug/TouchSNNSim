import torch
import torch.nn as nn
import snntorch as snn
from bicsnn.models import *
from snntorch import spikeplot as splt
from snntorch import spikegen
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import argrelextrema
from scipy.interpolate import interp1d

DEBUG = False
DirOut = "../plots/"
save_fig = False

def generate_sinusoidal_wave(frequency, amplitude, duration, sampling_rate):
  """
  Generates a sinusoidal wave.

  Args:
    frequency: The frequency of the wave in Hz.
    amplitude: The amplitude of the wave.
    duration: The duration of the wave in seconds.
    sampling_rate: The sampling rate of the wave.

  Returns:
    A NumPy array containing the sinusoidal wave.
  """

  # Calculate the number of samples
  num_samples = int(duration * sampling_rate)

  # Generate time points
  time = np.linspace(0, duration, num_samples)

  # Calculate the angular frequency
  angular_frequency = 2 * np.pi * frequency

  # Generate the sinusoidal wave
  wave = amplitude * np.sin(angular_frequency * time)

  return wave

def sample_to_changes(sample, f, threshold, save=save_fig):
    ''' Convert one sample time-based to event-based
        sample: time-based sample
        f: frequency of the time-based sequence
        threshold: create an event at certain threshold value
        Find the local max and min values of the sequence and applied interpolation in time based on threshold to find
        the correspondent event time.
    '''
    Precision = 4   # Fix numerical errors due to float values in arange method 29.000000000000014
    n = sample.shape[0]
    dt = 1/f
    taxel_samples = np.transpose(sample, (1, 0)).tolist()
    sample_list = list()
    for nt, taxel in enumerate(taxel_samples):
        # Find indexes in the sequence with local maximum and minimum to apply interpolation.
        txl = np.array(taxel, dtype=int)
        #   max
        ind_max = np.squeeze(np.array(argrelextrema(txl, np.greater_equal)))
        d_ixtr = np.insert(np.diff(ind_max), 0, -1)    # Match dimensions of the index
        max_p = ind_max[d_ixtr != 1]
        #   min
        ind_min = np.squeeze(np.array(argrelextrema(txl, np.less_equal)))
        d_ixtr = np.insert(np.diff(ind_min), 0, -1)  # Match dimensions of the index
        min_p = ind_min[d_ixtr != 1]
        #   add index with same values
        all_indx = np.append(max_p, min_p)
        i = 0
        while i < len(all_indx):
            try:
                ival = all_indx[i]
                if txl[ival + 1] - txl[ival] == 0:
                    all_indx = np.append(all_indx, np.array(ival + 1))
            except IndexError:
                None
            i += 1
        # Corresponding values in the sequence
        all_t = np.unique(np.sort(all_indx))
        all_values = txl[all_t]
        # Find the events [ON, OFF]
        taxel_list = list()
        on_events = np.array([]); off_events = np.array([])
        # Compare each pair of points and generate event times based on threshold
        last_value = all_values[0]  # Last value storage controls when threshold is not reached
        for i in range(len(all_values) - 1):
            d_pair = all_values[i+1] - last_value
            if d_pair > 0:
                start = last_value + threshold
                stop = all_values[i+1] + 0.0001
                spk_values = np.round(np.arange(start, stop, threshold), Precision)
                # Interpolation with all the values of the pair
                pts = all_t[i+1] - all_t[i] + 1
                t_interp = np.linspace(all_t[i], all_t[i+1], pts, dtype=int)
                vals_interp = txl[t_interp]
                f = interp1d(vals_interp, t_interp.astype(float), 'linear')
                on_events = np.append(on_events, np.apply_along_axis(f, 0, spk_values))
                last_value = spk_values[-1] if spk_values.size > 0 else last_value   # Change value of sensor when spike
            elif d_pair < 0:
                start = last_value - threshold
                stop = all_values[i+1] - 0.0001 # No Threshold
                spk_values = np.round(np.arange(start, stop, -1*threshold), Precision)
                # Interpolation with all the values of the pair
                pts = all_t[i+1] - all_t[i] + 1
                t_interp = np.linspace(all_t[i], all_t[i+1], pts, dtype=int)
                vals_interp = txl[t_interp]
                f = interp1d(vals_interp, t_interp, 'linear')
                off_events = np.append(off_events, np.apply_along_axis(f, 0, spk_values))
                last_value = spk_values[-1] if spk_values.size > 0 else last_value    # Change value of sensor when spike
        # Assign events
        taxel_list.append((on_events * dt).tolist())
        taxel_list.append((off_events * dt).tolist())
        sample_list.append(taxel_list)
        # Plot conversions. Run in debug mode
        if DEBUG:
            plt.rcParams['text.usetex'] = True
            f1 = plt.figure()
            axes = plt.axes()
            n = len(txl)
            scale = 1/5
            axes.set_xlim([0, ((scale * n) - 0.5) * dt])
            axes.set_ylim([-0.5, 0.5])
            if taxel_list[0]:
                plt.eventplot(taxel_list[0], lineoffsets=0.15,
                              colors='green', linelength=0.25)
            if  taxel_list[1]:
                plt.eventplot(taxel_list[1], lineoffsets=-0.15,
                              colors='red', linelength=0.25)

            axes.set_ylabel(r'$\vartheta = ${}'.format(str(threshold)))
            if save:
                plt.savefig('{}encoding_TH{}_taxel_{}_events.png'.format(DirOut, str(threshold), str(nt)), dpi=200)
            f2 = plt.figure()
            axes = plt.axes()
            axes.set_xlim([0, ((scale * n) - 0.5) * dt])
            plt.plot(np.arange(start=0, stop=(n - 0.5) * dt, step=dt), txl - txl[0], '-o')
            axes.set_ylabel("Sensor value")
            axes.set_xlabel('t(s)')
            if save:
                plt.savefig('{}encoding_TH{}_taxel_{}_sample.png'.format(DirOut, str(threshold), str(nt)), dpi=200)

    return sample_list

def convert_to_seq(time_list, samples):
    # This only works in the ms domain - every sample represents 1ms
    event_list = []
    for events in time_list[0]:
        event_t = np.zeros([samples])
        indx_event = [int(i * 1000) for i in events]
        event_t[indx_event] = 1
        event_list.append(event_t)
    return event_list

def main_encoding():
    # Generate sinusoid
    frequency = 10  # Hz
    amplitude = 5
    duration = 0.5  # seconds
    sampling_rate = 1000  # samples/second

    wave = generate_sinusoidal_wave(frequency, amplitude, duration, sampling_rate)
    # Output is a list with timestamps of the events
    time_list = sample_to_changes(np.expand_dims(wave, axis=1), sampling_rate,1)
    # Convert the timestamps to time sequence for convenience
    spk_seq = convert_to_seq(time_list, int(duration * sampling_rate))
    data_en = spikegen.delta(torch.tensor(wave), threshold=0.1, padding=True, off_spike=True)
    # Plot the wave
    fig, ax = plt.subplots(4, figsize=(12, 8), sharex=True,
                           gridspec_kw={'height_ratios': [1, 0.4, 0.4, 0.4]})
    ax[0].plot(wave)
    splt.raster(torch.tensor(spk_seq[0]), ax[1], s=400, c="black", marker="|")
    splt.raster(torch.tensor(spk_seq[1]), ax[2], s=400, c="black", marker="|")
    splt.raster(data_en, ax[3], s=400, c="black", marker="|")
    plt.xlabel("Time (samples 1ms)")
    plt.ylabel("Amplitude")
    plt.title("Sinusoidal Wave")
    plt.show()

if __name__ == '__main__':
    main_encoding()