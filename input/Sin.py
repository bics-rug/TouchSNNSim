import torch
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from snntorch import spikegen
import scipy.integrate as int
import numpy as np

#Plotting functions.
def cur_in_plot(cur_in, input, w, T):
    fig, ax = plt.subplots(2, figsize=(12, 5), sharex=True, gridspec_kw={'height_ratios': [1, 0.2]})
    ax[0].plot(cur_in)
    ax[0].set_ylabel("Input Current ($I_{in}$) [$mA$]")
    ax[0].set_ylim([-w, w])

    on_spks = input[:, 0, 0]
    off_spks = input[:, 0, 1]
    splt.raster(on_spks, ax[1], s=400, marker='|', c="green")
    splt.raster(off_spks, ax[1], s=400, marker='|', c="red")
    plt.ylabel("Input Spikes")
    ax[1].set_xlabel("Time steps [$ms$]")

    fig.suptitle("Spike Encoding for Sinusoidal Input Current ($f_{in}$" + f"= {round(1/T,2)*10**3} [Hz])", fontsize=20)
    plt.show()
def plot_mem(mem, w, num_steps, title=False):
  if title:
    plt.title(title)
  plt.plot(mem)
  plt.xlabel("Time step")
  plt.ylabel("Membrane Potential")
  plt.xlim([0, num_steps])
  plt.ylim([0, w])
  plt.show()

def plot_spk(spk, w, num_steps, title=False):
    fig = plt.figure(facecolor='white', figsize=(10, 2))
    ax = fig.add_subplot(111)
    if title:
        plt.title(title)
    splt.raster(spk,ax,s=400,marker='|',color="black")
    plt.xlabel("Time step")
    plt.ylabel("")
    plt.xlim([0, num_steps])
    plt.ylim([0, w])
    plt.show()

#Define a figure that plots the on spikes and off spikes in two separate plots on top of eachother.
def plot_onoff_spk(spk_on,spk_off, w, num_steps, title=False):
    fig,ax = plt.subplots(2,1,facecolor='white', figsize=(15, 7), sharex=True)
    if title:
        fig.suptitle(title,fontsize=30)
    splt.raster(spk_off,ax[0],s=400,marker='|',color="red")
    splt.raster(spk_on,ax[1],s=400,marker='|',color="green")
    plt.xlabel("Time step [$ms$]",fontsize=15)
    plt.ylabel("")
    ax[0].set_xlim([0, num_steps])
    ax[0].set_ylim([-w, w])
    ax[1].set_xlim([0, num_steps])
    ax[1].set_ylim([-w, w])
    plt.show()

#%%
def sin_enc(w, period, threshold, num_steps):
    #Define the input stimulus.
    omega = 2*torch.pi/period
    sin_in = torch.cos(omega*torch.arange(0,num_steps,1))
    cur_in = w*sin_in

    #Plot the input stimulus.
    #cur_in_plot(cur_in,w,num_steps,"Input Stimulus")

    #Spike data delta-mod.
    spk_data = spikegen.delta(cur_in,threshold=threshold,padding=True,off_spike=True)
    on_spks = torch.Tensor([0 if i == -1 else 1 for i in spk_data])
    off_spks = torch.Tensor([0 if i == 1 else 1 for i in spk_data])

    # Formatted spike data.
    spk_data_out = torch.zeros((num_steps,1,2))
    spk_data_out[:,0,0] = on_spks
    spk_data_out[:,0,1] = off_spks

    #Plot the on and off spikes.
    #plot_onoff_spk(on_spks,off_spks,w,num_steps,"Spike encoding through $\delta$-Modulation")

    return spk_data_out, cur_in

def triv_sin_enc(num_steps, plot=False):
    spk_data_out = torch.zeros((num_steps,1,2))
    for i in range(0,num_steps):
        if i % 2 == 0:
            spk_data_out[i,0,0] = 1
            spk_data_out[i,0,1] = 0
        else:
            spk_data_out[i,0,1] = 1
            spk_data_out[i,0,0] = 0

    if plot:
        fig = plt.figure(facecolor='white', figsize=(10, 2))
        ax = fig.add_subplot(111)

        on_spks = spk_data_out[:, 0, 0]
        off_spks = spk_data_out[:, 0, 1]
        splt.raster(on_spks, ax, s=400, marker='|', c="green")
        splt.raster(off_spks, ax, s=400, marker='|', c="red")
        plt.ylabel("Input Spikes")
        ax.set_xlabel("Time steps [$ms$]")
        plt.show()

    return spk_data_out

# def sin_prob(num_steps,period):
#     omega = 2 * torch.pi / period
#
#     spk_data_out = torch.zeros((num_steps, 1, 2))
#     on_spks = []
#     off_spks = []
#     for t_0 in range(0,num_steps):
#         spk_data = int.quad(lambda t: np.cos(omega*t),t_0,t_0+t_0*np.pi/4*period)
#         print(spk_data)
#         if 0.0 >= spk_data[0] and 0.5 < spk_data[0]:
#             on_spks.append(1)
#         else:
#             off_spks.append(0)
#
#         spk_data_out[:, 0, 0] = torch.tensor(on_spks)
#         spk_data_out[:, 0, 1] = torch.tensor(off_spks)
#
#     return spk_data_out

#Plotting the input spikes, coloured red and green.
w = 10
T = 100

cur_in_plot(sin_enc(w,T,0.5,200)[1],sin_enc(w,T,0.5,200)[0],w,200)
# on_spks = sin_prob(200,T)[:,0,0]
# off_spks = sin_prob(200,T)[:,0,1]
# plot_onoff_spk(on_spks,off_spks,w,200,"Spike encoding")





