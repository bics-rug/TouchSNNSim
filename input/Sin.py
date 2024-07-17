import torch
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from snntorch import spikegen

#Plotting functions.
def cur_in_plot(cur_in, w, num_steps,title=False):
    if title:
        plt.title(title)
    plt.plot(cur_in)
    plt.xlabel("Time steps [$ms$]")
    plt.ylabel("Current Input ($I_{in}$) [$mA$]")
    plt.xlim([0, num_steps])
    plt.ylim([-w, w])
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
    spk_data = spikegen.delta(cur_in,threshold=threshold,off_spike=True)
    on_spks = torch.Tensor([0 if i == -1 else 1 for i in spk_data])
    off_spks = torch.Tensor([0 if i == 1 else 1 for i in spk_data])

    # Formatted spike data.
    spk_data_out = torch.zeros((num_steps,1,2))
    spk_data_out[:,0,0] = on_spks
    spk_data_out[:,0,1] = off_spks

    #Plot the on and off spikes.
    #plot_onoff_spk(on_spks,off_spks,w,num_steps,"Spike encoding through $\delta$-Modulation")

    return spk_data_out, cur_in







