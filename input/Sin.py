import snntorch as snn
import torch
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from snntorch import spikegen

#Environment parameters.
step_num = 100
step_size = 1

#Plotting functions.
def cur_in_plot(cur_in, w, title=False):
    if title:
        plt.title(title)
    plt.plot(cur_in)
    plt.xlabel("Time steps [$ms$]")
    plt.ylabel("Current Input ($I_{in}$) [$mA$]")
    plt.xlim([0, step_num])
    plt.ylim([-w, w])
    plt.show()
def plot_mem(mem, w, title=False):
  if title:
    plt.title(title)
  plt.plot(mem)
  plt.xlabel("Time step")
  plt.ylabel("Membrane Potential")
  plt.xlim([0, step_num])
  plt.ylim([0, w])
  plt.show()

def plot_spk(spk, w,title=False):
    fig = plt.figure(facecolor='white', figsize=(10, 2))
    ax = fig.add_subplot(111)
    if title:
        plt.title(title)
    splt.raster(spk,ax,s=400,marker='|',color="black")
    plt.xlabel("Time step")
    plt.ylabel("")
    plt.xlim([0, step_num])
    plt.ylim([0, w])
    plt.show()

#Define a figure that plots the on spikes and off spikes in two separate plots on top of eachother.
def plot_onoff_spk(spk_on,spk_off, w, title=False):
    fig,ax = plt.subplots(2,1,facecolor='white', figsize=(15, 7), sharex=True)
    if title:
        fig.suptitle(title,fontsize=30)
    splt.raster(spk_off,ax[0],s=400,marker='|',color="red")
    splt.raster(spk_on,ax[1],s=400,marker='|',color="green")
    plt.xlabel("Time step [$ms$]",fontsize=15)
    plt.ylabel("")
    ax[0].set_xlim([0, step_num])
    ax[0].set_ylim([-w, w])
    ax[1].set_xlim([0, step_num])
    ax[1].set_ylim([-w, w])
    plt.show()

#%%
def sin_enc(w, period, threshold):
    #Define the input stimulus.
    omega = 2*torch.pi/period
    sin_in = torch.sin(omega*torch.arange(0,step_num+1,step_size))
    cur_in = w*sin_in

    #Plot the input stimulus.
    cur_in_plot(cur_in,w,"Input Stimulus")

    #Spike data delta-mod.
    spk_data = spikegen.delta(cur_in,threshold=threshold,off_spike=True)
    on_spks = torch.Tensor([0 if i == -1 else i for i in spk_data])
    off_spks = torch.Tensor([0 if i == 1 else i for i in spk_data])

    #Plot the on and off spikes.
    plot_onoff_spk(on_spks,off_spks,w,"Spike encoding through $\delta$-Modulation")

    return spk_data

sin_enc(10, 10, 5)








