import torch
import torch.nn as nn
import snntorch as snn
from bicsnn.models import *
from snntorch import spikeplot as splt
from snntorch import spikegen
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def check_cuda_avail():
    print(torch.cuda.is_available())
    print(torch.cuda.is_available())

    print(torch.cuda.device_count())
    print(torch.cuda.current_device())

    print(torch.cuda.device(0))
    print(torch.cuda.get_device_name(0))

    return

def plot_spk_cur_mem_spk(in_spk, cur, mem, spk, thr_line=False, vline=False, title=False, ylim_max1=1.25, ylim_max2=1.25):
  # Generate Plots
    fig, ax = plt.subplots(4, figsize=(12,8), sharex=True,
                        gridspec_kw = {'height_ratios': [1, 1, 1, 0.4]})

    # Plot output spike using spikeplot
    splt.raster(in_spk, ax[0], s=400, c="black", marker="|")
    if vline:
        ax[0].axvline(x=vline, ymin=0, ymax=6.75, alpha=0.15, linestyle="dashed", c="black", linewidth=2, zorder=0,
                      clip_on=False)
    ax[0].set_ylabel("Input spikes")

    # Plot input current
    ax[1].plot(cur.detach().cpu().numpy(), c="tab:orange")
    ax[1].set_ylim([0, ylim_max1])
    ax[1].set_xlim([0, 200])
    ax[1].set_ylabel("Input Current ($I_{in}$)")
    if title:
        ax[1].set_title(title)

    # Plot membrane potential
    ax[2].plot(mem.detach().cpu().numpy())
    ax[2].set_ylim([0, ylim_max2])
    ax[2].set_ylabel("Membrane Potential ($U_{mem}$)")
    if thr_line:
        ax[2].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
    plt.xlabel("Time step")

    # Plot output spike using spikeplot
    splt.raster(spk, ax[3], s=400, c="black", marker="|")
    if vline:
        ax[3].axvline(x=vline, ymin=0, ymax=6.75, alpha=0.15, linestyle="dashed", c="black", linewidth=2, zorder=0, clip_on=False)
    ax[3].set_ylabel("Output")
    plt.yticks([])

    return

def plot_snn_spikes(spk_in, spk1_rec, spk2_rec, num_steps, title):
    # Generate Plots
    fig, ax = plt.subplots(3, figsize=(8,7), sharex=True,
                        gridspec_kw = {'height_ratios': [1, 1, 0.4]})

    # Plot input spikes
    splt.raster(spk_in[:,0], ax[0], s=0.03, c="black")
    ax[0].set_ylabel("Input Spikes")
    ax[0].set_title(title)

    # Plot hidden layer spikes
    splt.raster(spk1_rec.reshape(num_steps, -1), ax[1], s = 0.05, c="black")
    ax[1].set_ylabel("Hidden Layer")

    # Plot output spikes
    splt.raster(spk2_rec.reshape(num_steps, -1), ax[2], c="black", marker="|")
    ax[2].set_ylabel("Output Spikes")
    ax[2].set_ylim([0, 10])

    return


class Net_LIF(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, num_steps, beta, weight):
        super().__init__()

        # Initialize layers
        self.fc1 = Weight_synapse(num_inputs, num_hidden, bias=False)
        self.fc1.initialize(weight[0], type='bin_random_prob', probabilities=torch.tensor([0.2, 0.8]))
        self.lif1 = snn.Leaky(beta=beta[0])
        self.fc2 = Weight_synapse(num_hidden, num_outputs, bias=False)#, parameter_optimize=False)
        self.fc2.initialize(weight[1], type='bin_random_prob')
        self.lif2 = snn.Leaky(beta=beta[1])
        self.num_steps = num_steps

    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk1_rec = []
        spk2_rec = []
        mem1_rec = []
        mem2_rec = []
        cur1_rec = []
        cur2_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur1_rec.append(cur1)
            cur2_rec.append(cur2)
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            mem1_rec.append(mem1)
            mem2_rec.append(mem2)

        return ({'cur': [torch.stack(cur1_rec, dim=0), torch.stack(cur2_rec, dim=0)],
                'spks': [torch.stack(spk1_rec, dim=0), torch.stack(spk2_rec, dim=0)],
                'mem': [torch.stack(mem1_rec, dim=0), torch.stack(mem2_rec, dim=0)]})

def net_lif_simulation():

    # Parameters network
    num_steps = 200     # t steps
    batch_size = 1

    # layer parameters
    num_inputs = 10
    num_hidden = 30
    num_outputs = 10
    beta = [0.99, 0.99] # It can be multivalue for individual neurons in layer
    weight = [0.2, 0.3]    # Weight value
    spike_prob = 0.1
    
    net = Net_LIF(num_inputs, num_hidden, num_outputs, num_steps, beta, weight)
    # Input
    spk_in = spikegen.rate_conv(torch.rand((num_steps, num_inputs)) * spike_prob).unsqueeze(1)
    # record outputs
    mem2_rec = []
    spk1_rec = []
    spk2_rec = []

    traces_rec = net(spk_in.view(num_steps, batch_size, -1))

    plot_snn_spikes(spk_in, traces_rec['spks'][0], traces_rec['spks'][1], num_steps, "Fully Connected Spiking Neural Network")
    #splt.traces(traces_rec['mem'][0].reshape(num_steps, -1), dim=[10,1])
    plot_spk_cur_mem_spk(spk_in[:,0,:].reshape(num_steps, -1), traces_rec['cur'][0][:,0,0].reshape(num_steps, -1), traces_rec['mem'][0][:,0,0].reshape(num_steps, -1), traces_rec['spks'][0][:,0,0].reshape(num_steps, -1), ylim_max1=1.25, ylim_max2=1.25)
    plt.show()

if __name__ == '__main__':
    net_lif_simulation()

