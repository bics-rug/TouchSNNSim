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

def plot_spk_cur_mem_spk(in_spk, cur, gain, epsc, mem, spk, thr_line=False, vline=False, title=False, ylim_max1=1.25,
                         ylim_max2=1.25):
    # Generate Plots
    fig, ax = plt.subplots(6, figsize=(12, 8), sharex=True,
                           gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 0.4]})

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
    ax[2].plot(gain.detach().cpu().numpy())
    ax[2].set_ylabel("Gain trace (Fac)")
    if thr_line:
        ax[2].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
    plt.xlabel("Time step")

    # Plot membrane potential
    ax[3].plot(epsc.detach().cpu().numpy())
    ax[3].set_ylabel("EPSC trace")
    if thr_line:
        ax[3].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
    plt.xlabel("Time step")

    # Plot membrane potential
    ax[4].plot(mem.detach().cpu().numpy())
    ax[4].set_ylabel("Membrane Potential ($U_{mem}$)")
    if thr_line:
        ax[4].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
    plt.xlabel("Time step")

    # Plot output spike using spikeplot
    splt.raster(spk, ax[5], s=400, c="black", marker="|")
    if vline:
        ax[5].axvline(x=vline, ymin=0, ymax=6.75, alpha=0.15, linestyle="dashed", c="black", linewidth=2, zorder=0,
                      clip_on=False)
    ax[5].set_ylabel("Output")
    plt.yticks([])

    return

def plot_snn_spikes(spk_in, spk1_rec, num_steps, title):
    # Generate Plots
    fig, ax = plt.subplots(2, figsize=(8, 7), sharex=True,
                           gridspec_kw={'height_ratios': [1, 1]})

    # Plot input spikes
    splt.raster(spk_in[:, 0], ax[0], s=0.03, c="black")
    ax[0].set_ylabel("Input Spikes")
    ax[0].set_title(title)

    # Plot hidden layer spikes
    splt.raster(spk1_rec.reshape(num_steps, -1), ax[1], s=0.05, c="black")
    ax[1].set_ylabel("Hidden Layer")

    return


class Net_TDE(nn.Module):
    #def __init__(self, num_inputs, num_hidden, num_outputs, num_steps, beta, weight):
    def __init__(self, num_steps):
        super().__init__()

        # Initialize layers
        self.fc1 = Weight_synapse(2, 2, bias=False)
        self.fc1.initialize(1, type='eye_conn')
        self.tde1 = TDE_neuron(1)
        #self.fc2 = Weight_synapse(num_hidden, num_outputs, bias=False)  # , parameter_optimize=False)
        #self.fc2.initialize(weight[1], type='bin_random_prob')
        #self.lif2 = snn.Leaky(beta=beta[1])
        self.num_steps = num_steps

    def forward(self, x):
        # Record the final layer
        spk1_rec = []
        gain_rec = []
        epsc_rec = []
        mem1_rec = []
        cur1_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x[step])
            spk1, state = self.tde1(cur1)
            cur1_rec.append(cur1)
            spk1_rec.append(spk1)
            gain_rec.append(state.gain)
            epsc_rec.append(state.epsc)
            mem1_rec.append(state.mem)

        return ({'cur': [torch.stack(cur1_rec, dim=0)],
                 'spks': [torch.stack(spk1_rec, dim=0)],
                 'gain': [torch.stack(gain_rec, dim=0)],
                 'epsc': [torch.stack(epsc_rec, dim=0)],
                 'mem': [torch.stack(mem1_rec, dim=0)]
                 })


def net_tde_simulation():
    # Parameters network
    num_steps = 200  # t steps
    batch_size = 1

    # layer parameters
    spike_prob = 0.3

    net = Net_TDE(num_steps)
    # Input
    spk_in = spikegen.rate_conv(torch.rand((num_steps, 2)) * spike_prob).unsqueeze(1)

    traces_rec = net(spk_in.view(num_steps, batch_size, -1))

    plot_snn_spikes(spk_in, traces_rec['spks'][0], num_steps,
                    "Fully Connected Spiking Neural Network")
    # splt.traces(traces_rec['mem'][0].reshape(num_steps, -1), dim=[10,1])
    plot_spk_cur_mem_spk(spk_in[:, 0, :].reshape(num_steps, -1), traces_rec['cur'][0][:, 0, 0].reshape(num_steps, -1),
                         traces_rec['gain'][0][..., 0].reshape(num_steps, -1),
                         traces_rec['epsc'][0][..., 0].reshape(num_steps, -1),
                         traces_rec['mem'][0][..., 0].reshape(num_steps, -1),
                         traces_rec['spks'][0][..., 0].reshape(num_steps, -1), ylim_max1=1.25, ylim_max2=1.25)
    plt.show()


if __name__ == '__main__':
    net_tde_simulation()

