import torch
import torch.nn as nn
import snntorch as snn
from bicsnn.models import *
from input.Sin import *
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

def plot_spk_cur_mem_spk(in_data, in_tde, gain, epsc, mem, spk, thr_line=False, vline=False, title=False, ylim_max1=1.25,
                         ylim_max2=1.25):
    # Generate Plots
    fig, ax = plt.subplots(6, figsize=(12, 8), sharex=True,
                           gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 0.4]})

    # Plot output spike using spikeplot
    splt.raster(in_data, ax[0], s=400, c="black", marker="|")
    if vline:
        ax[0].axvline(x=vline, ymin=0, ymax=6.75, alpha=0.15, linestyle="dashed", c="black", linewidth=2, zorder=0,
                      clip_on=False)
    ax[0].set_ylabel("Input spikes")

    # Plot output spike using spikeplot
    splt.raster(in_tde, ax[1], s=400, c="black", marker="|")
    if vline:
        ax[1].axvline(x=vline, ymin=0, ymax=6.75, alpha=0.15, linestyle="dashed", c="black", linewidth=2, zorder=0,
                      clip_on=False)
    ax[1].set_ylabel("Output LIF")


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

def plot_snn_spikes(spk_in, lif_spk, tde_spk, num_steps, title):
    # Generate Plots
    fig, ax = plt.subplots(3, figsize=(8, 7), sharex=True,
                           gridspec_kw={'height_ratios': [1, 1, 1]})

    # Plot input spikes
    splt.raster(spk_in[:, 0], ax[0], s=10, c="black")
    ax[0].set_ylabel("Input Spikes")
    ax[0].set_title(title)

    splt.raster(lif_spk.reshape(num_steps, -1), ax[1], s=10, c="black")
    ax[1].set_ylabel("LIF")

    splt.raster(tde_spk.reshape(num_steps, -1), ax[2], s=10, c="black")
    ax[2].set_ylabel("TDE")

    return


class Net_OnOff(nn.Module):
    #def __init__(self, num_inputs, num_hidden, num_outputs, num_steps, beta, weight):
    def __init__(self, num_steps):
        super().__init__()

        # Initialize layers
        self.fc1 = Weight_synapse(2, 2, bias=False)
        self.fc1.initialize(1, type='eye_conn')
        self.lif = snn.Leaky(beta=0.98, output=True)
        self.inh = Weight_synapse(2, 2, bias=False)
        self.inh.initialize(-0.9, type='eye_conn')
        # We can add a synaptic operation here later if we want to modulate input of TDE
        self.tde1 = TDE_neuron(1)
        self.num_steps = num_steps

    def forward(self, x):
        # Record the final layer
        lif_curin_rec = []
        lif_spk_rec = [torch.tensor([[0.0, 0.0]])] # Start in 0 for the self inhibition
        lif_mem_rec = []

        tde_spk_rec = []
        tde_gain_rec = []
        tde_epsc_rec = []
        tde_mem_rec = []

        for step in range(self.num_steps):
            cur_exc = self.fc1(x[step])
            cur_inh = self.inh(lif_spk_rec[-1]) # Inhibit based on output of lif neuron in previous state

            lif_spk, lif_mem = self.lif(cur_exc+cur_inh)

            spk1, state = self.tde1(lif_spk)

            # L1
            lif_curin_rec.append(cur_exc+cur_inh)
            lif_spk_rec.append(lif_spk)
            lif_mem_rec.append(lif_mem)
            # L2
            tde_spk_rec.append(spk1)
            tde_gain_rec.append(state.gain)
            tde_epsc_rec.append(state.epsc)
            tde_mem_rec.append(state.mem)

        return ({'tde':{
                     'spks': [torch.stack(tde_spk_rec, dim=0)],
                     'gain': [torch.stack(tde_gain_rec, dim=0)],
                     'epsc': [torch.stack(tde_epsc_rec, dim=0)],
                     'mem': [torch.stack(tde_mem_rec, dim=0)]},
                 'lif':{
                     'cur':[torch.stack(lif_curin_rec, dim=0)],
                     'mem':[torch.stack(lif_mem_rec, dim=0)],
                     'spks':[torch.stack(lif_spk_rec[1:], dim=0)]}
                 })

def net_lif_simulation():
    # Parameters network
    num_steps = 200  # t steps
    batch_size = 1

    # layer parameters
    spike_prob = 0.3

    net = Net_OnOff(num_steps)
    # Input
    spk_in = spikegen.rate_conv(torch.rand((num_steps, 2)) * spike_prob).unsqueeze(1)

    traces_rec = net(spk_in.view(num_steps, batch_size, -1))

    plot_snn_spikes(spk_in, traces_rec['lif']['spks'][0], traces_rec['tde']['spks'][0], num_steps,
                    "In(ON-OFF) --> LIF (with inh) --> TDE")
    # splt.traces(traces_rec['mem'][0].reshape(num_steps, -1), dim=[10,1])

    plot_spk_cur_mem_spk(spk_in[:, 0, :].reshape(num_steps, -1), traces_rec['lif']['spks'][0].reshape(num_steps, -1),
                         traces_rec['tde']['gain'][0][..., 0].reshape(num_steps, -1),
                         traces_rec['tde']['epsc'][0][..., 0].reshape(num_steps, -1),
                         traces_rec['tde']['mem'][0][..., 0].reshape(num_steps, -1),
                         traces_rec['tde']['spks'][0][..., 0].reshape(num_steps, -1), ylim_max1=1.25, ylim_max2=1.25)
    plt.show()


if __name__ == '__main__':
    net_lif_simulation()

