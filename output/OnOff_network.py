import numpy as np
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
    ax[4].set_ylabel("$U_{mem}$")
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
    splt.raster(spk_in[:, 0], ax[0], s=10, marker='|', c="black")
    ax[0].set_ylabel("Input Spikes")
    ax[0].set_title(title)

    splt.raster(lif_spk.reshape(num_steps, -1), ax[1], s=10, marker='|', c="black")
    ax[1].set_ylabel("LIF")

    splt.raster(tde_spk.reshape(num_steps, -1), ax[2], s=10, marker='|', c="black")
    ax[2].set_ylabel("TDE")

    return

def plot_freq_mempot(func_in, freqs, net_output, num_steps, batch_size, thr, w, title):
    fig, ax = plt.subplots(6, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1, 0.4, 1, 0.4, 1, 0.4]})

    # Plot input function along with the membrane potential.
    in1 = func_in(period = 1/freqs[0],num_steps = num_steps, threshold = thr, w = w)
    ax[0].plot(in1[1])
    ax[0].set_xlim([0, num_steps])
    ax[0].set_ylim([-w, w])
    ax[0].set_ylabel(f"$I_{{in}}$ [mA] ({freqs[0]} kHz)")
    tde_spk1 = net_output(in1[0],num_steps,batch_size)['tde']['spks'][0]
    tde_spk1_fmt = tde_spk1[...,0].reshape(num_steps,-1)
    splt.raster(tde_spk1_fmt, ax[1], s=400, c="black", marker="|")
    ax[1].set_yticks([])

    in2 = func_in(period = 1/freqs[1],num_steps = num_steps, threshold = thr, w = w)
    ax[2].plot(in2[1])
    ax[2].set_xlim([0, num_steps])
    ax[2].set_ylim([-w, w])
    ax[2].set_ylabel(f"$I_{{in}}$ [mA] ({freqs[1]} kHz)")
    tde_spk2 = net_output(in2[0],num_steps,batch_size)['tde']['spks'][0]
    tde_spk2_fmt = tde_spk2[...,0].reshape(num_steps,-1)
    splt.raster(tde_spk2_fmt, ax[3], s=400, c="black", marker="|")
    ax[3].set_yticks([])

    in3 = func_in(period = 1/freqs[2],num_steps = num_steps, threshold = thr, w = w)
    ax[4].plot(in3[1])
    ax[4].set_xlim([0, num_steps])
    ax[4].set_ylim([-w, w])
    ax[4].set_ylabel(f"$I_{{in}}$ [mA] ({freqs[2]} kHz)")
    tde_spk3 = net_output(in3[0],num_steps,batch_size)['tde']['spks'][0]
    tde_spk3_fmt = tde_spk3[...,0].reshape(num_steps,-1)
    splt.raster(tde_spk3_fmt, ax[5], s=400, c="black", marker="|")
    ax[5].set_yticks([])

    fig.suptitle(title, fontsize=20)
    plt.show()

def plot_spkr_inr(neuron, func_in, net_output, avg_spk_rate, num_steps, batch_size, thr, w, title):
    inr_lst = []
    spkr_lst = []
    for f in np.linspace(0.5,1,50):
        inr_lst.append(f)
        input = func_in(period = 1/f,num_steps = num_steps, threshold = thr, w = w)
        output = net_output(input[0],num_steps,batch_size)
        spks = output[neuron]['spks'][0]
        avg_spkr, avg_spkr_err = avg_spk_rate(spks, num_steps, 1/f)
        spkr_lst.append([avg_spkr, avg_spkr_err])
        npspkr_lst = np.array(spkr_lst)

    poly_fit = np.polyfit(inr_lst,npspkr_lst[:,0],3)

    plt.errorbar(inr_lst,npspkr_lst[:,0],npspkr_lst[:,1],c='black',marker='o',ls='none')
    plt.plot(inr_lst,np.polyval(poly_fit,inr_lst),c='red',label='Polynomial Fit')
    plt.xlabel("$f_{in}$ [kHz]")
    plt.ylabel("Average Spiking Rate [kHz]")
    plt.title(title)
    plt.show()

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

# def net_lif_simulation():
#     # Parameters network
#     num_steps = 200  # t steps
#     batch_size = 1
#
#     # layer parameters
#     spike_prob = 0.3
#
#     net = Net_OnOff(num_steps)
#     # Input
#     spk_in = spikegen.rate_conv(torch.rand((num_steps, 2)) * spike_prob).unsqueeze(1)
#
#     traces_rec = net(spk_in.view(num_steps, batch_size, -1))
#
#     plot_snn_spikes(spk_in, traces_rec['lif']['spks'][0], traces_rec['tde']['spks'][0], num_steps,
#                     "In(ON-OFF) --> LIF (with inh) --> TDE")
#     # splt.traces(traces_rec['mem'][0].reshape(num_steps, -1), dim=[10,1])
#
#     plot_spk_cur_mem_spk(spk_in[:, 0, :].reshape(num_steps, -1), traces_rec['lif']['spks'][0].reshape(num_steps, -1),
#                          traces_rec['tde']['gain'][0][..., 0].reshape(num_steps, -1),
#                          traces_rec['tde']['epsc'][0][..., 0].reshape(num_steps, -1),
#                          traces_rec['tde']['mem'][0][..., 0].reshape(num_steps, -1),
#                          traces_rec['tde']['spks'][0][..., 0].reshape(num_steps, -1), ylim_max1=1.25, ylim_max2=1.25)
#     plt.show()

def net_output(input, num_steps, batch_size):
    model = Net_OnOff(num_steps)
    print(input.view(num_steps, batch_size, -1))
    output = model(input.view(num_steps, batch_size, -1))
    return output

def avg_spk_rate(spks, num_steps, period):
   #compute the spiking rate for every 10 timesteps
   lst_spkr = []
   for t in range(0,num_steps, int(period)):
       spkr = np.count_nonzero(spks[t:t+int(period)])/period
       lst_spkr.append(spkr)
   avg_spkr = np.average(np.array(lst_spkr))
   avg_spkr_err = np.std(np.array(lst_spkr))/np.sqrt(len(lst_spkr))
   return avg_spkr, avg_spkr_err

def sin_in_simulation():
    # Network parameters
    num_steps = 200  # t steps
    w = 10 # Amplitude
    T = 10 # Period
    thr = 1 # Threshold
    batch_size = 1

    # Model instantiation
    output = net_output(sin_enc(w,T,thr,num_steps)[0], num_steps, batch_size)

    # Data extraction
    lif_spks = output['lif']['spks'][0]
    tde_spks = output['tde']['spks'][0]

    # Plotting
    plot_snn_spikes(sin_enc(w,T,thr,num_steps)[0], lif_spks, tde_spks, num_steps, "In(ON-OFF) --> LIF (with inh) --> TDE")
    plot_spk_cur_mem_spk(sin_enc(w,T,thr,num_steps)[0].reshape(num_steps,-1),
                            lif_spks.reshape(num_steps,-1),
                            output['tde']['gain'][0][...,0].reshape(num_steps,-1),
                            output['tde']['epsc'][0][...,0].reshape(num_steps,-1),
                            output['tde']['mem'][0][...,0].reshape(num_steps,-1),
                            tde_spks[...,0].reshape(num_steps,-1), ylim_max1=1.25, ylim_max2=1.25)
    plot_freq_mempot(sin_enc, [0.75, 0.85, 0.95], net_output, num_steps, batch_size, thr, w, "Sinuoidal Frequency Modulated Response of TDE Neuron")
    plot_spkr_inr('lif', sin_enc, net_output, avg_spk_rate, num_steps, batch_size, thr, w, "Average Spiking Rate of LIF Neuron as a function of $f_{in}$")


if __name__ == '__main__':
    #sin_in_simulation()
    net_output(sin_enc(10, 10, 1, 200)[0], 200, 1)


