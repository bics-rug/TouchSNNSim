import os
import logging
import time
import argparse
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from parameters import params
from bicsnn.utils import print_args, set_random_seed, generate_input_stdp_test_poisson_rates
from bicsnn.models import Fusi_syn, ALIF_neuron
from bicsnn.output import plot_output_traces, plot_calcium_traces, output_log_traces

set_random_seed(0)

def fusinet_network(nb_inputs, nb_outputs, params, dataset, s_teach, dt_ms, folder_run, w_init=None, logging=None, suffix=''):
    """
    Compute traces in response to pre post spikes.
    :param nb_inputs: number of pre neurons
    :param nb_outputs: number of post neurons
    :param params: dictionary with parameters to load
    :param dataset: dataset to train
    :param s_teach: teach signal torch.tensor with spikes to teach:
    :param dt_ms: sim dt (in ms)
    :param folder_run: in case of figures/traces
    :param logging: logging handler
    :return:
    """
    batch_size = 1
    data, labels = dataset[:]
    (n_samples, n_time_steps, n_neurons) = data.shape

    # Encapsulate in Dataset generator
    generator_data = DataLoader(dataset, batch_size=batch_size, shuffle=False) #num_workers = 2 but error parallel so 0
    generator_teach = DataLoader(s_teach, batch_size=batch_size, shuffle=False) #num_workers = 2 but error parallel so 0

    # Create Fusi Synapse:
    w_fusi = Fusi_syn(nb_inputs, nb_outputs, imported_params=params)
    w_fusi.initialize(batch_size, probabilities=torch.tensor([0.79, 0.21])) # activation equivalent to the object size -- 0.5, 0.5

    # Create Neurons --> Link the synapse
    net = ALIF_neuron(batch_size, nb_outputs, w_fusi,
                      tau_mem=10, beta_adapt=0, dt=dt_ms, thr_0=1, seed_init=19,
                      output_folder=folder_run, is_adaptive=False, w_teach=3) # 3 5 is the minimum for spike in teach elicit spike in out but do we want that or combine with pre

    s_out = torch.zeros((n_samples, n_time_steps, nb_outputs))
    mem_out = torch.empty((n_samples, n_time_steps, nb_outputs))
    calcium = torch.empty((n_samples, nb_inputs, nb_outputs, n_time_steps))
    w = torch.empty((n_samples, nb_inputs, nb_outputs, n_time_steps))
    w_eff = torch.empty((n_samples, nb_inputs, nb_outputs, n_time_steps))

    # Iterate through the dataset
    for i_sam, ((x_local, y_local), x_teach) in enumerate(zip(generator_data, generator_teach)):
        # Reset calcium trace to remove the dependency of history
        w_fusi.reset_calcium()
        for t in range(n_time_steps):
            # Weight update:
            out = net(x_local[:, t, :], teacher=x_teach[0][:, t, :])
            w_fusi.update(x_local[:, t, :], out, net.state.mem)

            s_out[i_sam, t, :] = out
            mem_out[i_sam, t, :] = net.state.mem
            calcium[i_sam, :, :, t] = w_fusi.x_calcium
            w[i_sam, :, :, t] = w_fusi.w
            w_eff[i_sam, :, :, t] = w_fusi.w_eff
        if logging:
            title, vpost = output_log_traces({'w': w, 's_out': s_out}, params, logging, suffix, index_sample=i_sam)

    # Output dictionary with traces
    out = {'mem_out': mem_out,
           's_out': s_out,
           'calcium': calcium,
           'w': w,
           'w_eff': w_eff
           }
    return out, net, w_fusi

def fausi_network(nb_inputs, nb_outputs, params, s_pre, s_teach, dt_ms, folder_run, w_init=None, protocol='ltd'):
    """
    Compute traces in response to pre post spikes.

    :param nb_inputs: number of pre neurons
    :param nb_outputs: number of post neurons
    :param params: dictionary with BCaLL parameters
    :param s_pre: torch.tensor with spikes pre: (batch_size x n_time_bins x nb_inputs)
    :param s_pos: torch.tensor with spikes post: (batch_size x n_time_bins x nb_outputs)
    :param dt_ms: sim dt (in ms)
    :param protocol: if ltd: t_pre > t_post, if ltp: t_pre > t_post
    :return:
    """

    batch_size = s_pre.shape[0]
    n_time_steps = s_pre.shape[1]

    # Fusi Synapse initialized to fixed value
    w_fusi = Fusi_syn(nb_inputs, nb_outputs, imported_params=params)
    start_w = w_init if w_init is not None else w_fusi.thr_w
    w_fusi.initialize(batch_size)
    w_fusi.reset_weights(start_w)

    # Create Neurons
    net = ALIF_neuron(batch_size,
                      nb_outputs,
                      w_fusi,
                      tau_mem=10,
                      beta_adapt=0,
                      dt=dt_ms,
                      thr_0=1,
                      seed_init=19,
                      output_folder=folder_run,
                      is_adaptive=False,
                      w_teach=5) # 5 is the minimum for spike in teach elicit spike in out but do we want that or combine with pre
    mem_out = []
    s_out = []
    calcium = []
    w = []
    w_eff = []
    #
    for t in range(n_time_steps):
        # Weight update:
        out = net(s_pre[:, t, :], teacher=s_teach[:, t, :])
        w_fusi.update(s_pre[:, t, :], out, net.state.mem)

        s_out.append(out)
        mem_out.append(net.state.mem)
        calcium.append(w_fusi.x_calcium)
        w.append(w_fusi.w)
        w_eff.append(w_fusi.w_eff)

    s_out = torch.stack(s_out, dim=2)
    mem_out = torch.stack(mem_out, dim=2)
    calcium = torch.stack(calcium, dim=3)
    w = torch.stack(w, dim=3)           #[:, :, 0, 0]
    w_eff = torch.stack(w_eff, dim=3)       #[:, :, 0, 0]

    i_s_pos = np.where(out == 1)[1]
    i_s_pre = np.where(s_pre == 1)[1]

    out = {'mem_out': mem_out,
           's_out': s_out,
           'calcium': calcium,
           'w': w,
           'w_eff': w_eff
           }
    return out

def fusi_example(args, v_pre=5, v_teach=5, winit=0, n_trials=100000, suffix="", plot_figs=False):
    """
    Test Fusi learning rule.
    :param args:
    """
    # Prepare folders
    timestr = time.strftime("%Y%m%d-%H%M%S")
    folder_run = Path('../run/fusi_example')
    folder_fig = folder_run.joinpath('fig')
    folder_fig.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=os.path.join(os.path.dirname(__file__), folder_run, 'results'+suffix+'.log'), level=logging.INFO,
                        format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', force=True)
    logging.info("START")

    # Params
    print_args(args.__dict__)
    nb_inputs = 1
    nb_outputs = 1
    stim_length_ms = args.stim_length_ms
    dt_ms = 1

    # Generate Input --> Poisson trains
    n_data_points = args.n_data_points  # Number of different poisson rates
    pre_rate = v_pre
    winit = winit                       # 0.5 is threshold
    start_rate = v_teach
    end_rate = 100
    rates = np.linspace(start_rate, end_rate, n_data_points, dtype=int)
    n_trials_per_pair = n_trials
    # Pre post pairs:
    #pre_post_rate_pairs = [(i, j) for i in rates for j in rates] # all combinations pre/teacher
    pre_post_rate_pairs = [(pre_rate, j) for i in np.arange(n_trials_per_pair) for j in rates] # all combinations pre/teacher
    # Create spike pairs at different fr
    s_pre, s_teach, corr = generate_input_stdp_test_poisson_rates(pre_post_rate_pairs, 1, stim_length_ms=stim_length_ms, dt_ms=1, sign_corr=0)

    # Run network  --> Fusi training synapses
    [output_test, net, wfusi] = fusinet_network(nb_inputs, nb_outputs, params[args.params_type], s_pre, s_teach, dt_ms,
                               folder_run, w_init=winit)

    # Output representation
    n_synapses = torch.numel(output_test['w'][:,:,:,-1])
    p_ltp = 100*int(torch.sum(output_test['w'][:,:,:,-1]>params[args.params_type]['thr_w']))/n_synapses
    p_ltd = 100*int(torch.sum(output_test['w'][:,:,:,-1]<params[args.params_type]['thr_w']))/n_synapses
    v_post = torch.mean(torch.sum(output_test['s_out'], [1, 2]).to(float))/(stim_length_ms/1000)

    log_string = ("************************ \n Params:     v_pre = {}  v_post = {}  v_teach = {}  w_init = {} synapses= {} \n P_LTP= {} \n P_LTD= {} \n **********************************************".format(pre_rate, v_post, start_rate, winit,n_synapses,p_ltp, p_ltd))
    logging.info(log_string)
    print(log_string)

    if plot_figs:
        # Plot Figures
        title = "Figure for {}  v_pre = {}  v_teach = {}  v_out = {}  w_init = {}    P_LTP: {}    P_LTD: {}".format(suffix, pre_rate, start_rate, v_post, winit, p_ltp, p_ltd)
        cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)

 #       fig = plot_learning_traces(output_test, pre_post_rate_pairs, s_pre[:][0], dt_ms, title=title, cmap=cmap)
 #       fig.set_size_inches(20, 10)
 #       fig.savefig(folder_fig.joinpath('fusi_traces_learning{}.pdf'.format(suffix)), format='pdf')

        fig = plot_output_traces(s_pre[:][0], s_teach[:][0], output_test, dt_ms, title=title, cmap=cmap)
        fig.set_size_inches(20, 10)
        fig.savefig(folder_fig.joinpath('fusi_traces_output{}.pdf'.format(suffix)), format='pdf')

        fig = plot_calcium_traces(s_pre[:][0], s_teach[:][0], v_post, output_test, dt_ms, params[args.params_type], title=title, cmap=cmap)
        fig.set_size_inches(20, 10)
        fig.savefig(folder_fig.joinpath('fusi_traces_calcium{}.pdf'.format(suffix)), format='pdf')

    print('End')
    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser('test_fusi')
    # Write input parameters of the script in code:
    parser.add_argument('--params_type', type=str, default='Fusi') # Key of synapse/neuron parameters in params
    parser.add_argument('--plot_percentage_change', action='store_true')
    parser.add_argument('--stim_length_ms', type=float, default=300)
    parser.add_argument('--n_data_points', type=float, default=1)
    args = parser.parse_args()

    fusi_example(args, v_pre=25, v_teach=150, winit=1, n_trials=100, suffix="_traces", plot_figs=True)