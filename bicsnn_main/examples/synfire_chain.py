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

def synchain_network(nb_neurons, nb_layers, params, dataset, dt_ms, folder_run, logging=None, suffix=''):
    """
    Compute traces in response to pre post spikes.
    :param nb_neurons_per_layer: number of pre neurons
    :param params: dictionary with parameters to load
    :param dataset: dataset to train
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

    # Create Neurons --> Link the synapse
    net = ALIF_neuron(batch_size, nb_neurons, w_fusi,
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
    return out, net

def synchain_example(args, suffix=''):
    """
    Test Synchain with time delay.
    :param args:
    """
    # Prepare folders
    timestr = time.strftime("%Y%m%d-%H%M%S")
    folder_run = Path('../run/synchain_example')
    folder_fig = folder_run.joinpath('fig')
    folder_fig.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=os.path.join(os.path.dirname(__file__), folder_run, 'results'+suffix+'.log'), level=logging.INFO,
                        format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', force=True)
    logging.info("START")

    # Params
    print_args(args.__dict__)
    nb_inputs = 1
    stim_length_ms = args.stim_length_ms
    dt_ms = 1

    # Generate Input --> Poisson trains
    n_data_points = args.n_data_points  # Number of different poisson rates
    pre_rate = 50
    start_rate = 50
    end_rate = 100
    rates = np.linspace(start_rate, end_rate, n_data_points, dtype=int)
    n_trials_per_pair = 100
    # Pre post pairs:
    pre_post_rate_pairs = [(pre_rate, j) for i in np.arange(n_trials_per_pair) for j in rates] # all combinations pre/teacher
    # Create spike pairs at different fr
    s_pre, s_teach, corr = generate_input_stdp_test_poisson_rates(pre_post_rate_pairs, 1, stim_length_ms=stim_length_ms, dt_ms=1, sign_corr=0)

    # Run network
    [output_test, net] = synchain_network(nb_inputs, 4, params[args.params_type], s_pre, dt_ms, folder_run)

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

    synchain_example(args, suffix='_synchain_')