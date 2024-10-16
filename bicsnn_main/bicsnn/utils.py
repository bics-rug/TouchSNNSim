import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import correlation_coefficient
import neo
import quantities as pq
from elephant.spike_train_correlation import spike_time_tiling_coefficient
import matplotlib.pyplot as plt
import pickle
import lzma
import scipy.io as sio
from sklearn.model_selection import train_test_split


def predict_labels(output):
    spks_n = torch.sum(output, axis=1)
    _, idx_testing = spks_n.max(1)
    return spks_n, idx_testing

def reassign_labels(labels_train, top_neurons_train, labels_test):
    ''' Reassign the labels value based on the index of the neuron result for unsupervised methods '''
    new_labels_train = torch.empty_like(labels_train)
    new_labels_test = torch.empty_like(labels_test)
    new_vals = []
    value_dict = dict()
    for value in torch.unique(labels_train):
        # Select index of neurons for sample class
        tn_value = top_neurons_train[labels_train == value].clone()
        val, _ = torch.mode(tn_value,0)
        while (val in new_vals) and (val != -1):
            tn_value = tn_value[tn_value != val]
            if tn_value.numel() == 0:
                val = -1
                break
            val, _ = torch.mode(tn_value, 0)
            val = int(val)
        new_labels_train[labels_train == value] = int(val)
        new_labels_test[labels_test == value] = int(val)
        new_vals.append(int(val))
        value_dict.update({value:val})
    return new_labels_train, new_labels_test, value_dict

def convolve_1d(signal, kernel):
    kernel = kernel[::-1]
    return [
        np.dot(
            signal[max(0, i):min(i + len(kernel), len(signal))],
            kernel[max(-i, 0):len(signal) - i * (len(signal) - len(kernel) < i)],
        )
        for i in range(1 - len(kernel), len(signal))
    ]

def kruskal_corr(a, b, dt, plot_fig=False):
    """
    Implement correlation metric of Kruskal et al. (2007).

    https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.2946
    :param a, first spike train (binary array)
    :param b, second spike train (binary array)
    :dt length boxcar filter F.
    """
    T = len(a)
    k = [1] * int(dt)

    # Convolution with box filter
    a_p = convolve_1d(a, k)
    # print('Mean a_p:', np.mean(a_p))
    N_a = np.sum(a)
    # print('a:', a_p)

    b_p = convolve_1d(b, k)
    # print('b', b_p)
    N_b = np.sum(b)

    prod = (a_p - N_a * dt / T) * (b_p - N_b * dt / T)
    cov_ab = (1 / T) * np.sum(prod)

    var_a = (1 / T) * np.sum((a_p - N_a * dt / T) ** 2)
    var_b = (1 / T) * np.sum((b_p - N_b * dt / T) ** 2)

    corr = cov_ab / (np.sqrt(var_a * var_b))
    # print(cov_ab)

    if plot_fig:
        fig, axs = plt.subplots(5, 1, sharex=True)
        axs[0].vlines(np.where(np.array(a) > 0)[0], 0, 1, label='s1')
        axs[0].legend()
        axs[0].set_yticks([])
        axs[1].vlines(np.where(np.array(b) > 0)[0], 0, 1, label='s2')
        axs[1].legend()
        axs[1].set_yticks([])
        axs[2].plot(a_p, '.-', label='s1_p')
        axs[2].plot(b_p, '.-', label='s2_p')
        axs[2].legend()
        axs[3].plot(a_p - N_a * dt / T, '.-', label="S1=s1_p-mean(s1_p)")
        axs[3].plot(b_p - N_b * dt / T, '.-', label="S2=s2_p-mean(s2_p)")
        axs[3].legend()
        axs[4].plot(prod, label='S1*S2')
        axs[4].legend()
        axs[4].set_xlabel('Samples')
        axs[0].set_title(f'corr: {corr}')
    else:
        fig = None

    return corr, fig

def inhomogeneous_poisson(rate_Hz, dt_sec=1e-3):
    """
    Generate one realization of a Poisson spike train.

    :param rate_Hz: Poisson spike rate (in Hz) of dimension 1
    :param dt_sec: time step (in sec)
    :return: spike_times (with time unit corresponding to the time bin size of rate_Hz)
    """
    n_bins = np.shape(rate_Hz)[0]
    probability = rate_Hz * dt_sec
    spikes = np.random.rand(n_bins) < probability
    spike_times = np.nonzero(spikes)[0]
    return spike_times

def inhomogeneous_poisson_trains(rate_Hz, dt_sec=1e-3):
    """
    Generator of Poisson spike trains.
    :param rate_Hz: rate of Poisson across time and across neurons
    :param dt_sec: time window of each bin (in sec)
    :return: spike times of poisson spike trains (in sec) across all elements of rate_Hz
    """
    (n_bins, n_neurons) = np.shape(rate_Hz)
    probability = rate_Hz * dt_sec
    spike_bins = np.zeros(rate_Hz.shape)
    for n in range(n_neurons):
        spk_time = inhomogeneous_poisson(rate_Hz[:,n])
        spike_bins[spk_time, n] = 1
    return spike_bins

def get_rates_across_time(n_input, stim_length_ms, dt_ms, rates=[]):
    """
    Generate matrix with poisson rates across time bins.

    :param n_input: number of input neurons
    :param stim_length_ms: stimulus length in ms
    :param dt_ms: time bin (ms)
    :param rates: list of poisson mean rate for each input unit (n_input) shape (1, n_inputs)

    :return:
        rate_across_time,
        labels
    """
    n_samples = int(stim_length_ms / dt_ms)
    assert (np.shape(rates)[1] == n_input)

    # Create matrix input rates by repeating input rates across all time bins (i.e., constant rate):
    rate_across_time = np.repeat(rates, n_samples, axis=0)  # n_samples x n_input

    return rate_across_time, rates

def inhomogeneous_poisson_generator(n_trials, rate_Hz, dt_sec=1e-3):
    """
    Generator of Poisson spike trains.

    :param n_trials: (ignored if rate_Hz is a list), number of repetitions of poisson rates of value rate_Hz
    :param rate_Hz: if int, rate of Poisson gen, else list of rates
    :param dt_sec: time window of each bin (in sec)
    :return: spike times of poisson spike trains (in sec) across all elements of rate_Hz
    """
    if len(rate_Hz.shape) == 1:  # rates across time of one neuron
        for trial in range(n_trials):
            yield inhomogeneous_poisson(rate_Hz, dt_sec)
    elif len(rate_Hz.shape) == 2:  # rates across time of more neurons, shape (n time bins x n neurons)
        for trial in range(rate_Hz.shape[1]):
            yield inhomogeneous_poisson(rate_Hz[:, trial], dt_sec)

def generate_input_stdp_test_poisson_rates(pre_post_rate_pairs, n_neurons, stim_length_ms=1000, dt_ms=1,
                                           sign_corr=0, metrics_corr=['pcc'], binsize=100 * pq.ms):
    """
    Generate input spike trains for experiment with multiple
    combinations of pre post Poisson rates.

    :param pre_post_rate_pairs: list of pairs of pre post mean poisson rates
    :param n_neurons: number of neurons = number of repetitions of the same poisson rate
    :param stim_length_ms: stimulus length in ms
    :param dt_ms: stimulus and simulation dt (in ms)
    :param pcc: if not 0, correlated spike trains with pearson correlation coefficient pcc are generated using
    the mixture method of "Generation of Correlated Spike Trains"
    (Brette 2009, https://doi.org/10.1162/neco.2009.12-07-657) to generate correlated spike trains.

    :return:
        s_pre, s_post tensors of pre post spike trains,
        corr: tensor of correlations
    """
    n_tot_pairs = len(pre_post_rate_pairs)
    t_ms = np.arange(int(stim_length_ms / dt_ms))
    n_time_bins = len(t_ms)
    s_pre = torch.zeros(n_tot_pairs, n_time_bins, n_neurons)
    s_post = torch.zeros(n_tot_pairs, n_time_bins, n_neurons)
    corr = {}
    for metric in metrics_corr:
        corr[metric] = torch.zeros(n_tot_pairs, n_neurons)
    # corr = None

    # Generate uncorrelated Poisson spike trains
    if sign_corr == 0:
        print(f'Generating uncorrelated Poisson spike trains')
        for trial, pre_post_rate in enumerate(pre_post_rate_pairs):

            # Spike pre:
            rate_pre = pre_post_rate[0]
            rate_per_neuron = np.array([np.ones(n_neurons) * rate_pre])
            rate_across_time_pre, _ = get_rates_across_time(n_neurons, stim_length_ms, dt_ms, rate_per_neuron)
            # Spike post:
            rate_post = pre_post_rate[1]
            rate_per_neuron = np.array([np.ones(n_neurons) * rate_post])
            rate_across_time_post, _ = get_rates_across_time(n_neurons, stim_length_ms, dt_ms, rate_per_neuron)
            for i_nrn, (spk_per_nrn_pre, spk_per_nrn_post) in enumerate(zip(inhomogeneous_poisson_generator(None,
                                                                                                            rate_across_time_pre,
                                                                                                            dt_sec=dt_ms * 1e-3),
                                                                            inhomogeneous_poisson_generator(None,
                                                                                                            rate_across_time_post,
                                                                                                            dt_sec=dt_ms * 1e-3))):
                # Get spike times of poisson realization for all neurons
                s_pre[trial, spk_per_nrn_pre, i_nrn] = 1
                # Get spike times of poisson realization for all neurons
                s_post[trial, spk_per_nrn_post, i_nrn] = 1

                # Add correlation:
                for metric in metrics_corr:
                    corr[metric][trial, i_nrn] = spike_corr(s_pre[trial, :, i_nrn].cpu(),
                                                            s_post[trial, :, i_nrn].cpu(),
                                                            metric, duration_ms=stim_length_ms, binsize=binsize)

    for metric in metrics_corr:
        corr[metric] = torch.nanmean(corr[metric], 1)

    return TensorDataset(s_pre, torch.zeros(s_pre.shape[0])), TensorDataset(s_post), corr

def spike_corr(s1, s2, method, duration_ms=1000, binsize=0.001 * pq.s):
    """
    Wrapper around functions to compute correlation coeff.
    """
    # pcc = np.cov(s1, s2)[0,1]/np.sqrt(np.var(s1)*np.var(s2))

    if method == 'pcc':
        s1 = neo.SpikeTrain(np.where(s1 > 0)[0], units='ms', t_stop=duration_ms)
        s2 = neo.SpikeTrain(np.where(s2 > 0)[0], units='ms', t_stop=duration_ms)
        corr = correlation_coefficient(BinnedSpikeTrain([s1, s2], bin_size=binsize))[0, 1]

    elif method == 'sttc':
        s1 = neo.SpikeTrain(np.where(s1 > 0)[0], units='ms', t_stop=duration_ms)
        s2 = neo.SpikeTrain(np.where(s2 > 0)[0], units='ms', t_stop=duration_ms)
        corr = spike_time_tiling_coefficient(s1, s2, dt=binsize)

    elif method == 'kruskal':
        corr, _ = kruskal_corr(np.array(s1), np.array(s2), dt=binsize)

    return corr

def generate_teaching_signal_dataset(v_teach, length_ms, labels):
    n_output = len(np.unique(labels))
    n_data = len(labels)
    input_freq = np.zeros((n_data, length_ms, n_output))

    # Build the teaching dataset to pair with the labels
    for i_neu, lab in enumerate(np.unique(labels)):
        ind_lab = (labels == lab)
        input_freq[ind_lab, :, i_neu] = v_teach
    # Generate spike trains
    input_teach = np.array([inhomogeneous_poisson_trains(rate, dt_sec=1e-3) for rate in input_freq])

    return TensorDataset(torch.IntTensor(input_teach)), input_teach

# Set random seed for numpy and torch
def set_random_seed(seed, add_generator=False, device=torch.device('cpu')):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if add_generator:
        generator = torch.Generator(device=device).manual_seed(seed)
        return generator
    else:
        return None

def print_args(args):
    """
    Print key and value of dictionary args.
    :param dictionary with params to print
    """
    print('Running experiment with:')
    for k, v in args.items():
        print(k, v)


class EventDataset(Dataset):
    """
    Event Dataset class.

    Used to load items of dataset.
    """

    def __init__(self, data_file: str, device=None):
        """
        :param data_file: Path to h5 file with dataset.
        """
        if type(data_file) == str:
            self.file = h5py.File(data_file, 'r')  # read h5 file if input is path
        else:
            self.file = data_file  # it is already the output of h5 read
        self.device = device

        self.dt = self.file.attrs['dt']
        self.timesteps = np.arange(0, self.file.attrs['time_window'], self.dt)
        self.n_time_bins = len(self.timesteps)
        self.n_neurons = self.file.attrs['n_neurons']

    def __len__(self):
        return len(self.file['labels'])

    def __getitem__(self, idx):
        # t_start = time.time()
        labels = self.file['labels'][idx]
        spikes_per_neuron = self.file['spikes_per_neuron'][idx]

        neuron_ids = np.concatenate([[i] * len(spk_time) for i, spk_time in enumerate(spikes_per_neuron)])
        spike_times = np.concatenate(spikes_per_neuron)

        i = np.array([spike_times, neuron_ids])
        v = np.ones(len(neuron_ids), dtype=int)
        sparse_spikes = torch.sparse_coo_tensor(i, v, (self.n_time_bins, self.n_neurons))

        return {'spikes': sparse_spikes.to(self.device),
                'labels': torch.tensor(labels, device=self.device)}


# ***** TDE example functions to refactor ****** #
Tau_init = [157, 143, 150, 148, 163, 189, 211, 193, 188, 174, 165, 175, 167,        160, 160, 157, 154, 156, 153, 160, 175, 172, 176, 188, 191, 202,        234, 241, 251, 242, 228, 145, 178, 180, 183, 182, 191, 199, 197,        190, 188, 190, 192, 192, 195, 192, 188, 183, 185, 181, 182, 185,        184, 191, 187, 194, 200, 204, 204, 216, 205, 166, 123, 165, 189,        195, 187, 185, 181, 176, 174, 175, 180, 181, 187, 189, 190, 188,        188, 186, 182, 177, 175, 173, 174, 173, 174, 177, 179, 178, 189,        175, 133, 113, 164, 186, 188, 182, 183, 176, 172, 170, 171, 175,        177, 181, 184, 185, 185, 182, 181, 177, 173, 171, 168, 167, 167,        165, 172, 173, 171, 181, 177, 135, 112, 161, 189, 187, 173, 176,        173, 166, 160, 161, 164, 170, 175, 181, 181, 181, 178, 178, 175,        171, 167, 165, 163, 160, 159, 164, 166, 161, 171, 158, 129, 125,        168, 192, 191, 184, 166, 169, 164, 155, 152, 155, 162, 168, 175,        178, 179, 177, 178, 179, 178, 176, 173, 172, 169, 166, 165, 166,        163, 180, 146, 131, 145, 183, 199, 201, 201, 178, 168, 169, 165,        164, 168, 175, 181, 186, 190, 189, 185, 186, 187, 188, 189, 186,        187, 185, 181, 179, 180, 175, 175, 158, 152, 160, 185, 197, 198,        205, 186, 175, 169, 166, 176, 177, 180, 186, 192, 194, 194, 193,        191, 192, 192, 191, 190, 192, 189, 186, 185, 189, 182, 183, 164,        168, 147, 186, 196, 197, 198, 183, 179, 171, 165, 171, 175, 178,        183, 187, 192, 190, 188, 191, 192, 196, 193, 190, 190, 186, 185,        183, 188, 181, 181, 177, 169, 146, 183, 189, 189, 189, 174, 177,        174, 161, 159, 160, 169, 176, 180, 182, 183, 184, 185, 188, 190,        188, 191, 188, 190, 185, 183, 187, 174, 191, 180, 161, 145, 181,        188, 186, 182, 168, 172, 173, 166, 156, 152, 160, 165, 170, 173,        175, 177, 182, 184, 186, 186, 190, 193, 193, 193, 187, 187, 181,        186, 171, 155, 137, 182, 190, 189, 186, 168, 170, 175, 168, 162,        151, 151, 160, 165, 172, 174, 177, 181, 184, 187, 186, 188, 194,        194, 193, 191, 186, 180, 190, 180, 149, 142, 180, 192, 194, 191,        177, 181, 183, 175, 170, 163, 158, 162, 167, 176, 178, 181, 183,        189, 191, 190, 194, 199, 198, 199, 195, 192, 182, 179, 176, 153,        127, 182, 192, 198, 195, 183, 184, 185, 182, 175, 170, 167, 166,        172, 178, 182, 186, 187, 193, 193, 194, 195, 196, 197, 196, 198,        195, 182, 182, 192, 152, 151, 184, 193, 197, 197, 186, 187, 186,        183, 176, 172, 170, 174, 169, 177, 182, 189, 194, 196, 199, 197,        196, 196, 196, 194, 194, 194, 189, 184, 178, 154, 151, 182, 191,        194, 194, 186, 185, 184, 182, 176, 171, 168, 172, 173, 171, 178,        185, 193, 193, 196, 197, 198, 194, 192, 190, 188, 190, 193, 187,        175, 154, 134, 175, 187, 189, 190, 180, 183, 184, 180, 173, 168,        165, 168, 174, 176, 174, 178, 185, 189, 192, 194, 195, 191, 188,        185, 183, 189, 194, 198, 179, 150, 122, 174, 187, 190, 188, 176,        182, 182, 179, 174, 169, 168, 171, 173, 176, 176, 174, 178, 184,        189, 191, 187, 184, 182, 179, 183, 186, 183, 195, 178, 144, 127,        171, 191, 192, 192, 180, 180, 183, 179, 175, 173, 173, 175, 181,        182, 182, 181, 177, 181, 185, 185, 184, 182, 180, 178, 181, 189,        178, 176, 172, 139, 123, 173, 194, 194, 193, 182, 180, 182, 183,        177, 177, 178, 180, 184, 185, 188, 185, 183, 181, 179, 179, 181,        180, 179, 177, 184, 184, 170, 179, 175, 138, 129, 174, 195, 197,        195, 183, 181, 184, 184, 181, 183, 183, 185, 189, 190, 190, 186,        185, 184, 179, 176, 178, 181, 176, 177, 181, 181, 172, 173, 160,        136, 134, 176, 195, 196, 198, 188, 181, 185, 184, 185, 189, 188,        190, 193, 193, 192, 190, 186, 188, 183, 178, 171, 176, 175, 177,        177, 179, 169, 173, 160, 134, 142, 180, 194, 195, 195, 187, 184,        187, 187, 188, 192, 196, 198, 199, 197, 196, 191, 188, 187, 185,        183, 178, 169, 169, 170, 175, 177, 171, 168, 172, 150, 158, 179,        188, 189, 189, 185, 187, 190, 190, 188, 191, 196, 199, 199, 199,        197, 192, 185, 182, 182, 180, 178, 170, 161, 162, 170, 179, 175,        172, 140, 152, 157, 182, 188, 185, 188, 184, 187, 191, 191, 190,        194, 200, 203, 202, 203, 198, 193, 186, 182, 181, 181, 178, 177,        165, 157, 161, 171, 168, 175, 147, 161, 160, 184, 186, 183, 184,        180, 188, 188, 192, 193, 194, 198, 198, 202, 202, 200, 195, 190,        185, 182, 182, 182, 177, 170, 164, 158, 169, 171, 173, 145, 174,        170, 187, 186, 184, 184, 181, 189, 194, 197, 197, 198, 203, 203,        201, 203, 202, 201, 195, 189, 186, 182, 182, 180, 176, 173, 164,        167, 159, 155, 126, 185, 183, 194, 188, 186, 185, 182, 189, 198,        206, 201, 210, 206, 204, 204, 206, 207, 202, 201, 194, 190, 183,        182, 180, 180, 181, 173, 171, 158, 151, 143, 166, 211, 194, 186,        183, 179, 179, 195, 207, 209, 206, 209, 202, 204, 201, 203, 202,        209, 204, 195, 187, 183, 180, 175, 180, 188, 192, 187, 187, 181,        165, 190, 214, 197, 187, 181, 183, 183, 201, 208, 212, 216, 212,        213, 214, 210, 209, 212, 200, 202, 196, 189, 175, 181, 179, 176,        187, 192, 203, 203, 203, 144, 176, 215, 184, 173, 174, 180, 186,        202, 208, 220, 215, 210, 217, 215, 233, 232, 241, 217, 209, 197,        197, 191, 175, 152, 175, 171, 191, 211, 197, 227, 244, 224, 186,        187, 159, 164, 165, 167, 187, 191, 189, 188, 183, 180, 179, 176,        176, 175, 175, 175, 173, 171, 173, 177, 187, 194, 201, 200, 195,        217, 214, 234, 199]
i_ch = [196,160,347,191,346,316,165,750,378,719,749,195,315,377,197,563,320,501,383,532,718,755,409,351,164,471,502,470,780,440,439,781,688,687,594,786,724,208,352,178,209,533,564,476,177,166,227,285,408,179,169,170,210,581,488,171,363,843,812,512,731,414,511,538,382,140,172,595,693,550,811,507,657,223,760,200,364,445,167,289,129,656,348,544,168,625,173,763,480,198,518,513,175,180,138,519,174,128,500,600,448,756,415,842,817,576,569,754,139,762,605,477,873,457,725,758,481,395,761,543,593,574,759,365,202,631,284,729,736,580,551,612,698,643,362,545,697,626,353,515,203,418,967,207,549,483,417,321,694,572,385,204,608,604,355,637,450,562,516,254,787,767,730,484,509,669,426,848,791,757,548,176,211,446,662,520,699,728,799,479,222,573,904,514,517,506,547,821,449,575,531,611,521,142,387,396,579,905,794,487,134,732,700,674,201,161,456,668,394,108,141,582,107,356,610,636,541,642,475,667,542,486,421,764,228,379,358,386,384,354,361,199,539,874,469,425,389,727,666,723,427,570,540,670,792,606,635,609,163,467,419,508,333,790,482,510,388,393,789,485,577,639,632,390,935,788,673,785,966,331,726,205,451,447,768,76,424,735,972,458,733,705,360,978,599,357,192,359,797,793,78,137,181,638,489,35,892,854,478,537,468,607,552,452,416,529,4,798,852,499,818,701,702,696,413,692,159,633,641,490,795,734,498,830,332,644,546,936,704,444,765,350,66,288,583,675,824,258,578,646,910,661,737,453,849,766,822,109,103,829,853,523,980,706,823,977,634,322,136,800,397,97,253,914,438,695,77,372,672,36,49,571,324,213,640,748,522,302,143,820,135,859,392,194,156,831,72,455,212,184,630,491,334,407,568,186,588,890,655,826,420,527,603,366,146,589,147,400,530,981,663,976,5,441,214,585,22,624,613,671,617,54,327,183,41,325,405,495,45,717,557,861,428,828,399,601,602,647,856,492,185,974,206,472,886,398,979,226,860,665,10,847,466,155,423,370,921,922,437,369,561,947,614,497,345,918,145,652,816,133,193,615,879,587,391,891,650,560,710,367,679,885,917,819,915,740,15,738,741,326,825,106,971,916,782,774,23,368,827,111,770,371,553,524,376,290,16,986,858,323,565,888,454,923,317,850,948,555,796,28,47,53,833,769,882,182,887,436,945,883,803,187,67,48,746,713,598,503,73,950,422,779,494,75,893,851,240,349,410,586,621,216,460,805,620,432,328,682,232,148,801,924,42,919,342,343,231,233,616,703,884,648,229,74,834,953,952,104,435,590,330,558,911,381,17,79,619,959,775,215,117,771,664,296,681,714,127,973,525,110,708,341,464,18,880,909,459,554,496,112,855,857,157,11,13,591,319,493,751,9,773,84,434,144,528,985,85,954,403,742,402,20,680,335,431,291,26,462,941,835,944,81,772,720,14,743,21,649,739,678,60,982,406,913,881,804,592,686,337,190,777,336,46,295,25,43,983,711,52,928,806,951,463,645,259,685,556,802,623,618,975,949,984,946,622,942,651,715,813,747,526,677,44,12,301,429,534,98,461,57,872,837,584,684,310,865,241,3,40,162,707,894,430,776,920,889,329,293,832,30,27,868,745,653,116,867,927,239,744,149,691,676,899,559,34,841,126,536,926,24,912,115,709,465,375,689,0,260,19,59,862,55,900,297,105,374,235,838,373,897,401,153,956,896,722,58,940,809,987,878,863,930,869,443,903,683,266,152,56,990,311,65,810,31,292,338,404,654,380,961,925,784,505,808,567,807,50,958,932,303,61,753,62,80,51,96,846,474,339,836,943,906,314,716,752,113,866,712,433,279,257,312,304,29,294,150,298,264,245,95,121,83,535,895,272,89,566,32,596,931,939,261,929,318,991,158,844,86,864,122,412,271,1,238,955,340,305,276,124,901,839,344,262,88,300,870,234,876,125,39,7,230,263,151,963,91,248,988,118,660,6,778,965,989,269,64,87,658,114,93,875,2,119,840,242,286,270,82,265,815,120,37,63,244,898,721,188,934,627,960,473,217,597,504,225,307,299,102,277,902,71,8,90,33,629,690,937,306,132,308,908,273,933,628,313,957,280,243,237,783,877,871,442,256,962,246,268,154,968,275,814,411,218,249,130,845,189,224,236,94,907,970,938,267,281,964,274,92,255,287,247,70,250,123,68,101,38,659,283,99,309,282,131,969,100,219,278,252,221,220,69,251]
f_ch =  [6,5,11,6,11,10,5,24,12,23,24,6,10,12,6,18,10,16,12,17,23,24,13,11,5,15,16,15,25,14,14,25,22,22,19,25,23,6,11,5,6,17,18,15,5,5,7,9,13,5,5,5,6,18,15,5,11,27,26,16,23,13,16,17,12,4,5,19,22,17,26,16,21,7,24,6,11,14,5,9,4,21,11,17,5,20,5,24,15,6,16,16,5,5,4,16,5,4,16,19,14,24,13,27,26,18,18,24,4,24,19,15,28,14,23,24,15,12,24,17,19,18,24,11,6,20,9,23,23,18,17,19,22,20,11,17,22,20,11,16,6,13,31,6,17,15,13,10,22,18,12,6,19,19,11,20,14,18,16,8,25,24,23,15,16,21,13,27,25,24,17,5,6,14,21,16,22,23,25,15,7,18,29,16,16,16,17,26,14,18,17,19,16,4,12,12,18,29,25,15,4,23,22,21,6,5,14,21,12,3,4,18,3,11,19,20,17,20,15,21,17,15,13,24,7,12,11,12,12,11,11,6,17,28,15,13,12,23,21,23,13,18,17,21,25,19,20,19,5,15,13,16,10,25,15,16,12,12,25,15,18,20,20,12,30,25,21,25,31,10,23,6,14,14,24,2,13,23,31,14,23,22,11,31,19,11,6,11,25,25,2,4,5,20,15,1,28,27,15,17,15,19,17,14,13,17,0,25,27,16,26,22,22,22,13,22,5,20,20,15,25,23,16,26,10,20,17,30,22,14,24,11,2,9,18,21,26,8,18,20,29,21,23,14,27,24,26,3,3,26,27,16,31,22,26,31,20,10,4,25,12,3,8,29,14,22,2,12,21,1,1,18,10,6,20,24,16,9,4,26,4,27,12,6,5,26,2,14,6,5,20,15,10,13,18,6,18,28,21,26,13,17,19,11,4,19,4,12,17,31,21,31,0,14,6,18,0,20,19,21,19,1,10,5,1,10,13,15,1,23,17,27,13,26,12,19,19,20,27,15,5,31,6,15,28,12,31,7,27,21,0,27,15,5,13,11,29,29,14,11,18,30,19,16,11,29,4,21,26,4,6,19,28,18,12,28,20,18,22,11,21,28,29,26,29,23,0,23,23,10,26,3,31,29,25,24,0,11,26,3,24,11,17,16,12,9,0,31,27,10,18,28,14,29,10,27,30,17,25,0,1,1,26,24,28,5,28,14,30,28,25,6,2,1,24,23,19,16,2,30,13,25,15,2,28,27,7,11,13,18,20,6,14,25,20,13,10,22,7,4,25,29,1,29,11,11,7,7,19,22,28,20,7,2,26,30,30,3,14,19,10,18,29,12,0,2,19,30,25,6,3,24,21,9,21,23,4,31,16,3,22,11,14,0,28,29,14,17,16,3,27,27,5,0,0,19,10,15,24,0,24,2,14,4,17,31,2,30,13,23,12,0,21,10,13,9,0,14,30,26,30,2,24,23,0,23,0,20,23,21,1,31,13,29,28,25,19,22,10,6,25,10,1,9,0,1,31,22,1,29,26,30,14,20,8,22,17,25,20,19,31,30,31,30,20,30,21,23,26,24,16,21,1,0,9,13,17,3,14,1,28,27,18,22,10,27,7,0,1,5,22,28,13,25,29,28,10,9,26,0,0,28,24,21,3,27,29,7,24,4,22,21,29,18,1,27,4,17,29,0,29,3,22,15,12,22,0,8,0,1,27,1,29,9,3,12,7,27,12,28,12,4,30,28,23,1,30,26,31,28,27,30,28,14,29,22,8,4,1,31,10,2,26,1,9,10,13,21,12,31,29,25,16,26,18,26,1,30,30,9,1,24,2,2,1,3,27,15,10,26,30,29,10,23,24,3,27,22,13,9,8,10,9,0,9,4,9,8,7,3,3,2,17,28,8,2,18,1,19,30,30,8,29,10,31,5,27,2,27,3,13,8,0,7,30,10,9,8,4,29,27,11,8,2,9,28,7,28,4,1,0,7,8,4,31,2,8,31,3,21,0,25,31,31,8,2,2,21,3,3,28,0,3,27,7,9,8,2,8,26,3,1,2,7,28,23,6,30,20,30,15,7,19,16,7,9,9,3,8,29,2,0,2,1,20,22,30,9,4,9,29,8,30,20,10,30,9,7,7,25,28,28,14,8,31,7,8,4,31,8,26,13,7,8,4,27,6,7,7,3,29,31,30,8,9,31,8,2,8,9,7,2,8,3,2,3,1,21,9,3,9,9,4,31,3,7,8,8,7,7,2,8]
t_ch = [11,6,6,5,5,6,11,6,6,6,5,10,5,5,12,5,11,5,11,5,5,11,6,10,10,6,6,5,5,6,5,6,6,5,5,11,11,23,12,24,24,6,6,11,23,12,11,6,5,25,15,16,25,24,24,17,23,6,6,17,18,11,15,11,10,17,18,6,11,24,5,11,6,6,16,15,24,11,13,11,6,5,7,18,14,5,19,19,16,13,23,18,21,26,15,24,20,5,4,11,15,12,12,5,11,19,11,10,16,18,16,12,5,24,12,14,17,24,17,16,4,16,15,25,17,11,5,16,24,23,25,24,16,24,22,19,15,6,13,20,18,16,6,22,23,19,15,12,12,14,14,19,20,15,15,17,17,4,21,6,12,23,17,20,13,18,24,11,16,13,22,22,26,12,11,25,17,15,24,14,5,15,5,19,22,10,21,15,16,17,4,23,26,19,16,25,22,6,19,23,11,19,18,24,16,7,23,17,23,16,18,25,15,16,22,16,14,23,10,16,15,22,19,20,12,7,18,15,13,14,21,14,12,6,4,23,18,14,15,10,25,12,13,19,17,17,15,21,9,2,17,12,24,15,18,14,17,22,14,21,20,19,12,19,5,13,23,10,5,22,13,20,18,13,25,15,22,22,11,25,20,24,20,17,10,17,7,19,22,18,17,14,27,18,25,5,24,17,13,10,3,18,26,19,14,2,5,23,15,3,12,19,20,14,10,10,4,13,22,26,20,21,2,24,23,25,20,6,23,10,21,9,5,10,26,25,18,11,21,27,11,10,25,20,12,22,16,17,11,23,16,28,19,25,17,16,14,13,13,26,26,5,5,15,4,13,16,0,22,6,19,13,15,28,21,4,27,24,20,14,12,22,21,9,1,25,11,22,27,30,10,27,25,4,10,0,31,22,4,20,18,0,14,26,23,0,24,29,3,20,12,15,6,7,29,28,23,4,25,20,29,24,18,29,11,16,2,31,15,4,31,24,26,22,28,12,13,28,19,28,31,13,21,7,18,27,18,10,23,14,11,10,1,0,21,30,22,23,3,29,3,17,26,1,4,19,22,1,10,10,8,27,11,30,20,23,31,2,29,27,29,17,18,13,16,28,16,26,29,17,19,14,10,17,7,31,24,28,21,19,27,31,27,29,4,12,17,25,21,14,7,20,21,24,7,13,18,29,21,29,17,23,28,26,14,28,19,2,15,15,29,1,6,18,2,0,9,7,12,20,20,4,30,14,25,14,24,8,7,29,1,31,27,31,0,30,19,0,16,25,27,25,12,20,1,2,15,17,28,21,16,29,13,13,29,23,22,12,1,1,21,0,12,9,18,18,31,29,0,30,25,28,13,18,31,1,3,12,30,18,27,0,31,19,12,10,26,28,0,20,18,20,2,12,14,2,9,29,7,10,30,23,0,21,1,24,24,24,0,30,31,21,30,26,29,13,27,29,11,30,14,20,29,7,15,31,22,30,27,28,30,21,3,14,13,30,3,4,28,4,2,27,16,17,26,13,22,30,22,30,0,21,30,26,12,3,30,28,3,30,14,19,23,16,2,12,0,2,7,3,31,27,14,13,23,27,7,6,28,27,4,0,27,2,0,29,25,4,10,8,26,26,28,1,21,21,20,15,27,31,28,0,1,2,24,31,28,23,0,26,9,26,0,1,4,4,2,9,27,25,13,23,28,0,3,7,1,13,20,29,25,25,1,19,13,2,19,1,1,30,30,30,26,29,9,28,10,3,26,10,26,0,1,9,4,1,19,29,26,29,1,4,4,0,14,29,1,3,8,0,26,9,9,2,9,1,20,28,2,25,31,9,0,19,21,4,9,9,30,31,13,7,4,3,8,21,30,31,31,0,10,2,26,30,16,27,20,17,29,2,29,22,8,27,25,28,8,2,7,1,9,14,31,8,30,3,7,25,28,30,9,24,2,22,25,31,27,29,0,2,2,3,15,27,22,2,18,8,1,9,8,14,16,28,2,30,0,27,26,9,7,3,4,28,22,3,26,7,22,0,7,3,27,3,26,7,23,21,18,9,28,7,1,28,31,8,2,4,7,31,8,0,8,8,9,29,21,10,30,3,10,9,29,3,9,8,7,28,9,30,9,26,3,8,3,27,1,27,21,8,9,3,8,9,1,30,21,31,7,28,8,8,1,1,7,8,3,8,20,1,8,9,8,20,2,3,27,31,7,8,31,9,2,31,7,9,8,8,4,7,31,3,8,8,8,2,31,4,4,3,8,3]

letters = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero', 'oh']
dtype = torch.float

def bin_finput(data_digits, bin_size):
    ''' Binned the input. 1ms is too much precision we dont need
    I can split the time axis reshaping it into a new dimension based on the bin size and apply max to easy binning
    When bin_size is not multiple of the samples (original 2586), we remove the latest bins to round them
    # a.reshape(2, 5, 2).max(axis=2)
    '''
    _, time, _ = data_digits.shape
    t_cut = time % bin_size
    data_digits = data_digits[:, :-t_cut, :] if t_cut != 0 else data_digits
    samples, time, fchannels = data_digits.shape
    n_time = int(time/bin_size) # new time dimension based on the bin_size splits
    # Reshape and max. dtype int may save space in pytorch when tensor
    data_binned = np.array(data_digits.reshape(samples, n_time, -1, fchannels).max(axis=2), dtype=int)
    return data_binned

def read_word_files(fin, format="pickle", bin_size=None, filterT=False):
    ''' Reads all the digit_category files specified in the directory
        Out: data, as array of arrays(time, channel)
             labels, category of dataset
    '''
    data_digits = np.array([])
    labels = np.array([], dtype=str)
    for category in letters:
        if format == "lzma":
            #name = 'job_scripts/SNN_gro/keyspot_torch/data/{}/{}_{}.pkl.lzma'.format(fin, fin, category)
            #name = '../data/{}/{}_{}.pkl.lzma'.format(fin, fin, category)
            name = 'data/{}/{}_{}.pkl.lzma'.format(fin, fin, category)
            with lzma.open(name, 'rb') as handle:
                d = pickle.load(handle)
                #data = np.array([d.sum(axis=0) for d in data_raw])
        else:
            name = 'data/{}/{}_{}.pickle'.format(fin, fin, category)
            with open(name, 'rb') as handle:
                d = pickle.load(handle)
        if filterT: # Shown in rasters than not relevant info after 1499
            d = d[:,:1500,:]
        #data_digits = np.vstack([data_digits, np.sum(d,1)]) if data_digits.size else np.sum(d,1)
        data_digits = np.vstack([data_digits, d]) if data_digits.size else d
        labels = np.append(labels, np.tile(np.array([category]), len(d)))
    if bin_size:
        data_digits = bin_finput(data_digits, bin_size)
    return data_digits, labels

def filter_basedon_matlabcorr(fname):
    # Read the .mat file
    data_mat = sio.loadmat(fname)
    # indexes in matlab start with 0
    f1_l = data_mat['nf1'] - 1
    f2_l = data_mat['nf2'] - 1
    # iterate through the list to remove the cells, remove also the symmetric n1,n2 and n2,n1
    index_cells_rm = []
    ff_ch = np.array(f_ch)
    tt_ch = np.array(t_ch)
    ii_ch = np.array(i_ch)
    for f1, f2 in zip(f1_l.reshape(-1).tolist(), f2_l.reshape(-1).tolist()):
        index_cells_rm.append(int(ii_ch[(ff_ch == f1) & (tt_ch == f2)]))
        #index_cells_rm.append(int(ii_ch[(ff_ch == f2) & (tt_ch == f1)])) Not it is directional
    # return the list of indexes to remove
    return index_cells_rm

def generate_keywords_dataset_singleKW(dataset_dir, device, letter_written=letters, bin_size=1, inference=False, train_percent=None, category=None):
    (data, labels_raw) = read_word_files(dataset_dir, format="lzma", bin_size=bin_size, filterT=True)
    labels_i = [letter_written.index(l) for l in labels_raw]
    data = np.array(data)
    labels = np.array(labels_i)

    if category:
        ind_cat = (labels == letter_written.index(category))
        ind_noncat = (labels != letter_written.index(category))
        #n_cat = np.sum()
        #n_samples = int(np.floor(n_cat / 10))
        (x_cat, y_cat) = ( data[ind_cat], labels[ind_cat])
        xtr, x_noncat, ytr, y_noncat = train_test_split(data[ind_noncat], labels[ind_noncat], test_size=0.1, shuffle=True,
                                                      stratify=labels[ind_noncat])
        data = np.vstack((x_cat, x_noncat))
        labels = np.hstack((np.zeros_like(y_cat), np.ones_like(y_noncat)))


    if inference:
        x_data = torch.tensor(data, device=device, dtype=dtype)
        y_data = torch.tensor(labels, device=device, dtype=torch.long)
        ds_full = TensorDataset(x_data, y_data)
        ds_list = [ds_full]
    else:
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, shuffle=True,
                                                            stratify=labels)  # if fix seed wanted add: random_state=42
        # I can use stratify to balance the dataset based on a top_category # one/notOne [true/false]
        if train_percent:
            xtr, x_train, ytr, y_train = train_test_split(x_train, y_train, test_size=train_percent, shuffle=True,
                                                          stratify=y_train)  # if fix seed wanted add: random_state=42

        x_train = torch.tensor(x_train, device=device, dtype=dtype)
        x_test = torch.tensor(x_test, device=device, dtype=dtype)
        y_train = torch.tensor(y_train, device=device, dtype=torch.long)
        y_test = torch.tensor(y_test, device=device, dtype=torch.long)

        ds_train = TensorDataset(x_train, y_train)
        ds_test = TensorDataset(x_test, y_test)
        ds_list = [ds_train, ds_test]

    return ds_list, labels

def load_layers_alpha(file_prefix, map_location, requires_grad=True):
    """ suffix = best | last """
    global nb_hidden
    # Layers and taus file
    lays = torch.load("{}_layers.pt".format(file_prefix), map_location=map_location)
    taus = torch.load("{}_tau_alpha.pt".format(file_prefix), map_location=map_location)
    lays[0].requires_grad = False # TDE weight layer doesnt required grad
    lays[1].requires_grad = requires_grad
    taus.requires_grad = requires_grad
    nb_hidden = lays[1].shape[0]
    return (lays, taus)

def save_spikes_variables(spks_mid, spks_out, labels):
    for i in np.unique(labels):
        kword = letters[i]
        ind_key = (labels == i)
        g_name = ('{}/layer1/spikes_layer1_{}.pkl.lzma'.format(params['Dataout_dirname'], kword))
        with lzma.open(g_name, 'wb') as handle:
            pickle.dump(np.array(spks_mid[ind_key]), handle, protocol=4)
        g_name = ('{}/layer2/spikes_layer2_{}.pkl.lzma'.format(params['Dataout_dirname'], kword))
        with lzma.open(g_name, 'wb') as handle:
            pickle.dump(np.array(spks_out[ind_key]), handle, protocol=4)
    return

def compute_classification_accuracy(dataset, layers=None, tau=None, early=False, conmatrix=False, consufix='No_Specified'):
    """ Computes classification accuracy on supplied data in batches. """

    generator = DataLoader(dataset, batch_size=params['Batch_size'],
                           shuffle=False, num_workers=params['NWORKERS'])
    accs = []
    nspks_midlayer = np.array([])
    nspks_outlayer = np.array([])
    trues = []
    preds = []
    for x_local, y_local in generator:
        x_local, y_local = x_local.to(device), y_local.to(device)
        output, others, _, _ = run_snn_w_tde(x_local, layers, tau, active_dropout=False)
        del x_local
        # with output spikes
        spk_l1 = others[1].detach().cpu().numpy()
        spk_l2 = others[2].detach().cpu().numpy()
        nspks_midlayer = np.concatenate([nspks_midlayer, spk_l1], axis=0) if nspks_midlayer.size else spk_l1
        nspks_outlayer = np.concatenate([nspks_outlayer, spk_l2], axis=0) if nspks_outlayer.size else spk_l2

        m = (torch.sum(others[-1], 1))  # sum over time
        _, am = (torch.max(m, 1))  # argmax over output units
        # compare to labels
        tmp = np.mean((y_local == am).detach().cpu().numpy())
        accs.append(float(tmp))
        if conmatrix:
            trues.extend(y_local.detach().cpu().numpy())
            preds.extend(am.detach().cpu().numpy())
        del y_local

    if conmatrix:
        cm = confusion_matrix(trues, preds, normalize='true')
        cm_df = pd.DataFrame(cm, index=[ii for ii in letters], columns=[jj for jj in letters])
        cm_name = ('{}/confussion_PD_{}.pkl'.format(params['Dataout_dirname'], consufix))
        cm_df.to_pickle(cm_name)

        ''' No seaborn in FRANKLIN container
        plt.figure(figsize=(12, 9))
        sn.heatmap(cm_df,
                   annot=True,
                   fmt='.1g',
                   # linewidths=0.005,
                   # linecolor='black',
                   cbar=False,
                   square=False,
                   cmap="YlGnBu")
        plt.xlabel('\nPredicted')
        plt.ylabel('True\n')
        plt.xticks(rotation=0)
        plt.savefig('{}/confussion_{}.png'.format(Dataout_dirname, consufix), dpi=300)
        plt.close()
        '''

    return np.mean(accs), [nspks_midlayer, nspks_outlayer]

def check_accuracies(ds_train, ds_test, best_layers, best_taus):
    # Train spikes
    train_acc, _ = compute_classification_accuracy(ds_train, layers=best_layers, tau=best_taus, early=True,
                                                   conmatrix=False, consufix='TrainSplit')
    # Test spikes
    test_acc, _ = compute_classification_accuracy(ds_test, layers=best_layers, tau=best_taus, early=True,
                                                  conmatrix=False, consufix='TestSplit')
    print("Train accuracy: {}%".format(np.round(train_acc * 100, 2)))
    print("Test accuracy: {}%".format(np.round(test_acc * 100, 2)))
    print("Test accuracy as it comes, without rounding: {}".format(test_acc))

def plot_figures_epochs(loss_hist, acc_hist, dirname):
    # Figure Lost function over time
    plt.figure()
    plt.plot(range(1, len(loss_hist) + 1), loss_hist, color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss ")
    plt.savefig('{}/Loss_per_epoch.png'.format(dirname), dpi=300)
    plt.close()

    # Figure Lost function over time
    plt.figure()
    plt.plot(range(1, len(acc_hist[0]) + 1), 100 * np.array(acc_hist[0]), color='blue')
    plt.plot(range(1, len(acc_hist[1]) + 1), 100 * np.array(acc_hist[1]), color='orange')
    plt.axhline(y=(100 * np.max(np.array(acc_hist[1]))), color='red') #, xmin=0, xmax=len(acc_hist[1]), color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend(["Training", "Test"], loc='lower right')
    plt.savefig('{}/accuracy_per_epoch.png'.format(dirname), dpi=300)
    plt.close()
    return

def save_network_params(best_layers, best_taus, loss_hist, acc_hist, last_layers, last_taus):
    # Save layers and taus in pytorch format for posterior loading
    lay_name = ('{}/snn_tau_train_best_layers.pt'.format(params['Dataout_dirname']))
    torch.save(best_layers, lay_name)
    tau_name = ('{}/snn_tau_train_best_tau_alpha.pt'.format(params['Dataout_dirname']))
    torch.save(best_taus, tau_name)
    lay_name = ('{}/snn_tau_train_last_layers.pt'.format(params['Dataout_dirname']))
    torch.save(last_layers, lay_name)
    tau_name = ('{}/snn_tau_train_last_tau_alpha.pt'.format(params['Dataout_dirname']))
    torch.save(last_taus, tau_name)
    # Save variables: return loss_hist, accs_hist, best_acc_layers, best_tau_layers, ttc_hist for visualization
    f_name = ('{}/snn_tau_train_loss_epoch.pkl.lzma'.format(params['Dataout_dirname']))
    with lzma.open(f_name, 'wb') as handle:
        pickle.dump(np.array(loss_hist), handle, protocol=4)
    g_name = ('{}/snn_tau_train_accs_tv_epoch.pkl.lzma'.format(params['Dataout_dirname']))
    with lzma.open(g_name, 'wb') as handle:
        pickle.dump(np.array(acc_hist), handle, protocol=4)
    h_name = ('{}/snn_tau_train_best_layers_tde.pkl.lzma'.format(params['Dataout_dirname']))
    with lzma.open(h_name, 'wb') as handle:
        tdelayer = best_layers[0].detach().cpu().numpy()
        pickle.dump(tdelayer, handle, protocol=4)
    hh_name = ('{}/snn_tau_train_best_layers_out.pkl.lzma'.format(params['Dataout_dirname']))
    with lzma.open(hh_name, 'wb') as handle:
        outlayer = best_layers[1].detach().cpu().numpy()
        pickle.dump(outlayer, handle, protocol=4)
    i_name = ('{}/snn_tau_train_best_taus.pkl.lzma'.format(params['Dataout_dirname']))
    with lzma.open(i_name, 'wb') as handle:
        tau_tdes = best_taus.detach().cpu().numpy()
        pickle.dump(tau_tdes, handle, protocol=4)
    h_name = ('{}/snn_tau_train_last_layers_tde.pkl.lzma'.format(params['Dataout_dirname']))
    with lzma.open(h_name, 'wb') as handle:
        tdelayer = last_layers[0].detach().cpu().numpy()
        pickle.dump(tdelayer, handle, protocol=4)
    hh_name = ('{}/snn_tau_train_last_layers_out.pkl.lzma'.format(params['Dataout_dirname']))
    with lzma.open(hh_name, 'wb') as handle:
        outlayer = last_layers[1].detach().cpu().numpy()
        pickle.dump(outlayer, handle, protocol=4)
    i_name = ('{}/snn_tau_train_last_taus.pkl.lzma'.format(params['Dataout_dirname']))
    with lzma.open(i_name, 'wb') as handle:
        tau_tdes = last_taus.detach().cpu().numpy()
        pickle.dump(tau_tdes, handle, protocol=4)
    # Save model layers for classification and confusion matrix later loading values
    return

params = {
# Simulation variables ##################################################################
# Best params: wdec 1e-4 L1L2null lr 0.0015 bin_size=15 epochs 3000
'Dataout_dirname': 'job_scripts/SNN_gro/keyspot_torch/tde_unbi_ncell_5p_1_Cat', #+Category,
#'Dataout_dirname' : "/Users/apequeno/Project/GRON/software/keyword_spotting/outputs/comp_corrlevels/tde_unbi_ncell_1p_1_Cat"+Category,
'Epochs' : 3000,           # 300 set the number of epochs you want to train the network here 500
'Batch_size' : 500,        # 500 timestep 15 -- 400 for timestep5 -- 200 timestep3 -- 150 timestep1 but! now del dummy
'NWORKERS' : 0,
'Bin_size' : 15,    # All comparatives at 15
'Step_size' : 15,   # Time in cell dynamics Every cell dynamics are calculated based on this number ( Therefore the output )
'Learning_rate' : 0.0015,  # Original 0.0015  x3 0.0045
'L1_reg' : 0,       # Weight decay already applies Ori 0.0015
'L2_reg' : 0,       # Weight decay already applies Ori 0.000001
'WeightDecay' : 0.0001,
'Use_dropout' : True,
'PDrop' : 0.1,
'Enforce_Wpos' : False,
# Load previous training
'Datain_dirname' : "job_scripts/SNN_gro/keyspot_torch/gpu_tde_3k_bin1",
#'Datain_dirname' : "/Users/apequeno/Project/GRON/software/keyword_spotting/outputs/comp_corrlevels/tde_corr_unbi_011_1",
'Fileprefix_saved' : "snn_tau_train_best",
'Load_saved_model' : False,
# Filter the channels whose correlation in the spike frequencies are smaller than threshold 0c5, 1 or 2 check Input_correlation_analysis.m
'FilterCorr' : True,
'FilterCorr_file' : "job_scripts/SNN_gro/keyspot_torch/data/spikes_tidigits_noise0/filter_channels_cells_5p_",#+Category, #data/spikes_tidigits_noise0/filter_channels_cells_10p_  job_scripts/SNN_gro/keyspot_torch/data/spikes_tidigits_noise0/filter_channels_corr2
#'FilterCorr_file' :'data/spikes_tidigits_noise0/filter_channels_cells_1p_'+Category,
'FilterTime': True,    # FIXED NO EFFECT always to 1500ms filter
'TauInit_corr': False,   # If true there will be the taus in the analysis for all categories and not specific to the category in question...
'Train_percent': None   # Train percent None if it does not apply otherwise 0.25, 0.50, 0.75, 0.90
}