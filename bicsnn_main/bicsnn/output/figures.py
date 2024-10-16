import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import random
from math import pi

sns.set_style("darkgrid")


def raster_plot_multi(x_pre, spike_times, bins=None):
    """
    Plot spike times, histogram of spike count and inter spike interval.

    :param spike_times: list of spike trains
    :return: figure, axis
    """
    fig, axs = plt.subplots(1, 2)
    plot_image_from_matrix(torch.sum(x_pre,0), axis=axs[0])
    (n_neurons, n_time) = spike_times.shape
    for i, spt in enumerate(spike_times):
        t_spike_time = np.where(spt == 1)
        axs[1].vlines(t_spike_time, i, i + 1, color='k')


    axs[1].set_xlim([0, n_time + 1])
    axs[1].set_ylim([0, n_neurons + 1])
    axs[1].set_ylabel('Neuron')
    axs[1].set_xlabel('Time (ms)')
    fig.set_size_inches(14, 3)
    # plt.subplots_adjust(wspace='0.3')
    fig.set_dpi(200)

    return fig, axs
def plot_learning_selected_traces_bar_separated(output, input_pre, input_teach, labels, index_bar, dt_ms, outclass_i, plot_dir, labels_i=None, title=None, cmap=None, onesample=False):
    """ TODO some refactor of the options to plot --> Too many
    Plot internal variables of BCaLL rule, in response to stdp protocol.

    :param output_ltp: dictionary with results from stdp protocol (t_pre < t_post)
    :param output_ltd: dictionary with results from stdp protocol (t_pre > t_post)
    :param dt_ms: sim dt (in ms)
    :return:
    """
    if labels_i is None:
        labels_i = outclass_i
    vmem = output['mem_out'][labels == labels_i, :, outclass_i]
    calcium = output['calcium'][labels == labels_i, 0, outclass_i, :] # one per output class repeated per input for calculation in the model
    weight = output['w'][labels == labels_i, :, outclass_i, :]
    s_pre = input_pre[labels == labels_i, :, :]
    s_teach = input_teach[labels == labels_i, :, :]

    n_trials = vmem.shape[0]
    n_time_steps = vmem.shape[1]
    n_inputs = calcium.shape[1]
    t = np.arange(n_time_steps) * dt_ms

    folder_fig = plot_dir.joinpath('learning_traces__class{:d}'.format(outclass_i))
    folder_fig.mkdir(parents=True, exist_ok=True)

    for i in range(n_trials):   # n_trials
        if (i % 10 != 0):
            continue
        # Plot membrane and calcium shared for all weights
        fig, axs = plt.subplots(2, 1, sharex=True, dpi=300)
        # Traces
        df = {'t': list(t),
              'mem_out': vmem[i,:].tolist(),
              'calcium': calcium[i,:].tolist()}
        sns.lineplot(data=pd.DataFrame(df), x='t', y='mem_out', ax=axs[0], palette=cmap) # hue='f_out',
        sns.lineplot(data=pd.DataFrame(df), x='t', y='calcium', ax=axs[1], palette=cmap)
        axs[0].set_ylabel('$Vmem$')
        axs[1].set_ylabel('$C$')

        plt.tight_layout
        fig.savefig(folder_fig.joinpath('learning_trace_neuron{:d}_labels{:d}_sample{:d}.png'.format(outclass_i, labels_i, i)), format='png')
        plt.close(fig)

        # Plot of weights divided in stim
        fig, axs = plt.subplots(3, 1, sharex=True, dpi=300)
        # Traces
        df = {'t': np.tile(list(t), np.sum(index_bar)),
              'w': weight[i, index_bar, :].flatten().tolist()} #~index_bar)
        for ii in np.where(index_bar)[0]:
            axs[0].plot(t, (s_pre[i, :, ii] * (ii + 1) * 10).tolist(), marker='.', linestyle='')
        axs[1].plot(t, (s_teach[i, :, outclass_i] * (outclass_i + 1) * 10).tolist(), marker='.', linestyle='')
        #for ii in range(s_teach.shape[2]):
        axs[0].set_ylim([0.5, None])
        axs[1].set_ylim([0.5, None])
        sns.lineplot(data=pd.DataFrame(df), x='t', y='w', ax=axs[2], palette=cmap)
        axs[0].set_ylabel('$s_{in}$')
        axs[1].set_ylabel('$s_{te}$')
        axs[2].set_ylabel('$W$')
        plt.tight_layout
        fig.savefig(folder_fig.joinpath('learning_trace_neuron{:d}_labels{:d}_sample{:d}_inputBAR.png'.format(outclass_i, labels_i, i)), format='png')
        plt.close(fig)

        if onesample:
            for jj in np.where(index_bar)[0]:
            # Plot of weights divided in stim
                fig, axs = plt.subplots(3, 1, sharex=True, dpi=300)
                # Traces
                df = {'t': np.tile(list(t), 1),
                      'w': weight[i, jj, :].flatten().tolist()}  # ~index_bar)
                ii = jj
                axs[0].plot(t, (s_pre[i, :, ii] * (ii + 1) * 10).tolist(), marker='.', linestyle='')
                axs[1].plot(t, (s_teach[i, :, outclass_i] * (outclass_i + 1) * 10).tolist(), marker='.', linestyle='')
                #for ii in range(s_teach.shape[2]):
                axs[0].set_ylim([0.5, None])
                axs[1].set_ylim([0.5, None])
                sns.lineplot(data=pd.DataFrame(df), x='t', y='w', ax=axs[2], palette=cmap)
                axs[0].set_ylabel('$s_{in}$')
                axs[1].set_ylabel('$s_{te}$')
                axs[2].set_ylabel('$W$')
                plt.tight_layout
                fig.savefig(folder_fig.joinpath('learning_trace_neuron{:d}_labels{:d}_sample{:d}_inputBAR_1weight{:d}.png'.format(outclass_i, labels_i, i, jj)), format='png')
                plt.close(fig)


        # Plot of weights divided in non-stim
        fig, axs = plt.subplots(3, 1, sharex=True, dpi=300)
        # Traces
        df = {'t': np.tile(list(t), np.sum(~index_bar)),
              'w': weight[i, ~index_bar, :].flatten().tolist()} #~index_bar)
        for ii in np.where(~index_bar)[0]:
            axs[0].plot(t, (s_pre[i, :, ii] * (ii + 1) * 10).tolist(), marker='.', linestyle='')
        axs[1].plot(t, (s_teach[i, :, outclass_i] * (outclass_i + 1) * 10).tolist(), marker='.', linestyle='')
        #for ii in range(s_teach.shape[2]):
        axs[0].set_ylim([0.5, None])
        axs[1].set_ylim([0.5, None])

        sns.lineplot(data=pd.DataFrame(df), x='t', y='w', ax=axs[2], palette=cmap)
        axs[0].set_ylabel('$s_{in}$')
        axs[1].set_ylabel('$s_{te}$')
        axs[2].set_ylabel('$W$')
        plt.tight_layout
        fig.savefig(folder_fig.joinpath('learning_trace_neuron{:d}_labels{:d}_sample{:d}_inputNONBAR.png'.format(outclass_i,labels_i, i)), format='png')
        plt.close(fig)

        if onesample:
            for jj in np.where(~index_bar)[0]:
                # Plot of weights divided in non-stim
                fig, axs = plt.subplots(3, 1, sharex=True, dpi=300)
                # Traces
                df = {'t': np.tile(list(t), 1),
                      'w': weight[i, jj, :].flatten().tolist()}  # ~index_bar)
                ii = jj
                axs[0].plot(t, (s_pre[i, :, ii] * (ii + 1) * 10).tolist(), marker='.', linestyle='')
                axs[1].plot(t, (s_teach[i, :, outclass_i] * (outclass_i + 1) * 10).tolist(), marker='.', linestyle='')
                #for ii in range(s_teach.shape[2]):
                axs[0].set_ylim([0.5, None])
                axs[1].set_ylim([0.5, None])

                sns.lineplot(data=pd.DataFrame(df), x='t', y='w', ax=axs[2], palette=cmap)
                axs[0].set_ylabel('$s_{in}$')
                axs[1].set_ylabel('$s_{te}$')
                axs[2].set_ylabel('$W$')
                plt.tight_layout
                fig.savefig(folder_fig.joinpath('learning_trace_neuron{:d}_labels{:d}_sample{:d}_inputNONBAR_sample1weight{:d}.png'.format(outclass_i,labels_i, i,jj)), format='png')
                plt.close(fig)

    return

def plot_individual_learning_traces(s_pre, output, dt_ms=0.001, title=None, cmap=None):
    """
    Plot internal variables of BCaLL rule, in response to stdp protocol.

    :param output_ltp: dictionary with results from stdp protocol (t_pre < t_post)
    :param output_ltd: dictionary with results from stdp protocol (t_pre > t_post)
    :param dt_ms: sim dt (in ms)
    :return:
    """
    n_time_steps = len(output['calcium'])
    t = np.arange(n_time_steps) * dt_ms

    fig, axs = plt.subplots(6, 1, sharex=True, dpi=300)
    # Traces t_post > t_pre:
    df = {'t': [], 'mem_out': [], 'calcium': [], 'w': [], 'w_eff': [], 'syn_in':[]}
    for n_in in range(output['w'].shape[0]):
        df['syn_in'].extend([n_in] * output['w'].shape[1])
        df['w'].extend(output['w'][n_in, :].tolist())
        df['w_eff'].extend(output['w_eff'][n_in, :].tolist())
        df['calcium'].extend(output['calcium'].tolist())
        df['t'].extend(list(t))
        df['mem_out'].extend(output['mem_out'].tolist())

        axs[0].plot(t, (s_pre[:,n_in]* (n_in+1)*10).tolist(), marker='.', linestyle='')

    axs[1].plot(t, output['s_out'].tolist(), marker='.', linestyle='')
    sns.lineplot(data=pd.DataFrame(df), x='t', y='mem_out', ax=axs[2], palette=cmap) #hue='f_out', ax=axs[2], palette=cmap)
    sns.lineplot(data=pd.DataFrame(df), x='t', y='calcium', ax=axs[3], palette=cmap)#hue='f_out', ax=axs[3], palette=cmap)
    sns.lineplot(data=pd.DataFrame(df), x='t', y='w', hue='syn_in', ax=axs[4], palette=cmap)#hue='f_out', ax=axs[4], palette=cmap)
    sns.lineplot(data=pd.DataFrame(df), x='t', y='w_eff', hue='syn_in', ax=axs[5], palette=cmap) #hue='f_out', ax=axs[5], palette=cmap)
    axs[0].set_ylabel('$s_{pre}$')
    axs[1].set_ylabel('$s_{post}$')
    axs[2].set_ylabel('$Vmem_{out}$')
    axs[3].set_ylabel('$C$')
    axs[4].set_ylabel('$w$')
    axs[5].set_ylabel('$w_{eff}$')
    if title is not None:
        axs[0].set_title(title)

    return fig

def plot_calcium_traces(s_pre, s_teach, v_post, output, dt_ms, params, title=None, cmap=None):

    n_trials = output['w_eff'].shape[0]
    fig, axs = plt.subplots(1, 1, sharex=True, dpi=300)
    n_time_steps = output['calcium'].shape[3]
    t = np.arange(n_time_steps) * dt_ms

    df = {'t': [], 'calcium': []}
    for i in range(n_trials):
        df['t'].extend(list(t))
        df['calcium'].extend(output['calcium'][i,0,0,:].tolist())
    sns.histplot(data=pd.DataFrame(df), x="calcium", kde=True, stat='probability')

    plt.axvline(params['theta_dep_low'], color='r', linestyle='--')
    plt.axvline(params['theta_dep_high'], color='r', linestyle='--')
    plt.axvline(params['theta_pot_low'], color='blue', linestyle='--')
    plt.axvline(params['theta_pot_high'], color='blue', linestyle='--')
    axs.set_title(title)
    #axs.set_ylabel('$s_{pre}$')
    #if title is not None:
    #    axs[0].set_title(title)
    return fig

def plot_output_traces(s_pre, s_teach, output, dt_ms, title=None, cmap=None):
    """
    :param output_ltp: dictionary with results from stdp protocol (t_pre < t_post)
    :param output_ltd: dictionary with results from stdp protocol (t_pre > t_post)
    :param dt_ms: sim dt (in ms)
    :return:
    """
    n_trials = output['w_eff'].shape[0]
    n_time_steps = output['w_eff'].shape[3]
    t = np.arange(n_time_steps) * dt_ms

    fig, axs = plt.subplots(3, 1, sharex=True, dpi=300)
    n_time_steps = output['w_eff'].shape[3]
    t = np.arange(n_time_steps) * dt_ms
    # Traces IN:
    for i in range(s_pre.shape[0]):
        # df['$\Delta t (ms)$'].extend([output['delta_t'][i]] * len(t))
        axs[0].plot(t, (s_pre[i, :, 0] * (i + 1) * 10).tolist(), marker='.', linestyle='')
        axs[1].plot(t, (s_teach[i, :, 0] * (i + 1) * 10).tolist(), marker='.', linestyle='')
        axs[2].plot(t, output['w_eff'][i, 0, 0,:].tolist())
    axs[0].set_ylabel('$s_{pre}$')
    axs[1].set_ylabel('$s_{teach}$')
    axs[2].set_ylabel('$w_{eff}$')
    if title is not None:
        axs[0].set_title(title)
    return fig

def plot_image_from_matrix(weight, folder_fig=None, title='default', axis=None):
    ''' '''
    val_l = int(np.sqrt(weight.numel()))
    if axis is None:
        fig, ax = plt.subplots()
        if (weight.numel() / val_l) == val_l:
            plt.imshow(weight.reshape(val_l, val_l))#), vmin=0, vmax=1)
        else:
            plt.imshow(weight.reshape(-1, 1))#, vmin=0, vmax=1)
        plt.legend()
        fig.savefig(folder_fig.joinpath(title), format='png', dpi=300)
        plt.close(fig)
    else:
        if (weight.numel() / val_l) == val_l:
            axis.imshow(weight.reshape(val_l, val_l))#, vmin=0, vmax=1)
        else:
            axis.imshow(weight.reshape(-1, 1))#, vmin=0, vmax=1)
        axis.axis("off")
        plt.legend()

def plot_image_in2weight(output):
    fig, ax = plt.subplots(1,2)
    plot_image_from_matrix(output[:, 0], axis=ax[0])
    plot_image_from_matrix(output[:, -1], axis=ax[1])
    return fig

def plot_image_all_weight(weight):
    (n_in, n_out, t) = weight.shape
    n_len_row = int(np.sqrt(n_out))
    n_len_col = int(np.ceil(n_out/n_len_row))
    fig, axs = plt.subplots(n_len_row, n_len_col)
    for neu in range(n_out):
        row = int(np.floor(neu/n_len_col))
        col = int(neu % n_len_col)
        plot_image_from_matrix(weight[:, neu, -1], axis=axs[row, col])
    return fig

def plot_weight_traces_bar_intime(output, plot_dir):
    (n_trials,n_input, n_output, n_t) = output['w_eff'].shape
    weight = output['w_eff'][:, :, :, -1]                         # last time t
    #weight_eff = output['w_eff'][-1,:,:,-1]

    for bar_class in range(n_output):
        folder_fig = plot_dir.joinpath('weight_classes_{:d}'.format(bar_class))
        folder_fig.mkdir(parents=True, exist_ok=True)
        for sample in range(n_trials):
            if sample % 50 == 0 or sample == (n_trials -1):
                fig, ax = plt.subplots()
                plt.imshow(weight[sample,:, bar_class].reshape(15,15), vmin=0, vmax=1)
                plt.legend()
                fig.savefig(folder_fig.joinpath('weight_barclass_{:d}_{:03d}.png'.format(bar_class, sample)), format='png')
                plt.close(fig)

    return fig

def spiderplot_from_mat(matrix, fig_plot):
    """ matrix needs to have rows as each of the figures and columns as each of the features (neuron coeff)"""

    names = ["N{}".format(n) for n in range(matrix.shape[1])]
    df = pd.DataFrame(matrix, columns=names)

    N = len(names)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], names)

    # Draw ylabels
    #ax.set_rlabel_position(0)
    #plt.yticks([10, 20, 30], ["10", "20", "30"], color="grey", size=7)
    #plt.ylim(0, 40)

    for bar in range(df.shape[0]):
        values = df.loc[bar].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label="Class {}".format(bar))
        ax.fill(angles, values, alpha=0.1)
        ax.set_yticks([])

    # Add legend
    plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0.5), fontsize='xx-small')
    plt.title("Coefficient for population decoding")
    plt.tight_layout()
    # Show the graph
    plt.savefig(fig_plot.joinpath('linear_classification_coefficients.png'), format='png', dpi=300)
    plt.close()

def barplot_from_mat(matrix, plot_dir):
    sns.set(font_scale=1.5)
    n_classes, n_coeff = matrix.shape
    coeffs = [n for n in range(n_coeff)]
    # Plot of colour intensity based on average spks --- Process the df for neuron_example
    fig, axs = plt.subplots(1, 1, sharex=True, dpi=300)
    df = {'True Bar Class': [], 'Coeff': [], 'Value': [] }
    for bar in range(n_classes):
        df['True Bar Class'].extend([str(bar)] * n_coeff)
        #df['Output Neuron'].extend([str('N{}'.format(out_bar))] * n_lab)
        df['Coeff'].extend(coeffs)
        df['Value'].extend(matrix[bar, :])
        #plt.axis('off')
    df = pd.DataFrame(df)
    # Plot of colour intensity based on average spks --- Process the df for neuron_example
    #fig, axs = plt.subplots(1, 1, sharex=True, dpi=300)

    g = sns.FacetGrid(df, col="True Bar Class")
    g.map(sns.barplot, "Coeff", "Value")
    g.set_xticklabels(rotation=90)
    g.set_xticklabels(rotation=90, size=8)
    #plt.xlabel('Coefficient')
    #plt.ylabel('Value')
    #plt.title('Population code coefficients')
    g.tight_layout()

    g.savefig(plot_dir.joinpath('svm_coefficients.png'), format='png')
    plt.close(g.fig)

    return