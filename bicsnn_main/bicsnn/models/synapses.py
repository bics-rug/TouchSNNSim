import math
import random
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from abc import ABC, abstractmethod
from collections import namedtuple
import itertools

class Synapse(ABC, object):#, nn.Module):
    Synstate = namedtuple('Synstate', ['syn_mem'])
    @abstractmethod # : no implemented here but force to be in the inherited class
    def initialize(self):
        self.stop_learning_flag = False
        self.state = self.Synstate(syn_mem=torch.zeros_like(self.tau_syn))
        self.j_eff = 1 if self.j_eff is None else self.j_eff
    def save_weight(self, folder_run, syn_name):
        file_name = folder_run.joinpath('Synapse_{}.pt'.format(syn_name))
        torch.save(self.w_eff, file_name)

    def load_weight(self, file_load):
        self.w = torch.load(file_load)
        self.w_eff = self.w.clone()

    def stop_learning(self):
        self.stop_learning_flag = True

    @torch.no_grad()
    def forward(self, input: Tensor) -> Tensor:
        h_syn = F.linear(input, self.w_eff * self.j_eff, self.bias)
        new_syn = (1 - self.dt / self.tau_syn) * self.state.syn_mem + h_syn
        self.state = self.Synstate(syn_mem=new_syn)
        return new_syn

    def reset(self):
        self.state = self.Synstate(syn_mem=torch.zeros_like(self.state.syn_mem))

class Fusi_syn(Synapse, nn.Module):
    def __init__(self, nb_inputs,
                 nb_outputs,
                 tau_syn=3.5,
                 imported_params=None,
                 prefix_param='x202x',
                 dt=1,
                 verbose=False,
                 w_max=None,
                 synaptic_percent=None):
        """
        Class plastic weights of Fusi rule.

        :param synaptic_target: normalization term that accounts for the activity in percentage and changes in plasticity
                 20/100 activation was approximated for 225 inputs at j_eff at 0.3
        ---------------------------------------------- Syn
        Bistability:
        :param alpha: upper weight of bistability
        :param beta: lower weight of bistability
        :param w_max: max weight trace
        :param w_min: min weight trace

        Weight update:

        :param imported_params: dict with parameters imported from file. If not None, the default params will be
        overwritten
        """
        super().__init__()
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs
        self.dt = dt
        self.w = None
        self.w_eff = None
        self.bias = None
        self.tau_syn = torch.nn.Parameter(torch.ones(nb_outputs), requires_grad=False) * tau_syn

        # Weight initialization:
        self.x_calcium = None
        self.j_eff = None
        self.delta_w = None
        self.mean_fwd_weight = None
        self.fwd_weight_scale = None
        self.synaptic_percent = synaptic_percent

        # Bistability:
        self.alpha = None
        self.beta = None
        self.thr_w = None
        self.w_max = None
        self.w_min = None

        self.stop_learning_flag = False

        if imported_params:
            # Overwrite parameters with input dictionary, layer based parameters are selected with prefix
            for param_name in list(imported_params.keys()):
                if isinstance(imported_params[param_name], str):
                    continue
                new_param_name = param_name.replace(prefix_param, '') if prefix_param in param_name else param_name
                eval_string = 'self.{}'.format(new_param_name) + ' = ' + str(imported_params[param_name])
                exec(eval_string)
                if verbose:
                    print(f'{param_name}: {imported_params[param_name]}')
        self.decay_stop = float(np.exp(-self.dt / self.tau_calcium))
        # Normalisation
        self.synaptic_activation_target = (self.nb_inputs * self.synaptic_percent if self.synaptic_percent is not None else None)

        if w_max is not None:
            # overwrite w_max if specified as input arg
            self.w_max = w_max

    def reset_weights(self, offset=0):
        self.w = torch.ones_like(self.w) * offset
        self.w_eff = self.w_max * (self.w >= self.thr_w) + self.w_min * (self.w < self.thr_w)

    def reset_calcium(self):
        self.x_calcium = torch.zeros_like(self.x_calcium)

    def initialize(self, file_load=None, probabilities=torch.tensor([0.5, 0.5])):
        """
        Initialize weight and synaptic traces.
        :param batch_size:
        :param probabilities  # 0 or 1 based on coding level of the input (bars with size 20% image)
        :return:
        """
        if file_load is not None:
            self.load_weight(file_load)
            self.x_calcium = torch.zeros([self.nb_outputs])
            # No Batch_size anymore control for that if it is old file
            w_sha = self.w_eff.shape
            self.w_eff = self.w_eff.reshape([w_sha[1], w_sha[2]]) if len(w_sha) == 3 else self.w_eff
            self.w = self.w.reshape([w_sha[1], w_sha[2]]) if len(w_sha) == 3 else self.w
            # Synapse has the tensor dimensions [Output, Input]
            if self.nb_inputs == w_sha[1]:
                self.w_eff = torch.t(self.w_eff)
                self.w = torch.t(self.w)
        else:
            # Weights evolve independently across samples in batch
            n_weights = self.nb_inputs * self.nb_outputs
            self.w = torch.multinomial(probabilities, num_samples=n_weights, replacement=True).reshape([self.nb_outputs, self.nb_inputs]).float()
            self.w_eff = torch.zeros_like(self.w, dtype = torch.float32)
            self.x_calcium = torch.zeros([self.nb_outputs])

            # Binary effective weight
            self.w_eff = (self.w_max * (self.w >= self.thr_w)
                          + self.w_min * (self.w < self.thr_w))
            # Synaptic normalisation based on the number of activations per output neuron
            if self.synaptic_activation_target is not None:
                for i in range(self.nb_outputs):
                    n_activations = torch.sum(self.w_eff[i, :])
                    self.w_eff[i, :] = self.w_eff[i, :] * (self.synaptic_activation_target / (n_activations if n_activations != 0 else 0.1))
        super().initialize()

    def drift(self):
        """
        Apply linear weight decay
        :return:
        """
        # Bistability:
        f = self.alpha * (self.w > self.thr_w).to(torch.bool) * (self.w_max > self.w).to(torch.bool)
        g = self.beta * (self.w < self.thr_w).to(torch.bool) * (self.w_min < self.w).to(torch.bool)

        # Weight decay:
        self.w = self.w + (f - g) * self.dt * 1 / self.tau_w #* no weight decay in original ?

    @torch.no_grad()
    def update(self, s_pre, s_post, v_post, no_update_eff=False):
        """
        Apply weight update
        :param s_pre: vector of spiking activity (at the current time step) of layer pre, size Nx1
        :param s_post: vector of spiking activity (at the current time step) of layer post, size Mx1
        :param v_post: membrane potential (at the current time step) of layer post, size
        :return:
        """
        if self.stop_learning_flag:
            return
        # Trace decay:
        old_xstop = self.x_calcium.clone()
        old_weff = self.w_eff.clone()
        self.x_calcium = self.decay_stop * self.x_calcium
        # Drift internal variable
        self.drift()

        # On post we update calcium value
        if torch.any(s_post):
            self.x_calcium = self.x_calcium + (self.c_eff * s_post) * self.dt
            #dt ??????

        # On pre (s_input): LTD or LTP as f(Calcium state)
        if torch.any(s_pre):
            # LTP
            #pot_a = self.a * (s_pre[:, :, None].to(torch.bool) & (v_post[:, None, :] >= self.thr_v) & (self.theta_pot_low <= self.x_calcium) & (self.x_calcium <= self.theta_pot_high))
            pot_a = torch.sum(self.a * (s_pre[:, :, None].to(torch.bool) & (v_post >= self.thr_v) & (self.theta_pot_low <= self.x_calcium) & (self.x_calcium <= self.theta_pot_high)), axis=0)
            # LTD
            #dep_b =  self.b * (s_pre[:, :, None].to(torch.bool) & (v_post[:, None, :] < self.thr_v) & (self.theta_dep_low <= self.x_calcium) & (self.x_calcium <= self.theta_dep_high))
            dep_b =  torch.sum(self.b * (s_pre[:, :, None].to(torch.bool) & (v_post < self.thr_v) & (self.theta_dep_low <= self.x_calcium) & (self.x_calcium <= self.theta_dep_high)), axis=0)
            self.w = self.w + (torch.t(pot_a) - torch.t(dep_b))

            # Capped w:
            self.w = torch.where(self.w > self.w_max, self.w_max, self.w)
            self.w = torch.where(self.w <= self.w_min, self.w_min, self.w)

            # Update effective weight:
            self.w_eff = (self.w_max * (self.w >= self.thr_w).to(torch.bool) +
                          self.w_min * (self.w < self.thr_w).to(torch.bool))
            # Synaptic normalisation based on the number of activations for output neuron
            if self.synaptic_activation_target is not None:
                for i in range(self.nb_outputs):
                    n_activations = torch.sum(self.w_eff[i, :])
                    self.w_eff[i, :] = self.w_eff[i, :] * (
                            self.synaptic_activation_target / (n_activations if n_activations != 0 else 0.1))
        if no_update_eff:
            self.w_eff = old_weff
        return

class Kernel_2Dmat(Synapse, nn.Module):
    ''' Use to define topographical kernels around the taxels
        From CoilNet --> dont modify for now
    '''
    def __init__(self, nb_inputs,
                 nb_outputs,
                 imported_params=None,
                 dt=1, bias=False,
                 batch_size=None, verbose=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        # Init params
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs
        self.dt = dt
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        if imported_params:
            # Overwrite parameters with input dictionary
            for param_name in list(imported_params.keys()):
                eval_string = "self.{}".format(param_name) + " = " + str(imported_params[param_name])
                exec(eval_string)
                if verbose:
                    print(f'{param_name}: {imported_params[param_name]}')
        self.bias = torch.empty(nb_outputs, **factory_kwargs) if bias else None
        self.w = torch.empty((nb_outputs, nb_inputs), **factory_kwargs)
        self.w_eff = self.w.clone()

    def construct_weight_kernel(self, size_i, size_j, kernel_i, kernel_j, parameter_optimize=False):
        ''' This weight is characterise by 2D vector that represents the input weight each weight with a unique kernel
            size_i x size_j     : size of the weight matrix input
            kernel_i x kernel_j : size of the connectivity kernel of 1's
        '''
        factory_kwargs = {'device': self.device, 'dtype': self.dtype}
        if size_i * size_j != self.nb_inputs:
            print("Error in weight definition")
            return
        w = torch.empty([self.nb_outputs, size_i, size_j])
        template = torch.zeros([size_i, size_j], dtype=int)

        ind_neuron = 0
        for i in range(size_i):
            for j in range(size_j):
                w_neuron = template.clone().detach()
                w_neuron[max(0, i-kernel_i):min(i+kernel_i, size_i-1), max(0, j-kernel_j):min(j+kernel_j, size_j-1)] = 1
                w[ind_neuron, :, :] = w_neuron
                ind_neuron += 1
        if parameter_optimize:
            self.weight = Parameter(torch.reshape(w, (self.nb_outputs, -1)), **factory_kwargs)
        else:
            self.weight = torch.reshape(w, (self.nb_outputs, -1,))

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

class LinearBinary(nn.Module):#nn.Linear):
    ''' Binarized the weights in the forward pass
            From CoilNet --> dont modify for now
    '''
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, parameter_optimize=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if parameter_optimize:
            self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs)) if bias else None
        else:
            self.weight = torch.empty((out_features, in_features), **factory_kwargs)
            self.bias = torch.empty(out_features, **factory_kwargs) if bias else None

    def initialize_binary_range(self):
        # This initialization screw all the learning, why? No negative terms...
        torch.nn.init.trunc_normal_(self.weight, mean=0.5, std=1, a=0, b=1)
        #self.binarize_weights_l1()

    def forward(self, input):
        ''' This forward pass binarized the weights in the forward pass but in the backward gradient descend uses
            the real value of the weight
        '''
        return F.linear(input, torch.clamp(self.weight.detach().round(), min=0, max=1) + (self.weight - self.weight.detach()), self.bias)


def build_connectivity_matrix_tde(channels, w_fac=100, w_trig=100, n_inputs=None, tau_fac=1, tau_trig=1, scale_w=0, max_dist=2):
    ''' Build the weight matrix that connect the input from channels to the inputs of the TDE neurons
        weight_matrix = [n_tde_neurons, n_inputs]
        [channel_1, channel_2] * [[ , ],  = facilitator_channel_TDE
                                  [ , ],
                                  [ , ],
                                  [ , ]]
        Same for trigger
        Puts to 0 the facilitator weights of the TDEs whose inputs corresponds to same channel
        Weight of facilitator and trigger is multiplied by its respective tau
    '''
    n_inputs = channels
    n_tde = n_inputs * n_inputs

    # Build weight matrix so input channels convert to the facilitator and triggering inputs of the TDE neurons
    w_fac_mat = np.array([]).reshape(n_tde, 0)
    w_trig_mat = np.array([]).reshape(0, n_inputs)
    for i in range(n_inputs):
        f_channel = np.zeros([n_tde,1])
        f_channel[i*n_inputs:(i+1)*n_inputs,:] = np.ones([n_inputs,1])
        w_fac_mat = np.hstack((w_fac_mat, f_channel))

        t_channel = np.identity(n_inputs)
        w_trig_mat = np.vstack((w_trig_mat, t_channel))

    # Remove TDE self-channels in facilitator weight
    for i in range(n_inputs):
        row = i*n_inputs + i
        column = i
        w_fac_mat[row, column] = 0.
    # Remove based on max_distance
    for column in range(0, n_inputs):
        for row in range(0, n_tde):
            if (row < column*(n_inputs+1)-max_dist) or (row > column*(n_inputs+1)+max_dist):
                w_fac_mat[row, column] = 0.

    # List of active TDEs. Only TDEs with 2 connections(facilitator and trigger) works
    connect_check = np.sum(w_trig_mat + w_fac_mat, 1)
    tde_list = np.arange(0, n_tde)
    tde_list = tde_list[(connect_check == 2)]
    # Weight scaling
    w_fac_mat = w_fac_mat * (w_fac * tau_fac)
    w_trig_mat = w_trig_mat * (w_trig * tau_trig)
    return w_fac_mat, w_trig_mat, tde_list

class Weight_synapse(Synapse, nn.Module):
    ''' Function for typical weight construction
    '''
    def __init__(self, nb_inputs,
                 nb_outputs,
                 imported_params=None,
                 dt=1, tau_syn=3.5, bias=False, prefix_param='x202x',
                 verbose=False, device=None, dtype=None, j_eff=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        # Init params
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs
        self.dt = dt
        self.tau_syn = torch.nn.Parameter(torch.ones(nb_outputs), requires_grad=False) * tau_syn
        self.device = device
        self.dtype = dtype
        self.j_eff = j_eff
        if imported_params:
            # Overwrite parameters with input dictionary, layer based parameters are selected with prefix
            for param_name in list(imported_params.keys()):
                if isinstance(imported_params[param_name], str):
                    continue
                new_param_name = param_name.replace(prefix_param, '') if prefix_param in param_name else param_name
                eval_string = 'self.{}'.format(new_param_name) + ' = ' + str(imported_params[param_name])
                exec(eval_string)
                if verbose:
                    print(f'{param_name}: {imported_params[param_name]}')

        self.bias = torch.empty(nb_outputs, **factory_kwargs) if bias else None
        self.w = torch.empty((nb_outputs, nb_inputs), **factory_kwargs)
        self.w_eff = self.w.clone()

    def initialize(self, weight_value, matrix=None, type=None,  probabilities=torch.tensor([0.5, 0.5]), size_i=1, size_j=1):
        """
        Initialize weight and synaptic traces.
        :param weight_value,
        :param type,
        :return:
        """
        # Weights for matrix multiplication
        if matrix is not None:
            self.w = torch.tensor(matrix)
        elif type is None:
            print("Error, choose a weight matrix type [] ")
        elif type == 'fully_conn':  # fully connected layer
            self.w = weight_value * torch.ones(self.nb_inputs, self.nb_outputs, dtype=torch.float32)
        elif type == 'eye_conn':  # only connected i=j --> self recurrent
            self.w = weight_value * torch.eye(self.nb_inputs, self.nb_outputs, dtype=torch.float32)
        elif type == 'inh_lateral':  # input to output all connected but the i=j --> usually itself in a recurrent
            self.w = weight_value * (1 - (torch.eye(self.nb_inputs, self.nb_outputs, dtype=torch.float32)))
        elif type == 'bin_random_prob':
            self.w = weight_value * torch.multinomial(probabilities, num_samples=self.nb_inputs * self.nb_outputs, replacement=True).reshape(
                [self.nb_inputs, self.nb_outputs]).float()
            # Effective weight, structured from the plastic ones
        elif type == 'rf_field':
            self.w = weight_value * self.rf_fields(size_i, size_j)
        elif type == 'rf_field_random':
            self.w = weight_value * self.rf_fields_random()
        elif type == 'rf_field_subfield':
            self.w = weight_value * self.rf_fields_subfield()
        else:
            print("ERROR: choose an initialization type")
        self.w = torch.t(self.w)
        self.w_eff = self.w
        super().initialize()

    def rf_fields(self, size_i, size_j):
        ## Build RF kernels based on 2D image of the input
        rfs = int(self.nb_inputs/(size_i * size_j))
        if rfs != self.nb_outputs:
            print("Error in weight definition Neurons are different than non-overlapping sizes {}x{} rfs {}= neu {}".format(size_i,size_j, rfs, self.nb_outputs))
            return

        w = torch.empty([self.nb_inputs, self.nb_outputs])
        side = int(math.sqrt(self.nb_inputs))
        template = torch.zeros([side, side], dtype=int)

        n = 0
        for i in range(int(side/size_i)):
            for j in range(int(side/size_j)):
                w_neuron = template.clone().detach()
                w_neuron[max(0, size_i * i):min(size_i * i + size_i, side),
                         max(0, size_j * j):min(size_j * j + size_j, side)] = 1
                w[:, n] = torch.flatten(w_neuron)
                n += 1

        return w

    def rf_fields_random(self):
        #randomize the input space
        rd_input = random.sample(range(self.nb_inputs), self.nb_inputs)
        w = torch.zeros([self.nb_inputs, self.nb_outputs])
        n_pern = int(math.floor(self.nb_inputs/self.nb_outputs))
        for n in range(self.nb_outputs):
            w[rd_input[int(n*n_pern):int((n+1)*n_pern)],n] = 1
        return w

    def rf_fields_subfield(self):
        #randomize the input space and select neighbours (3 in total)
        # 225 input in 25 neurons = 9 non-overlap   x3 overlap 27    9x3
        n_pern = int(math.floor(self.nb_inputs/self.nb_outputs))
        fields= 6
        size_f = 3
        base_index = random.sample(list(itertools.product(range(15), repeat=2)), self.nb_inputs)
        rd_input = base_index + base_index   # Overlap 2
        w = torch.zeros([15,15, self.nb_outputs])
        for n in range(self.nb_outputs):
            for s in range(fields):
                tup_field = rd_input.pop()
                w[tup_field[0], tup_field[1], n] = 1
                poss = list([(-1,0),(0,-1),(1,0),(0,1),(-1,-1),(1,-1),(1,1),(-1,1)])
                field = 1
                for p in poss:
                    if field == size_f:
                        break
                    try:
                        new_tup = (tup_field[0] + p[0],tup_field[1] + p[1])
                        rd_input.remove(new_tup)
                        w[new_tup[0], new_tup[1], n] = 1
                        field += 1
                    except(ValueError):
                        pass
        return torch.flatten(w,end_dim=1)

class Weight_builder(ABC, object):
    @abstractmethod # : no implemented here but force to be in the inherited class
    def build_matrix(self):
        pass

class Weight2D_builder(Weight_builder):
    def __init__(self):
        pass
    def build_matrix(self, nb_inputs, nb_outputs, weight_value, type=None, size_i=1, size_j=1):
        ''' Build RF fields in 2D according to kernel parameters '''
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs

        if type == 'kernels':
            w = weight_value * self.rf_fields_kernels(size_i, size_j)
        elif type == 'random':
            w = weight_value * self.rf_fields_random()
        elif type == 'unstructured':
            w = weight_value * self.rf_fields_unstructured()
        return w

    def rf_field_kernels(self, size_i, size_j):
        ## Build RF kernels based on 2D image of the input
        rfs = int(self.nb_inputs/(size_i * size_j))
        if rfs != self.nb_outputs:
            print("Error in weight definition Neurons are different than non-overlapping sizes {}x{} rfs {}= neu {}".format(size_i,size_j, rfs, self.nb_outputs))
            return

        w = torch.empty([self.nb_inputs, self.nb_outputs])
        side = int(math.sqrt(self.nb_inputs))
        template = torch.zeros([side, side], dtype=int)

        n = 0
        for i in range(int(side/size_i)):
            for j in range(int(side/size_j)):
                w_neuron = template.clone().detach()
                w_neuron[max(0, size_i * i):min(size_i * i + size_i, side),
                         max(0, size_j * j):min(size_j * j + size_j, side)] = 1
                w[:, n] = torch.flatten(w_neuron)
                n += 1
        return w

    def rf_field_random(self):
        pass

    def rf_field_unstructured(self):
        pass

class Weight1D_builder(Weight_builder):
    def __init__(self):
        pass
    def build_matrix(self, nb_inputs, nb_outputs, weight_value, matrix=None, type=None,  probabilities=torch.tensor([0.5, 0.5])):
        # Weights for matrix multiplication
        if matrix is not None:
            w = torch.tensor(matrix)
        elif type is None:
            print("Error, choose a weight matrix type [] ")
        elif type == 'fully_conn':  # fully connected layer
            w = weight_value * torch.ones(nb_inputs, nb_outputs, dtype=torch.float32)
        elif type == 'eye_conn':  # only connected i=j --> self recurrent
            w = weight_value * torch.eye(nb_inputs, nb_outputs, dtype=torch.float32)
        elif type == 'inh_lateral':  # input to output all connected but the i=j --> usually itself in a recurrent
            w = weight_value * (1 - (torch.eye(nb_inputs, nb_outputs, dtype=torch.float32)))
        elif type == 'bin_random_prob':
            w = weight_value * torch.multinomial(probabilities, num_samples=nb_inputs * nb_outputs,
                                                      replacement=True).reshape(
                [nb_inputs, nb_outputs]).float()
        return w