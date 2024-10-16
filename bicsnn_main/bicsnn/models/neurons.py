import snntorch
import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
from abc import ABC, abstractmethod
from snntorch._neurons import SpikingNeuron
import math

class Neuron(ABC, nn.Module):
    Neuronstate = namedtuple('Neuronstate', ['state_variables'])
    @abstractmethod
    def initialize(self):
        pass
    @abstractmethod
    def forward(self):
        pass

class ALIF_neuron(Neuron):
    Neuronstate = namedtuple('Neuronstate', ['syn_exc', 'syn_inh', 'syn_teacher', 'mem', 'S', 'b', 'thr'])

    def __init__(self, batch_size, nb_neurons, weight_exc, weigh_inh_rec=None, dt=1, tau_mem=10, tau_adapt=50, beta_adapt=1, thr_0=1,
                 w_teach=0, gain_exc=1, mem_reset=0, device=None,
                 inh_type='one2one', exc_type='one2one', tau_epsp=3.5, tau_ipsp=5.5, folder_name_state=None,
                 tau_ifr=None, plastic=True, pop_name=None, output_folder=None, seed_init=None,
                 is_adaptive=True):
        """
        Adaptive Leaky Integrate & Fire (ALIF) neuron model.
        :param nb_neurons: number of output units
        :param tau_mem: membrane decay time constant (ms)
        :param dt: simulation time constant (ms)
        :param tau_adapt: time constant adaptation (ms)
        :param bcall_params: dictionary with BCaLL synapse parameters.
        :param inh_type: if 'one2one', the inhibitory neurons will have one to one connections to the ALIF neurons,
        if 'all2all', the connectivity between inhibitory input and ALIF neurons will be all to all
        :param tau_epsp, time constant (in ms) of epsp
        :param tau_ipsp, time constant (in ms) of ipsp
        """
        super(ALIF_neuron, self).__init__()

        self.folder_name_state = folder_name_state
        if self.folder_name_state:
            # If loading state from folder, is in testing mode, then turn off teacher and increase input weight
            self.is_teacher_on = 0
            w_max = 10
        else:
            # If in training mode, teacher signal is on:
            self.is_teacher_on = 1

        self.dt = dt
        self.thr_0 = torch.nn.Parameter(torch.ones(nb_neurons), requires_grad=False) * thr_0
        self.beta_adapt = torch.nn.Parameter(torch.ones(nb_neurons), requires_grad=False) * beta_adapt
        self.tau_mem = torch.nn.Parameter(torch.ones(nb_neurons), requires_grad=False) * tau_mem
        self.tau_epsp = torch.nn.Parameter(torch.ones(nb_neurons), requires_grad=False) * tau_epsp
        self.tau_ipsp = torch.nn.Parameter(torch.ones(nb_neurons), requires_grad=False) * tau_ipsp
        self.plastic = plastic
        self.pop_name = pop_name
        self.output_folder = output_folder
        self.seed_init = seed_init
        self.exc_type = exc_type
        self.inh_type = inh_type

        # Plastic excitatory synapses:
        self.weight_exc = weight_exc

        self.gain_exc = gain_exc
        self.weight_teach = torch.nn.Parameter(torch.ones(nb_neurons), requires_grad=False) * w_teach

        self.weight_inh_rec = weigh_inh_rec

        # Membrane voltage reset:
        self.mem_reset = mem_reset

        self.nb_neurons = nb_neurons
        if tau_ifr:
            self.beta_ifr = torch.exp(-self.dt / torch.tensor(tau_ifr))
        else:
            # Use the neuron membrane time constant
            self.beta_ifr = torch.exp(-self.dt / self.tau_mem)

        self.tau_adapt = torch.nn.Parameter(torch.ones(nb_neurons), requires_grad=False) * tau_adapt
        self.rho = torch.exp(-self.dt / torch.tensor(tau_adapt))

        self.state = None
        self.device = device
        self.ifr = 0
        self.is_adaptive = is_adaptive

    def stop_learning(self):
        self.plastic = False

    def initialize(self, output_act):
        self.state = self.Neuronstate(syn_exc=torch.zeros_like(output_act, device=self.device),
                                      syn_inh=torch.zeros_like(output_act, device=self.device),
                                      syn_teacher=torch.zeros_like(output_act, device=self.device),
                                      mem=torch.zeros_like(output_act, device=self.device) + self.mem_reset,
                                      S=torch.zeros_like(output_act, device=self.device, dtype=float),
                                      b=torch.zeros_like(output_act, device=self.device) + self.thr_0,
                                      thr=torch.zeros_like(output_act, device=self.device))

    def reset(self):
        self.state = None

    @torch.no_grad()
    def forward(self, input, inh_input=None, teacher=None):
        """ It should never modify the weights since it is managed externally, load it in the weight class if you want
        """
        # Excitatory input:
        h1_exc = torch.einsum('ij, ijk->ik', input, self.weight_exc.w_eff * self.weight_exc.j_eff) # .type(torch.FloatTensor).to(self.device) * self.weight_exc.j_eff)
        #h1_exc = input @ (self.weight_exc.w_eff * self.weight_exc.j_eff)

        if self.state is None:
            self.initialize(h1_exc)  # the size of the post layer is inferred from the size of the matrix weight

        # Inhibitory input (coding-level dependent inhibition)
        if inh_input is not None:
            h1_inh = torch.einsum('ij, jk->ik', inh_input, self.weight_inh)
            #h1_inh = inh_input @ self.weight_inh
        else:
            h1_inh = torch.zeros_like(h1_exc)
        # Recurrent inhibition:
        if self.weight_inh_rec is not None:
            h1_inh += torch.einsum('ij, jk->ik', self.state.S.type(torch.FloatTensor), self.weight_inh_rec.w_eff)   #.type(torch.FloatTensor).to(self.device), #.type(torch.FloatTensor).to(self.device))
            #h1_inh += self.state.S @ self.weight_inh_rec.w_eff
            #h1_inh += self.state.S.type(torch.FloatTensor) @ self.weight_inh_rec.w_eff
        # Teacher input:
        if teacher is not None:
            h1_teach = teacher * self.weight_teach * self.is_teacher_on  # teacher off if in testing mode
        else:
            h1_teach = torch.zeros_like(h1_exc)

        # Reset mem: if there was a spike, reset to mem_reset, else keep last mem voltage
        # Compute ifr: -- For BCall rule
        self.ifr = self.ifr * self.beta_ifr + self.state.S

        # Leak & Integrate
        new_mem = self.state.mem - (self.dt / self.tau_mem) * (self.state.mem - self.mem_reset -
                                                               (self.state.syn_exc - self.state.syn_inh + self.state.syn_teacher))
        new_mem = torch.max(torch.FloatTensor([self.mem_reset]).expand_as(new_mem), new_mem)

        b = self.state.b - (self.dt / self.tau_adapt * (self.state.b - self.thr_0)) * int(self.is_adaptive)

        # Fire
        mthr = new_mem - b  # compare mthr to spiking threshold
        out = torch.zeros_like(mthr)#.to(torch.bool)
        out[mthr > 0] = 1.0

        is_spike = torch.where(out == 1.0)
        if len(is_spike[0]) > 0:
            new_mem[is_spike] = self.mem_reset
            b[is_spike] += self.beta_adapt[is_spike[1]] * int(self.is_adaptive)

        new_syn_exc = (1 - self.dt/self.tau_epsp) * self.state.syn_exc + h1_exc
        new_syn_inh = (1 - self.dt/self.tau_ipsp) * self.state.syn_inh + h1_inh

        if self.plastic:
            new_syn_teacher = (1 - self.dt/self.tau_epsp) * self.state.syn_teacher + h1_teach
        else:
            new_syn_teacher = torch.zeros_like(new_syn_exc)

        thr = b

        self.state = self.Neuronstate(syn_exc=new_syn_exc,
                                      syn_inh=new_syn_inh,
                                      syn_teacher=new_syn_teacher,
                                      mem=new_mem,
                                      S=out,
                                      b=b,
                                      thr=thr)

        return out

class LIF_neuron(snntorch.Leaky):
    def __init__(self, tau_mem=10, threshold=1.0, spike_grad=None, surrogate_disable=False, init_hidden=True,
                 inhibition=False, learn_beta=False, learn_threshold=False, reset_mechanism="subtract",
                 state_quant=False, output=True, graded_spikes_factor=1.0, learn_graded_spikes_factor=False):
        beta = (1 - 1/tau_mem)
        super().__init__(beta, threshold=threshold, spike_grad=spike_grad, surrogate_disable=surrogate_disable,
                           init_hidden=init_hidden, inhibition=inhibition, learn_beta=learn_beta,
                           learn_threshold=learn_threshold, reset_mechanism=reset_mechanism, state_quant=state_quant,
                           output=output, graded_spikes_factor=graded_spikes_factor,
                           learn_graded_spikes_factor=learn_graded_spikes_factor)

class TDE_neuron(Neuron, SpikingNeuron):
    Neuronstate = namedtuple('Neuronstate', ['gain', 'epsc', 'S', 'mem'])

    def __init__(self, nb_neurons, dt=1, mem_reset=0, tau_beta=2, tau_alpha=8, tau_gain=25, device=None, #150,250
                    threshold = 1.0, spike_grad = None, surrogate_disable = False, learn_beta = False,
                    output = False, graded_spikes_factor = 1.0, learn_graded_spikes_factor = False,
                    reset_mechanism="subtract"):
        """
        Leaky Integrate & Fire (LIF) neuron model.
        :param nb_neurons: number of output units
        :param tau_mem: membrane decay time constant (ms)
        :param dt: simulation time constant (ms)
        :param tau_syn: list of time constants pair to each synapse in the list of synapses
        """
        super(TDE_neuron, self).__init__( threshold=threshold, spike_grad=spike_grad, surrogate_disable=surrogate_disable, output=output, graded_spikes_factor=graded_spikes_factor, learn_graded_spikes_factor=learn_graded_spikes_factor, reset_mechanism=reset_mechanism)
        self.dt = dt

        # Membrane voltage reset:
        self.mem_reset = mem_reset
        self.nb_neurons = nb_neurons

        # Use the neuron membrane time constant
        self.alpha_gain = math.exp(-self.dt / tau_gain)
        #alpha = alpha.unsqueeze(0).repeat(bs, 1)

        self.alpha_lif = math.exp(-self.dt / tau_alpha)
        self.beta_lif = math.exp(-self.dt / tau_beta)

        self.state = None
        self.device = device
        self.ifr = 0

    def initialize(self, output_act):
        self.state = self.Neuronstate(gain=torch.zeros_like(output_act, device=self.device),
                                      epsc=torch.zeros_like(output_act, device=self.device),
                                      mem=torch.zeros_like(output_act, device=self.device) + self.mem_reset,
                                      S=torch.zeros_like(output_act, device=self.device, dtype=float))

    def reset(self):
        self.state = None

    @torch.no_grad()
    def forward(self, input):
        """ input is already current at time step to integrate, but it is bidimensional Facilitatory and Thrigger
            [neurons x 2]
            Gain traces is clamp at value 1, it does not add to the gain trace over the maximum
            Gain trace also restart after receiving a thrigger spike
        """
        hfac = input[..., 0]
        htrig = input[..., 1]

        if self.state is None:
            self.initialize(hfac)

        # Leak and integrate with Trigger permission
        new_gain = torch.clamp(self.alpha_gain * self.state.gain + hfac, max=1)  # Gain
        # Epsc --> Current that has the time constant
        new_epsc = self.alpha_lif * self.state.epsc + new_gain * htrig  # --> You can add a new parameter trained here that is the gain of the current
        # Reset gain if thrig
        new_gain = torch.tensor([0.0]) if htrig == 1.0 else new_gain
        # Integrate current in the membrane
        new_mem = self.beta_lif * self.state.mem + new_epsc  # H2 is just 0 or 1 because of spike. 1: copy the trace of syn_t ; 0: nothing

        # Fire
        out = self.fire(new_mem)
        rst = out.detach()  # We do not want to backprop through the reset

        mem = new_mem * (1.0 - rst)

        self.state = self.Neuronstate(gain=new_gain,
                                      epsc=new_epsc,
                                      mem=mem,
                                      S=out)

        return out, self.state

class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """
    scale = 5
    def __init__(self):
        scale = 5

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad
