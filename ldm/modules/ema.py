import torch
from torch import nn
import logging
from contextlib import contextmanager, nullcontext
logger = logging.getLogger(__name__)

class LitEma(nn.Module):
    def __init__(self, model, decay=0.999, use_num_upates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.m_name2s_name = {}
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int) if use_num_upates
        else torch.tensor(-1, dtype=torch.int))

        for name, p in model.named_parameters():
            if p.requires_grad:
                # remove as '.'-character is not allowed in buffers
                s_name = name.replace('.', '')
                self.m_name2s_name.update({name: s_name})
                self.register_buffer(s_name, p.clone().detach().data)

        self.collected_params = []

    def reset_num_updates(self):
        del self.num_updates
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int))

    def forward(self, model):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                else:
                    assert not key in self.m_name2s_name

    def copy_to(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert not key in self.m_name2s_name

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)


# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Routines for post-hoc EMA and power function EMA proposed in the paper
"Analyzing and Improving the Training Dynamics of Diffusion Models"."""

import copy
import numpy as np
import torch

#----------------------------------------------------------------------------
# Convert power function exponent to relative standard deviation
# according to Equation 123.

def exp_to_std(exp):
    exp = np.float64(exp)
    std = np.sqrt((exp + 1) / (exp + 2) ** 2 / (exp + 3))
    return std

#----------------------------------------------------------------------------
# Convert relative standard deviation to power function exponent
# according to Equation 126 and Algorithm 2.

def std_to_exp(std):
    std = np.float64(std)
    tmp = std.flatten() ** -2
    exp = [np.roots([1, 7, 16 - t, 12 - t]).real.max() for t in tmp]
    exp = np.float64(exp).reshape(std.shape)
    return exp

#----------------------------------------------------------------------------
# Construct response functions for the given EMA profiles
# according to Equations 121 and 108.

def power_function_response(ofs, std, len, axis=0):
    ofs, std = np.broadcast_arrays(ofs, std)
    ofs = np.stack([np.float64(ofs)], axis=axis)
    exp = np.stack([std_to_exp(std)], axis=axis)
    s = [1] * exp.ndim
    s[axis] = -1
    t = np.arange(len).reshape(s)
    resp = np.where(t <= ofs, (t / ofs) ** exp, 0) / ofs * (exp + 1)
    resp = resp / np.sum(resp, axis=axis, keepdims=True)
    return resp

#----------------------------------------------------------------------------
# Compute inner products between the given pairs of EMA profiles
# according to Equation 151 and Algorithm 3.

def power_function_correlation(a_ofs, a_std, b_ofs, b_std):
    a_exp = std_to_exp(a_std)
    b_exp = std_to_exp(b_std)
    t_ratio = a_ofs / b_ofs
    t_exp = np.where(a_ofs < b_ofs, b_exp, -a_exp)
    t_max = np.maximum(a_ofs, b_ofs)
    num = (a_exp + 1) * (b_exp + 1) * t_ratio ** t_exp
    den = (a_exp + b_exp + 1) * t_max
    return num / den

#----------------------------------------------------------------------------
# Calculate beta for tracking a given EMA profile during training
# according to Equation 127.

def power_function_beta(std, t_next, t_delta):
    beta = (1 - t_delta / t_next) ** (std_to_exp(std) + 1)
    return beta

#----------------------------------------------------------------------------
# Solve the coefficients for post-hoc EMA reconstruction
# according to Algorithm 3.

def solve_posthoc_coefficients(in_ofs, in_std, out_ofs, out_std): # => [in, out]
    in_ofs, in_std = np.broadcast_arrays(in_ofs, in_std)
    out_ofs, out_std = np.broadcast_arrays(out_ofs, out_std)
    rv = lambda x: np.float64(x).reshape(-1, 1)
    cv = lambda x: np.float64(x).reshape(1, -1)
    A = power_function_correlation(rv(in_ofs), rv(in_std), cv(in_ofs), cv(in_std))
    B = power_function_correlation(rv(in_ofs), rv(in_std), cv(out_ofs), cv(out_std))
    X = np.linalg.solve(A, B)
    X = X / np.sum(X, axis=0)
    return X

#----------------------------------------------------------------------------
# Class for tracking power function EMA during the training.


class PowerEMAWrapper(nn.Module):
    def __init__(self, net, stds=[0.050, 0.100]):
        super().__init__()
        self.power_ema = PowerFunctionEMA(net, stds)
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int))

        self.collected_params = []

    def reset_num_updates(self):
        del self.num_updates
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int))

    def forward(self, model, batch_size=32):
        self.power_ema.update(self.num_updates, batch_size)

    def copy_to(self, model, std_ind=0):
        sd = self.power_ema.state_dict()["emas"][std_ind]
        model.load_state_dict(sd)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)






class PowerFunctionEMA(nn.Module):
    @torch.no_grad()
    def __init__(self, net, stds=[0.050, 0.100]):
        super().__init__()
        self.net = net
        self.stds = stds
        self.emas = [copy.deepcopy(net.state_dict()) for _std in stds]

        # clear parameters which are not being updated
        for name, param in net.named_parameters():
            if not param.requires_grad:
                for ema in self.emas:
                    if name in ema:
                        print(f"deleting {name}")
                        del ema[name]

        self.cur_nimg = 0

        self.latest_sd = None

    def move_to_device(self, device):
        self.emas = [{k: p.to(device) for k, p in ema.items()} for ema in self.emas]

    @torch.no_grad()
    def reset(self):
        for name, param in self.net.named_parameters():
            for ema in self.emas:
                if name in ema:
                    ema[name].copy_(param)

    @torch.no_grad()
    def update(self, batch_size):
        self.cur_nimg += batch_size

        # forward
        for std, ema in zip(self.stds, self.emas):
            beta = power_function_beta(std=std, t_next=self.cur_nimg, t_delta=batch_size)
            for name, p_net in self.net.named_parameters():
                if name in ema:
                    logger.debug(f"ema_device: {ema[name].device},  net_device: {p_net.device}")
                    ema[name].lerp_(p_net, 1 - beta)

    @torch.no_grad()
    def get(self):
        for ema in self.emas:
            for name, p_net in self.net.named_parameters():
                if name in ema:
                    ema[name].copy_(p_net)
        return [(ema, f'-{std:.3f}') for std, ema in zip(self.stds, self.emas)]

    def state_dict(self, **kwargs):
        return dict(stds=self.stds, emas=self.emas, cur_nimg=self.cur_nimg)

    def load_state_dict(self, state, **kwargs):
        self.stds = state['stds']
        if 'cur_nimg' not in state:
            print("Warning: 'cur_nimg' not found in state, using 0. This will likely lead to incorrect EMA updates.")
        else:
            self.cur_nimg = state['cur_nimg']
        for i, s_ema in enumerate(state['emas']):
            self.emas[i] = {k: v.clone().detach() for k, v in s_ema.items()}

    @contextmanager
    def ema_scope(self, context=None, std_ind=0):
        latest_sd = copy.deepcopy(self.net.state_dict())
        self.net.load_state_dict(self.emas[std_ind], strict=False)
        if context is not None:
            logger.info(f"{context}: Switched to EMA weights")

        try:
            yield None
        finally:
            self.net.load_state_dict(latest_sd)
            if context is not None:
                logger.info(f"{context}: Switched back to original weights")



#----------------------------------------------------------------------------
# Class for tracking traditional EMA during training.

class TraditionalEMA:
    @torch.no_grad()
    def __init__(self, net, halflife_Mimg=float('inf'), rampup_ratio=0.09):
        self.net = net
        self.halflife_Mimg = halflife_Mimg
        self.rampup_ratio = rampup_ratio
        self.ema = copy.deepcopy(net)
        self.cur_nimg = 0

    @torch.no_grad()
    def reset(self):
        for p_net, p_ema in zip(self.net.parameters(), self.ema.parameters()):
            p_ema.copy_(p_net)

    @torch.no_grad()
    def update(self, batch_size):
        halflife_Mimg = self.halflife_Mimg
        if self.rampup_ratio is not None:
            halflife_Mimg = min(halflife_Mimg, self.cur_nimg / 1e6 * self.rampup_ratio)
        beta = 0.5 ** (batch_size / max(halflife_Mimg * 1e6, 1e-8))
        for p_net, p_ema in zip(self.net.parameters(), self.ema.parameters()):
            p_ema.lerp_(p_net, 1 - beta)

        self.cur_nimg += batch_size

    @torch.no_grad()
    def get(self):
        for p_net, p_ema in zip(self.net.buffers(), self.ema.buffers()):
            p_ema.copy_(p_net)
        return self.ema

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state):
        self.ema.load_state_dict(state)


class DummyEMA():
    @torch.no_grad()
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cur_nimg = 0

    def move_to_device(self, device):
        pass

    @torch.no_grad()
    def reset(self):
        pass

    @torch.no_grad()
    def update(self, batch_size):
        pass

    @torch.no_grad()
    def get(self):
        pass

    def state_dict(self, **kwargs):
        return {}

    def load_state_dict(self, state):
        pass

    @contextmanager
    def ema_scope(self, context=None, std_ind=0):
        try:
            yield None
        finally:
            pass
