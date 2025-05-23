#!/usr/bin/env python
"""
util_dsp.py

Utilities for signal processing

MuLaw Code adapted from
https://github.com/fatchord/WaveRNN/blob/master/utils/distribution.py

DCT code adapted from
https://github.com/zh217/torch-dct

"""

from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020-2021, Xin Wang"

######################
### WaveForm utilities
######################

def label_2_float(x, bits):
    """output = label_2_float(x, bits)
    
    Assume x is code index for N-bits, then convert x to float values
    Note: dtype conversion is not handled

    inputs:
    -----
       x: data to be converted Tensor.long or int, any shape. 
          x value should be [0, 2**bits-1]
       bits: number of bits, int
    
    Return:
    -------
       output: tensor.float, [-1, 1]
    
    output = 2 * x / (2**bits - 1.) - 1.
    """
    return 2 * x / (2**bits - 1.) - 1.

def float_2_label(x, bits):
    """output = float_2_label(x, bits)
    
    Assume x is a float value, do N-bits quantization and 
    return the code index.

    input
    -----
       x: data to be converted, any shape
          x value should be [-1, 1]
       bits: number of bits, int
    
    output
    ------
       output: tensor.float, [0, 2**bits-1]
    
    Although output is quantized, we use torch.float to save
    the quantized values
    """
    #assert abs(x).max() <= 1.0
    # scale the peaks
    peak = torch.abs(x).max()
    if peak > 1.0:
        x /= peak
    # quantize
    x = (x + 1.) * (2**bits - 1) / 2
    return torch.clamp(x, 0, 2**bits - 1)

def mulaw_encode(x, quantization_channels=256, scale_to_int=True):
    """x_mu = mulaw_encode(x, quantization_channels, scale_to_int=True)

    https://github.com/pytorch/audio/blob/89807cf7cdc9b40a398f2e74f15cb5a79084df60/torchaudio/functional/functional.py#L473

    Adapted from torchaudio
    https://pytorch.org/audio/functional.html mu_law_encoding

    input
    -----
       x (Tensor): Input tensor, float-valued waveforms in (-1, 1)
       quantization_channels (int): Number of channels
       scale_to_int: Bool
         True: scale mu-law to int
         False: return mu-law in (-1, 1)
        
    output
    ------
       x_mu: tensor, int64, Input after mu-law encoding
    """
    # mu 
    mu = quantization_channels - 1.0
    
    # no check on the value of x
    if not x.is_floating_point():
        x = x.to(torch.float)
    mu = torch.tensor(mu, dtype=x.dtype, device=x.device)
    x_mu = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu)
    if scale_to_int:
        x_mu = ((x_mu + 1) / 2 * mu + 0.5).to(torch.int64)
    return x_mu

def mulaw_decode(x_mu, quantization_channels, input_int=True):
    """Adapted from torchaudio
    https://pytorch.org/audio/functional.html mu_law_encoding

    Args:
        x_mu (Tensor): Input tensor
        quantization_channels (int): Number of channels
        input_int: Bool
          True: convert x_mu (int) from int to float, before mu-law decode
          False: directly decode x_mu (float) 
           
    Returns:
        Tensor: Input after mu-law decoding (float-value waveform (-1, 1))
    """
    mu = quantization_channels - 1.0
    if not x_mu.is_floating_point():
        x_mu = x_mu.to(torch.float)
    mu = torch.tensor(mu, dtype=x_mu.dtype, device=x_mu.device)
    if input_int:
        x = ((x_mu) / mu) * 2 - 1.0
    else:
        x = x_mu
    x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.0) / mu
    return x


def alaw_encode(x, A=86.5, scale_to_int=True):
    """a_law = alaw_encode(x, A, scale_to_int=False)

    input
    -----
       x (Tensor): Input tensor, float-valued waveforms in (-1, 1)
       quantization_channels (int): Number of channels
       scale_to_int: Bool
         True: scale a-law to int
         False: return a-law in (-1, 1)
        
    output
    ------
       x_a: tensor, int64, Input after a-law encoding
    """
    # no check on the value of x
    if not x.is_floating_point():
        x = x.to(torch.float)

    A = torch.tensor(A, dtype=x.dtype, device=x.device)
    x_a = torch.sign(x) * (1+torch.log(A * torch.abs(x))) / (1+torch.log(A))

    select=torch.where(torch.abs(x) < A**-1)

    x_a[select] = torch.sign(x[select]) * A * torch.abs(x[select]) / (1+torch.log(A))

    if scale_to_int:
        x_a = ((x_a + 1) / 2 * A + 0.5).to(torch.int64) #not use now 
    return x_a

######################
### DCT utilities
### https://github.com/zh217/torch-dct
### LICENSE: MIT
### 
######################

def dct1(x):
    """
    Discrete Cosine Transform, Type I
    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    return torch.fft.rfft(
        torch.cat([x, x.flip([1])[:, 1:-1]], dim=1), 1)[:, :, 0].view(*x_shape)


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I
    Our definition if idct1 is such that idct1(dct1(x)) == x
    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/ scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.fft.rfft(v, 1, onesided=False)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi/(2*N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/ scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, 
                     device=X.device)[None, :]*np.pi/(2*N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.fft.irfft(V, 1, onesided=False)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


class LinearDCT(torch_nn.Linear):
    """Implement any DCT as a linear layer; in practice this executes around
    50x faster on GPU. Unfortunately, the DCT matrix is stored, which will 
    increase memory usage.
    :param in_features: size of expected input
    :param type: which dct function in this file to use"""
    def __init__(self, in_features, type, norm=None, bias=False):
        self.type = type
        self.N = in_features
        self.norm = norm
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        # initialise using dct function
        I = torch.eye(self.N)
        if self.type == 'dct1':
            self.weight.data = dct1(I).data.t()
        elif self.type == 'idct1':
            self.weight.data = idct1(I).data.t()
        elif self.type == 'dct':
            self.weight.data = dct(I, norm=self.norm).data.t()
        elif self.type == 'idct':
            self.weight.data = idct(I, norm=self.norm).data.t()
        self.weight.requires_grad = False # don't learn this!

if __name__ == "__main__":
    print("util_dsp.py")
