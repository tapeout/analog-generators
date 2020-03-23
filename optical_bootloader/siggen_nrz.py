#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from sigconv_4b5b import *

'''
This script generates an NRZ waveform file.
'''

def gen_nrz(tstart, trf, vlow, vhigh, bits, tbit):
    '''
    Inputs:
        tstart: Float. Time (s) before beginning the actual bitstream.
        trf:    Float. Rise and fall time (s).
        vlow:   Float. Voltage (V) corresponding to 0.
        vhigh:  Float. Voltage (V) corresponding to 1.
        bits:   Iterable collction of 0s and 1s to be converted into
                NRZ output.
        tbit:   Float. Time (s) high or low.
    Outputs:
        Returns two values, t and vout.
        t:      Vector of time values (s).
        vout:   Vector of output values, index-matched to t.
    Raises:
        ValueError if there's a value other than 0 or 1 in bits.
    '''
    if tstart <= 0:
        t = [0]
        vout = [0]
    else:
        t = [0, tstart]
        vout = [0,0]
    
    for b in bits:
        t_current = t[-1]
        t = t + [t_current+trf, t_current+trf+tbit]
        if b == 0:
            vout = vout + [0,0]
        elif b == 1:
            vout = vout + [1,1]
        else:
            raise ValueError('{0} is not a binary value!'.format(b))
            
    return t, vout

if __name__ == '__main__':
    nlen = 3*100
    bits = [random.randint(0,1) for _ in range(nlen)]
    bits = conv_3b4b(bits)
    fileName = './nrz_data.txt'
    
    f = 1.84e6 # Hz
    tper = 1/f
    
    nrz_specs = dict(tstart = 0,
                    trf = .01*tper,
                    vlow = 0,
                    vhigh = 1,
                    bits = bits,
                    tbit = .99*tper)
    t_data, v_data = gen_nrz(**nrz_specs)
    
    with open(fileName, 'w') as f:
        for i in range(len(t_data)):
            t = t_data[i]
            v = v_data[i]
            f.write('{:.6g} {:.6g}\n'.format(t, v))
        print('Last Time = %.4g' % t)
