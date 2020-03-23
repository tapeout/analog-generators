#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

'''
This script generates a PWL waveform file.
'''

def gen_pwm(tstart, thigh1, tlow1, thigh0, tlow0, bits, vlow, vhigh, tr, tf):
    '''
    Inputs:
        tstart: Float. Time (s) before beginning the actual bitstream.
        thigh1: Float. High-time (s) for a pulse corresponding to 1.
        tlow1:  Float. Low-time (s) for a pulse corresponding to 1. 
        thigh0: Float. High-time (s) for a pulse corresponding to 0.
        tlow0:  Float. Low-time (s) for a pulse corresponding to 0.
        bits:   Iterable collection of 0s and 1s to be converted 
                into pulse-width modulated output.
        vlow:   Float. Voltage (V) corresponding to low side of a pulse.
        vhigh:  Float. Voltage (V) corresponding to high side of a pulse.
        tr:     Float. Rise time (s).
        tf:     Float. Fall time (s).
    Outputs:
        Returns two values, t and vout.
        t:      Vector of time values (s).
        vout:   Vector of output values, index-matched to t.
    Raises:
        ValueError if there's a value other than 0 or 1 in bits.
    '''
    if tstart <= 0:
        t = [0, max(tlow1, tlow0)]
        vout = [0, 0]
    else:
        t = [0, tstart, tstart+max(tlow1, tlow0)]
        vout = [0, 0, 0]
    for b in bits:
        t_current = t[-1]
        if b == 0:
            t = t + [t_current+tr, t_current+tr+thigh0, \
                    t_current+tr+thigh0+tf, t_current+tr+thigh0+tf+tlow0]
        elif b == 1:
            t = t + [t_current+tr, t_current+tr+thigh1, \
                    t_current+tr+thigh1+tf, t_current+tr+thigh1+tf+tlow1]
        else:
            raise ValueError('{0} is not a binary value!'.format(b))
            
        vout = vout + [vhigh, vhigh, vlow, vlow]
    
    return t, vout
    
if __name__ == '__main__':
    nlen = 100
    bits = [random.randint(0,1) for _ in range(nlen)]
    fileName = './pwm_data.txt'
    
    pwm_specs = dict(tstart=0,
                    thigh1 = 1000e-9, 
                    tlow1 = 1000e-9, 
                    thigh0 = 400e-9, 
                    tlow0 = 400e-9,
                    bits = bits,
                    vlow = 0, 
                    vhigh = 1,
                    tr = 10e-9,
                    tf = 10e-9)
    
    t_data, v_data = gen_pwm(**pwm_specs)
    
    with open(fileName, 'w') as f:
        for i in range(len(t_data)):
            t = t_data[i]
            v = v_data[i]
            f.write('{:.6g} {:.6g}\n'.format(t, v))
        print('Last Time = %.4g' % t)
