#!/usr/bin/env python
# -*- coding: utf-8 -*-

def conv_4b5b(bits):
    '''
    Inputs:
        bits: Iterable collection of 0s and 1s to be
            converted to 4b5b form. Zero pads at the front
            if there aren't a multiple of 4 bits.
    Outputs:
        Returns the input bits converted to 4b5b.
    Raises:
        ValueError if there's a value other than 0 or 1 in the bits.
    '''
    if len(bits)%4 != 0:
        bits_fixed = [0]*(4-len(bits)%4) + bits
    else:
        bits_fixed = bits
    lut_4b5b = ([1,1,1,1,0],
                [0,1,0,0,1],
                [1,0,1,0,0],
                [1,0,1,0,1],
                [0,1,0,1,0],
                [0,1,0,1,1],
                [0,1,1,1,0],
                [0,1,1,1,1],
                [1,0,0,1,0],
                [1,0,0,1,1],
                [1,0,1,1,0],
                [1,0,1,1,1],
                [1,1,0,1,0],
                [1,1,0,1,1],
                [1,1,1,0,0],
                [1,1,1,0,1])
    bits_split = [bits_fixed[4*i:4*(i+1)] for i in range((len(bits_fixed)+4-1) // 4 )]
    result = []
    for chunk in bits_split:
        chunk_str = ''.join([str(b) for b in chunk])
        chunk_dec = int(chunk_str, 2)
        result = result + lut_4b5b[chunk_dec]
    return result
    
def conv_3b4b(bits):
    if len(bits)%3 != 0:
        bits_fixed = [0]*(3-len(bits)%3) + bits
    else:
        bits_fixed = bits
    lut_3b4b = ([1,0,1,1],
                [0,1,1,0],
                [1,0,1,0],
                [1,1,0,0],
                [1,1,0,1],
                [0,1,0,1],
                [1,0,0,1],
                [0,1,1,1])
    bits_split = [bits_fixed[3*i:3*(i+1)] for i in range((len(bits_fixed)+3-1) // 3 )]
    result = []
    for chunk in bits_split:
        chunk_str = ''.join([str(b) for b in chunk])
        chunk_dec = int(chunk_str, 2)
        result = result + lut_3b4b[chunk_dec]
    return result
