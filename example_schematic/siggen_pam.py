#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script generates a pwl waveform file.
"""

def de_bruijn(k, n):
    """
    De Bruijn sequence for alphabet k
    and subsequences of length n.
    """
    a = [0] * k * n
    sequence = []

    def db(t, p):
        if t > n:
            if n % p == 0:
                sequence.extend(a[1:p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)
    db(1, 1)

    return sequence + sequence[:n]


def format_neg(vec):
    if vec[0][0] >= 0.0:
        return vec
    for i in range(len(vec)):
        cx, cy = vec[i]
        if cx == 0.0:
            return vec[i:]
        elif cx > 0.0:
            px, py = vec[i - 1]
            zy = (0.0 - px) / (cx - px) * cy + (cx - 0.0) / (cx - px) * py
            return [(0.0, zy)] + vec[i:]


def code_to_voltage(code, amp):
    return (code - 1.5) / 1.5 * amp


def get_time_value_pairs(pattern, tbit, amp, td=0, tr=10e-12):
    vinit = code_to_voltage(pattern[0], amp)
    tcur = td
    if tcur >= 0.0:
        outac = [(0.0, vinit)]
    else:
        outac = [(tcur, vinit)]

    if tcur > 0.0:
        outac.append((tcur, vinit))
        
    for idx, code in enumerate(pattern):
        vac = code_to_voltage(code, amp)
        vac_next = vac if idx == len(pattern) - 1 else code_to_voltage(pattern[idx + 1], amp)
        outac.append((tcur + tbit - tr, vac))
        outac.append((tcur + tbit, vac_next))
        tcur += tbit

    return format_neg(outac)

if __name__ == '__main__':
    nsym = 64
    nlen = 2
    data = de_bruijn(nsym, nlen)
    data = data + data[:2 * nlen]
    # print(data)
    tbit = 5.0e-8
    amp = 1
    tdelay = 50e-12
    tr = 200e-12
    fname = './pam_data.txt'

    inac = get_time_value_pairs(data, tbit, amp, td=tdelay, tr=tr)

    with open(fname, 'w') as f:
        for t, v in inac:
            f.write('{:.12g} {:.12g}\n'.format(t, v))
print('last time = %.4g' % t)
