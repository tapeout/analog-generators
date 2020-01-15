# -*- coding: utf-8 -*-

import numpy as np

from bag.util.search import FloatBinaryIterator
from bag.data.lti import LTICircuit, get_w_3db, get_stability_margins
from helper_funcs import verify_ratio

def design_levelshift_bias(db_n, vdd, voutcm, voutdiff, vb_n, 
        error_tol=0.01, vtail_res=5e-3, vstar_min=.15, ibias_max=2e-6):
    '''
    Designs gate biasing network for level shifting (current shunting) devices.
    Minimizes current consumption.
    Inputs:
        db_n:       Database for NMOS  device characterization data.
        vb_n:       Float. Back-gate/body voltage (V) of NMOS devices.      
        error_tol:  Float. Fractional tolerance for ibias error when computing
                    device ratios.
        vtail_res:  Float. Resolution for sweeping tail voltage (V).
        vstar_min:  Float. Minimum vstar voltage (V) for design.
    Outputs:
        Returns a dictionary with the following:
        Rtop:       Float. Value of the top resistor (ohms).
        Rbot:       Float. Value of the lower resistor (ohms).
        nf_in:      Integer. Number of fingers for the "input" devices.
        nf_tail:    Integer. Number of fingers for the tail device.
        vtail:      Float. Tail voltage (V).
        ibias:      Float. Bias current (A).
    '''
    voutn = voutcm - voutdiff/2
    voutp = voutcm + voutdiff/2
    vgtail = voutn
    
    best_ibias = np.inf
    best_op = None
    
    # Sweep device tail voltage
    vtail_min = vstar_min
    vtail_max = voutn
    vtail_vec = np.arange(vtail_min, vtail_max, vtail_res)
    for vtail in vtail_vec:
        print("VTAIL:\t{0}".format(vtail))
        op_tail = db_n.query(vgs=vgtail, vds=vtail, vbs=vb_n)
        op_in = db_n.query(vgs=vdd-vtail, vds=voutn-vtail, vbs=vb_n-vtail)
        
        # Based on max current, set upper limit on tail device size
        nf_tail_max = round(ibias_max/abs(op_tail['ibias']))
        if nf_tail_max < 1:
            print("FAIL: Tail too small")
            continue
        
        # Sweep size of tail device until good bias current matching
        # between the input and tail device is found
        nf_tail_vec = np.arange(2, nf_tail_max, 2)
        for nf_tail in nf_tail_vec:
            # Don't bother sizing up if the current consumption is higher
            # than the last viable solution
            ibias = abs(op_tail['ibias']*nf_tail)
            if ibias > best_ibias:
                break
                
            imatch_good, nf_in = verify_ratio(abs(op_tail['ibias']), abs(op_in['ibias']),
                nf_tail, error_tol)
            if imatch_good:
                print("SUCCESS")
                Rtop = (vdd-voutp)/ibias
                Rbot = voutdiff/ibias
                best_op = dict(
                    Rtop=Rtop,
                    Rbot=Rbot,
                    nf_in=nf_in,
                    nf_tail=nf_tail,
                    vtail=vtail,
                    ibias=ibias,
                    voutp=voutp,
                    voutn=voutn)
            else:
                print("FAIL: Bad current match")
    
    if best_op == None:
        raise ValueError("PREAMP BIAS: No solution")
    return best_op
