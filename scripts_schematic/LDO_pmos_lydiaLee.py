# -*- coding: utf-8 -*-
# Lydia Lee
# Spring 2019

# ASSUMPTIONS:


import pprint
import numpy as np
import scipy.optimize as sciopt

from math import isnan, ceil, floor
from bag.util.search import BinaryIterator
from verification_ec.mos.query import MOSDBDiscrete
from scipy import signal
from bag.data.lti import LTICircuit, get_w_3db, get_stability_margins
from helper_funcs import cond_print, get_db, kB, estimate_vth

debug_mode = True

def design_seriesReg(db_n, db_p, sim_env,
                vb_n, vb_p, vdd, T, vnoiseref_std, cdecap, rload_min, rload_max, iload_min, iload_max, 
                vnoiseout_std_max, psrr_min, pm_min, vout_target, vout_pcterror_max,
                cc_max, vstar_min, calc_pcterror_max=0.05):
    """
    Inputs:
    Returns:
    """
    # Constants
    kBT = kB*T
    
    ## Part 1: Assume ideal amplifier with output range within 1 vstar of top and bottom rails
    # Size PMOS device appropriately
    # From here on out, check 2 operating conditions for everything: min and max load current
    vref = vout_target
    vthp = estimate_vth(db_p, vdd, vb_p-vdd, "pmos")
    p_vgsmax_op = db_p.query(vgs=vstar_min-vdd, vds=vref-vdd, vbs=vb_p-vdd)
    p_vgsmin_op = db_p.query(vgs=-vthp, vds=vref-vdd, vbs=vb_p-vdd)
    
    nf_p_min = ceil(iload_max/p_vgsmax_op['ibias'])
    nf_p_max = floor(iload_min/p_vgsmin_op['ibias'])
    if nf_p_min > nf_p_max:
        cond_print("(FAIL) Impossible PMOS sizing", debug_mode)
        return None
    
    nf_p_vec = np.arange(nf_p_min, nf_p_max, 1)
    
    for nf_p in nf_p_vec:
        ## Part 2: Design feedback amplifier
        # Basic NMOS diff pair
        
        ## Part 2a: Define minimum feedback amp gain from static error
        # Assume the output bias is same as the input bias
        # --- Minimum load current
        vgs_min = -vthp
        vgs_max = vstar_min - vdd
        # search the appropriate biasing conditions to match the current
        vgs_vec = np.arange(vgs_min, vgs_max = 10e-3)
        closest_vgs = 0
        
        for vgs in vgs_vec:
            p_op = db_p.query(vgs=vgs, vds=vref-vdd, vbs=vb_p-vdd)
            
        
        # Check phase margin (calculate necessary compensation)
        # Check noise (vref + amplifier (both input-referred), vdd, PMOS)
        # Check power supply rejection (may not be necessary?)
        
        
        
        ## Amplifier design should spit out:
        # gain
        # bandwidth given a particular load
        # psrr(?)
        # output noise current
        # location of poles
        # output linearity range
    
def run_main():
    interp_method = 'spline'
    sim_env = 'TT'
    nmos_spec = 'specs_mos_char/nch_w0d5_100nm.yaml'
    pmos_spec = 'specs_mos_char/pch_w0d5_100nm.yaml'
    intent = 'lvt'
    
    nch_db = get_db(nmos_spec, intent, interp_method=interp_method,
                    sim_env=sim_env)
    pch_db = get_db(pmos_spec, intent, interp_method=interp_method,
                    sim_env=sim_env)

    specs = dict()
    LDO_specs = design_seriesReg(**specs)
    pprint.pprint(LDO_specs)
