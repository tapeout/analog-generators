# -*- coding: utf-8 -*-
# Lydia Lee
# Spring 2019

# ASSUMPTIONS:


import pprint
import numpy as np
import scipy.optimize as sciopt

from math import isnan, ceil, floor
from bag.util.search import BinaryIterator, FloatBinaryIterator
from verification_ec.mos.query import MOSDBDiscrete
from scipy import signal
from bag.data.lti import LTICircuit, get_w_3db, get_stability_margins
from helper_funcs import *

debug_mode = True

def design_regulator_series(db_n, db_p, sim_env,
                vb_n, vb_p, vdd, T, vnoiseref_std, cdecap, rload_min, rload_max, cload_min, cload_max, iload_min, iload_max, 
                vnoiseout_std_max, psrr_min, pm_min, vout_target, vout_fracerror_max,
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
    cond_print("VTHP: {}".format(vthp), debug_mode)
    p_vgsmax_op = db_p.query(vgs=vstar_min-vdd, vds=vref-vdd, vbs=vb_p-vdd)
    p_vgsmin_op = db_p.query(vgs=-vthp, vds=vref-vdd, vbs=vb_p-vdd)
    
    nf_p_min = ceil(iload_max/p_vgsmax_op['ibias'])
    nf_p_max = floor(iload_min/p_vgsmin_op['ibias'])
    cond_print("NF_P_MIN: {}\nNF_P_MAX: {}".format(nf_p_min, nf_p_max), debug_mode)
    if nf_p_min > nf_p_max:
        cond_print("(FAIL) Impossible PMOS sizing", debug_mode)
        return None
    
    nf_p_vec = np.arange(nf_p_min, nf_p_max, 1)
    
    def retrieve_p_op(iload, nf_p):
        """
        Returns the operating condition of the PMOS given vds,
        vbs, and iload.
        """
        vgs_max = -vthp
        vgs_min = vstar_min - vdd
        # search the appropriate biasing conditions to match the current
        vgs_iter = FloatBinaryIterator(vgs_min, vgs_max, tol=0.01, search_step=10e-3)
        while vgs_iter.has_next():
            vgs = vgs_iter.get_next()
            p_op = db_p.query(vgs=vgs, vds=vref-vdd, vbs=vb_p-vdd)
            ibias_temp = p_op['ibias']*nf_p
            if ibias_temp < iload:
                vgs_iter.down()
            else:
                vgs_iter.up()
        return p_op
    
    for nf_p in nf_p_vec:
        # Retrieving the operating conditions of the series PMOS
        p_op_imin = retrieve_p_op(iload_min, nf_p)
        p_op_imax = retrieve_p_op(iload_max, nf_p)
        vg_imin = p_op_imin['vgs'] + vdd
        vg_imax = p_op_imax['vgs'] + vdd
        
        ## Part 2: Design feedback amplifier
        ## Part 2a: Spec minimum feedback amp gain from static error
        # fractional error ~ 1/(1+Af)
        gm_imin = p_op_imin['gm'] * nf_p
        gm_imax = p_op_imax['gm'] * nf_p
        
        ro_imin = 1/(p_op_imin['gds'] * nf_p)
        ro_imax = 1/(p_op_imax['gds'] * nf_p)
        
        f_imin = gm_imin*parallel(ro_imin, rload_min)
        f_imax = gm_imax*parallel(ro_imax, rload_min)
        
        A_min = max(abs((1-vout_fracerror_max)/(vout_fracerror_max*f_imin)), abs((1-vout_fracerror_max)/(vout_fracerror_max*f_imax)))
        cond_print("Min Gain: {}".format(A_min))
        
        # Check if it's possible with a single stage, non-cascode device
        n_in_op_imin = db_n.query(vgs=vref, vds=vg_imin, vbs=vb_n)
        n_in_op_imax = db_n.query(vgs=vref, vds=vg_imax, vbs=vb_n)
        p_in_op_imin = db_p.query(vgs=vref-vdd, vds=vg_imin-vdd, vbs=vb_p-vdd)
        p_in_op_imax = db_p.query(vgs=vref-vdd, vds=vg_imax-vdd, vbs=vb_p-vdd)
        
        a0_n_imin = abs(n_in_op_imin['gm']/n_in_op_imin['gds'])/2
        a0_n_imax = abs(n_in_op_imax['gm']/n_in_op_imax['gds'])/2
        nmos_1stage_viable = min(a0_n_imin, a0_n_imax) >= A_min
        if nmos_1stage_viable:
            cond_print("(Y) NMOS single-stage gain: {}".format(min(a0_n_imin, a0_n_imax)), debug_mode)
        else:
            cond_print("(N) NMOS single-stage gain: {}".format(min(a0_n_imin, a0_n_imax)), debug_mode)
        a0_p_imin = abs(p_in_op_imin['gm']/p_in_op_imin['gds'])/2
        a0_p_imax = abs(p_in_op_imax['gm']/p_in_op_imax['gds'])/2
        pmos_1stage_viable = min(a0_p_imin, a0_p_imax) >= A_min
        if pmos_1stage_viable:
            cond_print("(Y) PMOS single-stage gain: {}".format(min(a0_p_imin, a0_p_imax)), debug_mode)
        else:
            cond_print("(N) PMOS single-stage gain: {}".format(min(a0_p_imin, a0_p_imax)), debug_mode)
        return
        # Use simplified model to determine Gm, Ro constraints
        # Ro2 = max(parallel(1/p_in_op_imax['gds'], rload_max), parallel(1/p_in_op_imin['gds'], rload_max))
        # Co2 = cload_max
        
        
        # TODO: Allow for different amplifier topologies
        
        
        # Check phase margin (calculate necessary compensation)
        # Check noise (vref + amplifier (both input-referred), vdd, PMOS)
        # Check power supply ripple
        
        
        
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

    specs = dict(db_n=nch_db,
                 db_p=pch_db,
                 sim_env=sim_env,
                 vb_n=0,
                 vb_p=0,
                 vdd=1.2,
                 T=300,
                 vnoiseref_std=0,
                 cdecap=0,
                 rload_min=10e3,
                 rload_max=1e6,
                 cload_min=100e-15,
                 cload_max=10e-12,
                 iload_min=100e-6,
                 iload_max=5e-3,
                 vnoiseout_std_max=np.inf,
                 psrr_min=0,
                 pm_min=0,
                 vout_target=0.8,
                 vout_fracerror_max=0.05,
                 cc_max=1e-12,
                 vstar_min=0.15)
    
    LDO_specs = design_regulator_series(**specs)
    pprint.pprint(LDO_specs)


if __name__ == '__main__':
    run_main()
