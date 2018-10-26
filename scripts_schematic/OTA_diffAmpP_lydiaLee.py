# -*- coding: utf-8 -*-
# Lydia Lee
# Fall 2018

# ASSUMPTIONS:
# (1) 100nm channel length, 500nm finger width.
# (2) LVT devices
# (4) 300K
# (5) TT process corner

import pprint

import numpy as np
import scipy.optimize as sciopt

from math import isnan
from bag.util.search import BinaryIterator
from verification_ec.mos.query import MOSDBDiscrete
from scipy import signal
from bag.data.lti import LTICircuit, get_w_3db, get_stability_margins

def get_db(spec_file, intent, interp_method='spline', sim_env='TT'):
    # initialize transistor database from simulation data
    mos_db = MOSDBDiscrete([spec_file], interp_method=interp_method)
    # set process corners
    mos_db.env_list = [sim_env]
    # set layout parameters
    mos_db.set_dsn_params(intent=intent)
    return mos_db

def design_biasing(db_n, db_p, sim_env,
    vdd, vout, vincm, nf_in, nf_load,
    vtail_target, itail_target, iref_res,
    vb_p, vb_n, error_tol=0.05,):
    """
    Inputs:
    Returns:
    """
    # find the closest match for operating conditions
    best_error = float('inf')
    best_op = None
    
    # operating conditions mirror that of the original diff pair
    p_op = db_p.query(vgs=vout-vdd, vds=vtail_target-vdd, vbs=vb_p-vdd)
    n_op = db_n.query(vgs=vout, vds=vout, vbs=vb_n-0)
    in_op = db_p.query(vgs=vincm-vtail_target, vds=vout-vtail_target, vbs=vb_p-vtail_target)
    
    nf_tail = int(round(abs(itail_target/p_op['ibias'])))
    if nf_tail < 1:
        # print("NF_TAIL: {} (FAIL)\n".format(nf_tail))
        return None
    itail_real = p_op['ibias']*nf_tail
    if (abs(itail_real)-abs(itail_target))/abs(itail_target) > error_tol:
        # print("ITAIL: {0:.2f}uA vs. {0:.2f}uA (FAIL)".format(itail_real*1e6, itail_target*1e6))
        return None
        
    # operating conditions mirror that of the original diff pair
    nf_tail_copy = nf_tail
    nf_in_copy = nf_in*2
    nf_load_copy = nf_load*2
    nf_mirr_min = 1
    nf_mirr_max = nf_load_copy
    nf_mirr_vec = np.arange(nf_mirr_min, nf_mirr_max, 1)
    for nf_mirr in nf_mirr_vec:
        iref = n_op['ibias']*nf_mirr
        iref_error = iref % iref_res
        if iref_error > error_tol:
            print("IREF: {0:.2f}uA (FAIL)".format(iref*1e6))
            continue
        else:
            break
    return dict(
        nf_tail=nf_tail,
        nf_bias_tail=nf_tail_copy,
        nf_bias_in=nf_in_copy,
        nf_bias_loadM=nf_load_copy,
        nf_bias_load1=nf_mirr,
        iref=iref)

def design_diffAmpP(db_n, db_p, sim_env,
        vdd, cload, vincm, T,
        gain_min, fbw_min, pm_min, inoiseout_std_max,
        vb_p, vb_n, error_tol=0.05):
    """
    Inputs:
    Returns:
    """
    # Constants
    kBT = 1.38e-23*T
    
    # TODO: change hardcoded value
    vstar_min = 0.15
    
    # find the best operating point
    best_ibias = float('inf')
    best_op = None
    
    # (loosely) constrain tail voltage
    # TODO: replace with binary search
    vtail_min = vincm + 2*vstar_min
    vtail_max = vdd
    vtail_vec = np.arange(vtail_min, vtail_max, 10e-3)

    # sweep tail voltage
    for vtail in vtail_vec:
        # (loosely) constrain output DC voltage
        # TODO: replace with binary search
        vout_min = vstar_min
        vout_max = vtail - vstar_min
        vout_vec = np.arange(vout_min, vout_max, 10e-3)
        # sweep output DC voltage
        for vout in vout_vec:
            in_op = db_p.query(vgs=vincm-vtail, vds=vout-vtail, vbs=vb_p-vtail)
            load_op = db_n.query(vgs=vout, vds=vout, vbs=vb_n-0)
            # TODO: constrain number of input devices
            nf_in_min = 4
            nf_in_max = 30
            nf_in_vec = np.arange(nf_in_min, nf_in_max, 2)
            for nf_in in nf_in_vec:
                ibranch = abs(in_op['ibias']*nf_in)
                if ibranch*2 > best_ibias:
                    continue
                # matching NMOS and PMOS bias current
                nf_load = int(abs(round(ibranch/load_op['ibias'])))
                if nf_load < 1:
                    continue 
                iload = load_op['ibias']*nf_load
                ibranch_error = (abs(iload)-abs(ibranch))/abs(ibranch)
                if ibranch_error > error_tol:
                    continue
                    
                # create LTICircuit
                amp = LTICircuit()
                amp.add_transistor(in_op, 'out', 'in', 'gnd', 'gnd', fg=nf_in)
                amp.add_transistor(load_op, 'out', 'gnd', 'gnd', 'gnd', fg=nf_load)
                amp.add_cap(cload, 'out', 'gnd')
                num, den = amp.get_num_den(in_name='in', out_name='out', in_type='v')
                
                gm = in_op['gm']*nf_in
                ro = 1/(in_op['gds']*nf_in+load_op['gds']*nf_load)
                
                # Check against gain
                gain = abs(num[-1]/den[-1])
                if gain < gain_min:
                    print("GAIN: {0:.2f} (FAIL)\n".format(gain))
                    continue
                print("GAIN: {0:.2f}".format(gain))
                
                # Check against bandwidth
                wbw = get_w_3db(num, den)
                if wbw == None:
                    print("BW: None (FAIL)\n")
                    continue
                fbw = wbw/(2*np.pi)
                if fbw < fbw_min:
                    print("BW: {0:.2f} (FAIL)\n".format(fbw))
                    continue
                print("BW: {0:.2f}".format(fbw))
                pm, gainm = get_stability_margins(num, den)
                
                # Check against phase margin
                if pm < pm_min or isnan(pm):
                    print("PM: {0:.2f} (FAIL)\n".format(pm))
                    continue
                print("PM: {0:.2f}".format(pm))
                
                # Check against noise
                inoiseout_std = np.sqrt(4*kBT*(in_op['gamma']*in_op['gm']*nf_in*2 + load_op['gamma']*load_op['gm']*nf_load*2))
                if inoiseout_std > inoiseout_std_max:
                    print("INOISE STD: {} (FAIL)\n".format(inoiseout_std))
                    continue
                print("INOISE STD: {}".format(inoiseout_std))
                
                # Check against best bias current
                if ibranch*2 < best_ibias:
                    biasing_spec = dict(db_n=db_n,
                                        db_p=db_p,
                                        sim_env=sim_env,
                                        vdd=vdd,
                                        vout=vout,
                                        vincm=vincm,
                                        nf_in=nf_in,
                                        nf_load=nf_load,
                                        vtail_target=vtail,
                                        itail_target=ibranch*2,
                                        iref_res=10e-9,
                                        vb_p=vb_p,
                                        vb_n=vb_n,
                                        error_tol=error_tol,)
                    biasing_params = design_biasing(**biasing_spec)
                    if biasing_params == None:
                        print("BIASING PARAMS (FAIL)\n")
                        continue
                    print("(SUCCESS)\n")
                    best_ibias = ibranch*2
                    best_op = dict(
                        itail=best_ibias,
                        nf_in=nf_in,
                        nf_load=nf_load,
                        vout=vout,
                        vtail=vtail,
                        gain=gain,
                        fbw=fbw,
                        pm=pm,
                        nf_tail=biasing_params['nf_tail'],
                        nf_bias_tail=biasing_params['nf_bias_tail'],
                        nf_bias_in=biasing_params['nf_bias_in'],
                        nf_bias_loadM=biasing_params['nf_bias_loadM'],
                        nf_bias_load1=biasing_params['nf_bias_load1'],
                        iref=biasing_params['iref'],
                        inoiseout_std=inoiseout_std
                        )
                    break
    if best_op == None:
        raise ValueError("No solution for P-in diffamp")
    return best_op
    
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

    specs = dict(
        db_n=nch_db,
        db_p=pch_db,
        sim_env=sim_env,
        vdd=1.0,
        cload=300e-15,
        vincm=0,
        T=300,
        gain_min=10,
        fbw_min=500e3,
        inoiseout_std_max=1,
        pm_min=70,
        vb_p=1.0,
        vb_n=0,
        error_tol=0.05
        )

    diffAmpP_specs = design_diffAmpP(**specs)
    pprint.pprint(diffAmpP_specs)
    print('done')

if __name__ == '__main__':
    run_main()
