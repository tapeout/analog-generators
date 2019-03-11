# -*- coding: utf-8 -*-
# Lydia Lee
# ASSUMPTIONS:

# (1) 500nm finger width.
# (2) RVT devices
# (3) All NMOS devices share a well, all PMOS devices share a well
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

debug_mode = True

def get_db(spec_file, intent, interp_method='spline', sim_env='TT'):
    # initialize transistor database from simulation data
    mos_db = MOSDBDiscrete([spec_file], interp_method=interp_method)
    # set process corners
    mos_db.env_list = [sim_env]
    # set layout parameters
    mos_db.set_dsn_params(intent=intent)
    return mos_db
    
def design_LNA_gm(db_n, sim_env,
        vdd, freq, Zs,
        c=0.5, delta=4/3, alpha=0.7,
        vds_target,
        vb_n, error_tol=0.05):
    """
    Chooses sizing for LNA input Gm device
    with inductive source degeneration and a series gate inductance.
    Inputs:
        db_n:       Database for NMOS device characterization data
        sim_env:    Simulation corner.
        vdd:        Float. Supply voltage in volts.
        freq:       Float. Operating frequency in Hertz.
        Zs:         Float. Output impedance of the source driving the LNA input.
        c:          Float. Correlation factor between the induced 
                    gate noise and drain thermal noise. Given in 
                    absolute value.
        delta:      Float. Coefficient for induced gate current noise. Default value for
                    long-channel devices.
        alpha:      Float. gm/gd0 ratio for noise calculations. 1 for long-channel devices.
        vds_target: Float. Target drain-source voltage for Gm device.
        vb_n:       Float. Body/back-gate voltage for NMOS devices in volts.
    Raises:
        ValueError: If unable to meet the specification requirements.
    Returns:
        A dictionary with the following key:value pairings:
        nf_MNA:     Integer. Number of fingers for device assuming fixed finger width.
        ls:         Float. Inductance of source degenerated inductor in henrys.
        vgs:        Float. Gate-drain voltage of the device in volts.
        lg:         Float. Inductance of the series gate inductor in henrys.
        cex:        Float. Capacitance added from the gate to source of the device.
    """
    # TODO: How to not hardcode this?
    nf_MNA_vec = np.arange(1, 10, 1)
    vgs = vds_target
    
    omega = freq*2*np.pi
    error = np.inf
    
    # TODO: INSERT SOME LOOP FOR BIASING CONDITION
    op_info = db_n.query(vgs=vgs, vds=vds_target, vbs=vb_n-0)
    gamma = op_info['gamma']
    for nf_MNA in nf_MNA_vec:
        cgs_temp = op_info['cgs'] * nf_MNA
        ct = cgs_temp + cex
        # Attempting to set Re(Zopt) = Re(Zs)
        # Using temporary variables to avoid overshooting and increasing
        # error; you can always adjust cext
        re_Zopt_num = np.sqrt(alpha**2 * delta * (1+c**2) / (5 * gamma))
        Zopt_den = omega * cgs_temp \
            * ((alpha**2 * delta)/(5*gamma*(1+c**2)) \
            + (ct/cgs_temp*alpha*c*np.sqrt(delta/(5*gamma)))**2)
        re_Zopt = re_Zopt_num/Zopt_den
        
        # Get as close as possible to the optimum without overshooting
        # with device sizing
        if re_Zopt > Zs:
            # Fine tune with cex. 
            # Error is now beginning to increase. Fix device size and 
            # adjust cex if necessary (binary iterative search)
            cex_iter = FloatBinaryIterator(0, 100e-15, tol=0.01, search_step=5e-15)
            while cex_iter.has_next():
                cex = cex_iter.get_next()
                ct = cgs + cex
                Zopt_den = omega * cgs_temp \
                    * ((alpha**2 * delta)/(5*gamma*(1+c**2)) \
                    + (ct/cgs_temp*alpha*c*np.sqrt(delta/(5*gamma)))**2)
                re_Zopt = re_Zopt_num/Zopt_den
                if re_Zopt > Zs:
                    cex_iter.up()
                else:
                    cex_iter.down()
            # After setting cex, break the nf_MNA loop
            break
        else:
            cgs = cgs_temp
    return
    
def design_LNA_bias(db_n, sim_env,
    vdd, ibias_res,
    vg_target, 
    vb_n, error_tol=0.05):
    """
    Inputs:
        db_n:
        sim_env:
        vdd:
        ibias_res:
        vg_target:
        vb_n:
        error_tol:
    Raises:
        ValueError: If unable to meet the specification requirements.
    Returns:
        A dictionary with the following key:value pairings:
        nf_MNBias:
        ibias:
    """
    op_info = db_n.query(vgs=vg_target, vds=vg_target, vbs=vb_n-0)
    nf_MNBias = 0
    while True:
        nf_MNBias += 1
        ibias = op_info['ibias'] * nf_MNBias
        error = ibias % ibias_res
        if abs(error/ibias) <= error_tol:
            return dict(
                nf_MNBias=nf_MNBias,
                ibias=ibias)
        cond_print("NF_BIAS: {}".format(nf_MNBias), debug_mode)
        
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

def run_bias():
    interp_method = 'spline'
    sim_env = 'TT'
    nmos_spec = 'specs_mos_char/nch_w0d5_30nm.yaml'
    intent = 'lvt'

    nch_db = get_db(nmos_spec, intent, interp_method=interp_method,
                    sim_env=sim_env)
    specs = dict(
        db_n=nch_db,
        sim_env=sim_env,
        vdd=1.2,
        ibias_res=100e-9,
        vg_target=0.5,
        vb_n=0,
        error_tol=0.05)
    
    bias_specs = design_LNA_bias(**specs)
    pprint.pprint(bias_specs)
    
    
def run_gm():
    interp_method = 'spline'
    sim_env = 'TT'
    nmos_spec = 'specs_mos_char/nch_w0d5_30nm.yaml'
    intent = 'lvt'
    nch_db = get_db(nmos_spec, intent, interp_method=interp_method,
                    sim_env=sim_env)
    specs = dict(
        db_n=nch_db,
        sim_env=sim_env,
        vdd=1.2,
        freq=2.5e9,
        Zs=50,
        c=0.5,
        delta=4/3,
        alpha=0.7,
        vds_target=0.5,
        vb_n=0,
        error_tol=0.05)
    
    bias_specs = design_LNA_gm(**specs)
    pprint.pprint(bias_gm)


def run_main():
    interp_method = 'spline'
    sim_env = 'TT'
    nmos_spec = 'specs_mos_char/nch_w0d5_30nm.yaml'
    intent = 'lvt'

    nch_db = get_db(nmos_spec, intent, interp_method=interp_method,
                    sim_env=sim_env)
                    
    specs = dict(
        db_n=nch_db,
        sim_env=sim_env)

    LNA_specs = design_LNA(**specs)
    pprint.pprint(LNA_specs)
    print('done')

if __name__ == '__main__':
    run_gm()
    # run_bias()
    # run_main()
