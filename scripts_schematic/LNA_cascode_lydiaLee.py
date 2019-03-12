# -*- coding: utf-8 -*-
# Lydia Lee
# Sean Huang
# ASSUMPTIONS:

# (1) 100nm finger width.
# (2) RVT devices
# (3) All NMOS devices share a well
# (4) 300K
# (5) TT process corner

import pprint

import numpy as np
import scipy.optimize as sciopt

from math import isnan
from bag.util.search import BinaryIterator, FloatBinaryIterator
from verification_ec.mos.query import MOSDBDiscrete
from scipy import signal
from bag.data.lti import LTICircuit, get_w_3db, get_stability_margins
from helper_funcs import cond_print, estimate_vth

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
        vdd, freq, re_Zs, 
        c=0.45, delta=4/3, alpha=0.7,
        vds_target=0.5, ls_max=2e-9,
        vb_n=0, error_tol=0.05):
    """
    Chooses sizing for LNA input Gm device
    with inductive source degeneration and a series gate inductance.
    Inputs:
        db_n:       Database for NMOS device characterization data
        sim_env:    Simulation corner.
        vdd:        Float. Supply voltage in volts.
        freq:       Float. Operating frequency in Hertz.
        re_Zs:      Float. Real portion of output impedance of the source 
                    driving the LNA input.
        c:          Float. Correlation factor between the induced 
                    gate noise and drain thermal noise. Given in 
                    absolute value.
        delta:      Float. Coefficient for induced gate current noise. Default value for
                    long-channel devices.
        alpha:      Float. gm/gd0 ratio for noise calculations. 1 for long-channel devices.
        vds_target: Float. Target drain-source voltage for Gm device.
        ls_max:     Float. Maximum allowable inductance in henrys.
        vb_n:       Float. Body/back-gate voltage for NMOS devices in volts.
        error_tol:  Float. Error tolerance (in fraction) for converging calculations
                    in the function.
    Raises:
        ValueError: If unable to meet the specification requirements.
    Returns:
        A dictionary with the following key:value pairings:
        nf_MNA:     Integer. Number of fingers for device assuming fixed finger width.
        ls:         Float. Inductance of source degenerated inductor in henrys.
        vgs:        Float. Gate-drain voltage of the device in volts.
        lg:         Float. Inductance of the series gate inductor in henrys.
        cex:        Float. Capacitance added from the gate to source of the device.
        ibias:      Float. Drain current in amperes for device.
        re_Zopt:    Float. Expected Re(Zopt) in ohms.
        im_Zopt:    Float. Expected Im(Zopt) in ohms.
        re_Zin:     Float. Expected Re(Zin) in ohms.
    """
    # TODO: How to not hardcode this?
    nf_MNA_min = 1
    nf_MNA_max = 100
    nf_MNA_vec = np.arange(nf_MNA_min, nf_MNA_max, 1)
    
    vstar_min = 0.15
    
    vgs_min = 0.3
    vgs_max = vdd - vstar_min
    vgs_vec = np.arange(vgs_min, vgs_max, 10e-3)
    vgs_ideal = vds_target  # For matching with the biasing network
    
    omega = freq*2*np.pi
    cgs = 0
    cex = 0
    
    # for vgs in vgs_vec:
    vgs = vds_target
    op_info = db_n.query(vgs=vgs, vds=vds_target, vbs=vb_n-0)
    cond_print("GM: {}".format(op_info['gm']), debug_mode)
    # Fudge factor
    omega_T = 100e9*2*np.pi #op_info['gm']/op_info['cgs']/26 #100e9*2*np.pi
    cond_print("wT: {}".format(omega_T), debug_mode)
    gamma = op_info['gamma']

    # Check if max sizing of device can meet spec with Cex constrained
    # to some maximum. If it doesn't, fail. Otherwise, find the value of
    # Cex for which it works.
    cgs_max = op_info['cgs'] * nf_MNA_max
    cond_print("CGS_MAX: {}".format(cgs_max), debug_mode)
    re_Zopt_num = np.sqrt(alpha**2 * delta * (1-c**2) / (5 * gamma))
    Zopt_den_maxsize = omega * cgs_max \
            * ((alpha**2 * delta * (1-c**2))/(5*gamma) \
            + (alpha*c*np.sqrt(delta/(5*gamma)))**2)
    re_Zopt_maxsize = re_Zopt_num/Zopt_den_maxsize
    cond_print("RE(ZOPT) w/ Max Cgs: {}".format(re_Zopt_maxsize), debug_mode)
    
    if re_Zopt_maxsize > re_Zs:
        cond_print("Max size alone insufficient, checking CEX...", debug_mode)
        # See if it's possible with the max feasible cex. If it is,
        # binary search. If it isn't, continue the loop.
        cex_max = 1e-12
        ct_max = cgs_max + cex_max
        Zopt_den_maxext = omega * cgs_max \
                * ((alpha**2 * delta * (1-c**2))/(5*gamma) \
                + (ct_max/cgs_max*alpha*c*np.sqrt(delta/(5*gamma)))**2)
        re_Zopt_maxext = re_Zopt_num/Zopt_den_maxext
        cond_print("Cexmax^2/Cgsmax: {}".format(cex_max**2/cgs_max), debug_mode)
        cond_print("ZOPT DENOM w/ Max Cgs, Cex: {}".format(Zopt_den_maxext), debug_mode)
        cond_print("RE(ZOPT) w/ Max Cgs, Cex: {}".format(re_Zopt_maxext), debug_mode)
        if re_Zopt_maxext > re_Zs:
            raise ValueError("No solution for gm stage with max sizing and max cex. Adjust biasing.")
        cond_print("Max sizing + some non-max Cex is viable! Continuing...", debug_mode)    
        cex_iter = FloatBinaryIterator(0, 100e-12, tol=1e-14, search_step=5e-15)
        while cex_iter.has_next():
            cex = cex_iter.get_next()
            cond_print("CEX: {}".format(cex), False)
            ct = cgs_max + cex
            Zopt_den = omega * cgs_max \
                * ((alpha**2 * delta * (1-c**2))/(5*gamma) \
                + (ct/cgs_max*alpha*c*np.sqrt(delta/(5*gamma)))**2)
            re_Zopt = re_Zopt_num/Zopt_den
            if re_Zopt > re_Zs:
                cex_iter.up()
            else:
                cex_iter.down()
        cond_print("Real portion matched. Checking Ls sizing...", debug_mode)
        # Calculate Ls to match the imaginary portion of Zopt
        im_Zopt_LHS = (ct/cgs_max*alpha*c*np.sqrt(delta/(5*gamma)))/Zopt_den
        # ls = 1/omega * (im_Zopt_LHS - im_Zs)
        ls = ct/(op_info['gm']*nf_MNA_max) * re_Zs # np.sqrt(alpha**2 * delta * (1-c**2)/(5*gamma))/(omega * omega_T * ct)
        if ls <= ls_max:
            im_Zopt = (ct/cgs_max*alpha*c*np.sqrt(delta/(5*gamma)))/Zopt_den_maxsize - omega*ls
            lg = -1/(omega**2*ct) + 2*ls + im_Zopt_LHS/omega # im_Zopt/omega
            re_Zin = op_info['gm']*nf_MNA_max*ls/ct
            return dict(nf_MNA=nf_MNA_max,
                        ls=ls,
                        vgs=vgs,
                        lg=lg,
                        cex=cex,
                        ibias=op_info['ibias'] * nf_MNA_max,
                        re_Zopt=re_Zopt,
                        im_Zopt=im_Zopt,
                        re_Zin=re_Zin)
        raise ValueError("No solution for gm stage with max sizing, non-max Cex (ginormo inductor). Adjust biasing.")
    # If you didn't need the maximum sizing of the device, get as 
    # close as possible without overshooting with Cgs, then fine-tune with Cex.
    for nf_MNA in nf_MNA_vec:
        cond_print("\nNF_MNA: {}".format(nf_MNA), debug_mode)
        cgs_temp = op_info['cgs'] * nf_MNA
        cond_print("CGS_PREV: {}\nCGS_NEXT: {}".format(cgs, cgs_temp), debug_mode)
        ct = cgs_temp + cex
        # Attempting to set Re(Zopt) = Re(Zs)
        # Using temporary variables to avoid overshooting and increasing
        # error; you can always adjust cext
        Zopt_den = omega * cgs_temp \
            * ((alpha**2 * delta * (1-c**2))/(5*gamma) \
            + (ct/cgs_temp*alpha*c*np.sqrt(delta/(5*gamma)))**2)
        re_Zopt = re_Zopt_num/Zopt_den
        cond_print("Re(Zopt) NUM: {}".format(re_Zopt_num), False)
        cond_print("Zopt DEN: {}".format(Zopt_den), False)
        cond_print("Re(Zopt): {}".format(re_Zopt), debug_mode)
        # Get as close as possible to the optimum without overshooting
        # with device sizing
        if re_Zopt < re_Zs:
            cgs = cgs_temp
            cond_print("CEX fine tune", debug_mode)
            # Fine tune with cex. 
            # Error is now beginning to increase. Fix device size and 
            # adjust cex if necessary (binary iterative search)
            cex_iter = FloatBinaryIterator(0, 100e-15, tol=1e-14, search_step=5e-15)
            while cex_iter.has_next():
                cex = cex_iter.get_next()
                ct = cgs + cex
                Zopt_den = omega * cgs \
                    * ((alpha**2 * delta * (1-c**2))/(5*gamma) \
                    + (ct/cgs*alpha*c*np.sqrt(delta/(5*gamma)))**2)
                re_Zopt = re_Zopt_num/Zopt_den
                if re_Zopt > re_Zs:
                    cex_iter.up()
                else:
                    cex_iter.down()
                    
            # After setting cex, check inductor sizing
            im_Zopt_LHS = (ct/cgs*alpha*c*np.sqrt(delta/(5*gamma)))/Zopt_den
            # ls = 1/omega * (im_Zopt_LHS - im_Zs)
            ls = ct/(op_info['gm']*nf_MNA) * re_Zs # np.sqrt(alpha**2 * delta * (1-c**2)/(5*gamma))/(omega * omega_T * ct)
            if ls <= ls_max:
                im_Zopt = (ct/cgs*alpha*c*np.sqrt(delta/(5*gamma)))/Zopt_den - omega*ls
                lg = -1/(omega**2*ct) + 2*ls + im_Zopt_LHS/omega # 1/omega * im_Zopt
                re_Zin = op_info['gm']*nf_MNA*ls/ct
                return dict(nf_MNA=nf_MNA,
                    ls=ls,
                    vgs=vgs,
                    lg=lg,
                    cex=cex,
                    ibias=op_info['ibias'] * nf_MNA,
                    re_Zopt=re_Zopt,
                    im_Zopt=im_Zopt,
                    re_Zin=re_Zin)
            else:
                cond_print("(FAIL) Inductor: {}".format(ls), debug_mode)
        else:
            cgs = cgs_temp
    
    if re_Zopt > re_Zs:
        raise ValueError("No solution for gm stage to meet real portion. Adjust biasing.")
    
    # Calculate Ls to match the imaginary portion of Zopt
    im_Zopt_LHS = (ct/cgs_temp*alpha*c*np.sqrt(delta/(5*gamma)))/Zopt_den
    # ls = 1/omega * (im_Zopt_LHS - im_Zs)
    ls = ct/(op_info['gm']*nf_MNA_max) * re_Zs #np.sqrt(alpha**2 * delta * (1-c**2)/(5*gamma))/(omega * omega_T * ct)
    
    if ls <= ls_max:
        im_Zopt = (ct/cgs_temp*alpha*c*np.sqrt(delta/(5*gamma)))/Zopt_den - omega*ls
        lg = -1/(omega**2*ct) + 2*ls + im_Zopt_LHS/omega #1/omega * im_Zopt
        re_Zin = op_info['gm']*nf_MNA_max*ls/ct
        return dict(nf_MNA=nf_MNA,
                    ls=ls,
                    vgs=vgs,
                    lg=lg,
                    cex=cex,
                    ibias=op_info['ibias'] * nf_MNA,
                    re_Zopt=re_Zopt,
                    im_Zopt=im_Zopt,
                    re_Zin=re_Zin)
    raise ValueError("No solution for gm stage (ginormo inductor). Adjust biasing.")
    
def design_LNA_bias(db_n, sim_env,
    vdd, ibias_res,
    vg_target, 
    vb_n, error_tol=0.05):
    """
    Inputs:
        db_n:       Database for NMOS device characterization data
        sim_env:    Simulation corner.
        vdd:        Float. Supply voltage in volts.
        ibias_res:  Float. Quantization resolution in amperes of bias current.
                    In other words, you probably can't reliably get
                    0.0003nA from a bias current source.
        vg_target:  Float. Target gate voltage of the biasing device in volts.
        vb_n:       Float. Body/back-gate voltage for NMOS devices in volts.
        error_tol:  Float. Error tolerance (in fraction) for converging calculations
                    in the function.
    Raises:
        ValueError: If unable to meet the specification requirements.
    Returns:
        A dictionary with the following key:value pairings:
        nf_MNBias:  Integer. Number of fingers for bias device assuming
                    fixed finger width.
        ibias:      Float. Bias current required for the bias device.
    """
    op_info = db_n.query(vgs=vg_target, vds=vg_target, vbs=vb_n-0)
    nf_MNBias = 0
    # TODO: How to avoid hardcoding this?
    while nf_MNBias < 1e3:
        nf_MNBias = nf_MNBias + 1
        ibias = op_info['ibias'] * nf_MNBias
        error = ibias % ibias_res
        if abs(error/ibias) <= error_tol:
            return dict(
                nf_MNBias=nf_MNBias,
                ibias=ibias)
        cond_print("NF_BIAS: {}".format(nf_MNBias), debug_mode)
    raise ValueError("No solution for biasing.")

def design_LNA_output(db_n, sim_env,
    vdd, vmid, cb, ibias_target,
    vb_n, error_tol=0.05):
    """
    Inputs:
        db_n:           Database for NMOS device characterization data
        sim_env:        Simulation corner.
        vdd:            Float. Supply voltage in volts.
        vmid:           Float. Source voltage of the cascode device in volts.
        cb:             Float. Capacitance attached to the gate of the cascode 
                        device in farads.
        ibias_target:   Float. Bias current in amperes for the cascode device.
        vb_n:           Float. Body/back-gate voltage for NMOS devices in volts.
        error_tol:      Float. Error tolerance (in fraction) for converging calculations
                        in the function.
    Raises:
        ValueError: If unable to meet the specification requirements.
    Returns:
        A dictionary with the following key:value pairings:
        nf_MNC: Integer. Number of fingers for cascode device assuming
                fixed finger width.
        vcas:   Float. Gate voltage for cascode device in volts.
        lo:     Float. Output inductance in henrys.
    """
    # TODO: How to avoid hardcoding this?
    vstar_min = 0.15
    vstar_max = (vdd-vmid)-vstar_min
    vstar_vec = np.flip(np.arange(vstar_min, vstar_max, 50e-3), axis=0)
    
    # Sweep across vstar, and the moment you reach tolerance (i.e. maximize
    # vstar to get the smallest possible device)
    for vstar in vstar_vec:
        op_info = db_n.query(vstar=vstar, vds=vdd-vmid, vbs=vb_n-vmid)
        nf_MNC = round(ibias_target/op_info['ibias'])
        ibias_realsies = op_info['ibias'] * nf_MNC
        error = abs(ibias_realsies-ibias_target)/ibias_target
        if error <= error_tol:
            vcas = op_info['vgs'] + vmid
            cond_print(1/(op_info['gm']*nf_MNC), debug_mode)
            return dict(nf_MNC=nf_MNC,
                        vcas=vcas,
                        lo='Fiddling required')
    raise ValueError("No solution for output stage.")
    

def design_LNA(db_n, sim_env,
    vdd, vmid, cb, freq, re_Zs, im_Zs=0, c=0.5,delta=4/3, alpha=0.7,
    ibias_res=100e-9,
    vb_n=0, error_tol=0.05):
    """
    Inputs:
        db_n:       Database for NMOS device characterization data
        sim_env:    Simulation corner.
        vdd:        Float. Supply voltage in volts.
        vmid:       Float. Source voltage of the cascode device in volts.
        freq:       Float. Operating frequency in Hertz.
        re_Zs:      Float. Real portion of output impedance of the source 
                    driving the LNA input.
        im_Zs:      Float. Imaginary portion of output impedance of the source
                    driving the LNA input.
        c:          Float. Correlation factor between the induced 
                    gate noise and drain thermal noise. Given in 
                    absolute value.
        delta:      Float. Coefficient for induced gate current noise. Default value for
                    long-channel devices.
        alpha:      Float. gm/gd0 ratio for noise calculations. 1 for long-channel devices.
        ibias_res:  Float. Quantization resolution in amperes of bias current.
                    In other words, you probably can't reliably get
                    0.0003nA from a bias current source.
        vb_n:       Float. Body/back-gate voltage for NMOS devices in volts.
        error_tol:  Float. Error tolerance (in fraction) for converging calculations
                    in the function.
    Raises:
    Returns:
    """
    gm_spec = dict(db_n=db_n,
                    sim_env=sim_env,
                    vdd=vdd,
                    freq=freq,
                    re_Zs=re_Zs,
                    im_Zs=im_Zs,
                    c=c,
                    delta=delta,
                    alpha=alpha,
                    vds_target=vmid,
                    vb_n=vb_n,
                    error_tol=error_tol)
    gm_values = design_LNA_gm(**gm_spec)
    bias_spec = dict(db_n=db_n,
                    sim_env=sim_env,
                    vdd=vdd,
                    ibias_res=ibias_res,
                    vg_target=gm_values['vgs'],
                    vb_n=vb_n,
                    error_tol=error_tol)
    bias_values = design_LNA_bias(**bias_spec)
    cascode_spec = dict(db_n=db_n,
                sim_env=sim_env,
                vdd=vdd,
                vmid=vmid,
                cb=cb,
                ibias_target=gm_values['ibias'],
                vb_n=vb_n,
                error_tol=error_tol)
    cascode_values = design_LNA_output(**cascode_spec)
    return dict(gm_values=gm_values,
                bias_values=bias_values,
                cascode_values=cascode_values)
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

def run_bias():
    interp_method = 'spline'
    sim_env = 'TT'
    nmos_spec = 'specs_mos_char/nch_w0d5_rvt_30nm.yaml'
    intent = 'rvt'

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
    nmos_spec = 'specs_mos_char/nch_w0d5_rvt_30nm.yaml'
    intent = 'rvt'
    nch_db = get_db(nmos_spec, intent, interp_method=interp_method,
                    sim_env=sim_env)
    
    specs = dict(
        db_n=nch_db,
        sim_env=sim_env,
        vdd=1.2,
        freq=2.5e9,
        re_Zs=50,
        c=0.5,
        delta=4/3,
        alpha=0.7,
        vds_target=0.5,
        vb_n=0,
        error_tol=0.05)
    
    gm_specs = design_LNA_gm(**specs)
    pprint.pprint(gm_specs)

def run_cascode():
    interp_method = 'spline'
    sim_env = 'TT'
    nmos_spec = 'specs_mos_char/nch_w0d5_rvt_30nm.yaml'
    intent = 'rvt'
    nch_db = get_db(nmos_spec, intent, interp_method=interp_method,
                    sim_env=sim_env)
                    
    specs = dict(db_n=nch_db,
                sim_env=sim_env,
                vdd=1.2,
                vmid=0.5,
                cb=20e-12,
                ibias_target=1.359e-3,
                vb_n=0,
                error_tol=0.05)
    
    cascode_specs = design_LNA_output(**specs)
    pprint.pprint(cascode_specs)

def run_main():
    interp_method = 'spline'
    sim_env = 'TT'
    nmos_spec = 'specs_mos_char/nch_w0d5_rvt_30nm.yaml'
    intent = 'rvt'

    nch_db = get_db(nmos_spec, intent, interp_method=interp_method,
                    sim_env=sim_env)
                    
    specs = dict(
        db_n=nch_db,
        sim_env=sim_env)

    LNA_specs = design_LNA(**specs)
    pprint.pprint(LNA_specs)
    print('done')

if __name__ == '__main__':
    # run_cascode()
    # run_gm()
    run_bias()
    # run_main()
