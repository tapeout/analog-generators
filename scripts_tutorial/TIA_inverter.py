# -*- coding: utf-8 -*-
# Skeleton by Lydia Lee

# ASSUMPTIONS:
# (1) 100nm channel length, 500nm finger width.
# (2) LVT devices
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

def get_db(spec_file, intent, interp_method='spline', sim_env='TT'):
    # initialize transistor database from simulation data
    mos_db = MOSDBDiscrete([spec_file], interp_method=interp_method)
    # set process corners
    mos_db.env_list = [sim_env]
    # set layout parameters
    mos_db.set_dsn_params(intent=intent)
    return mos_db

def design_inverter_tia_eqn(db_n, db_p, sim_env,
        vg_res, rf_res,
        vdd, cpd, cload,
        rdc_min, fbw_min, pm_min,
        vb_n, vb_p):
    """
    Designs a transimpedance amplifier with an inverter amplifier
    in resistive feedback. Equation-based.
    Inputs:
        db_n/p:     Databases for NMOS and PMOS device characterization data, 
                    respectively.
        sim_env:    Simulation corner.
        vg_res:     Float. Step resolution when sweeping gate voltage.
        rf_res:     Float. Step resolution when sweeping feedback resistance.
        vdd:        Float. Supply voltage.
        cpd:        Float. Input parasitic capacitance.
        cload:      Float. Output load capacitance.
        rdc_min:    Float. Minimum DC transimpedance.
        fbw_min:    Float. Minimum bandwidth (Hz).
        pm_min:     Float. Minimum phase margin.
        vb_n/p:     Float. Back-gate/body voltage of NMOS and PMOS, respectively.
    Raises:
        ValueError: If unable to meet the specification requirements.
    Returns:
        A dictionary with the following key:value pairings:
        vg:     Float. Input bias voltage.
        nf_n:   Integer. NMOS number of channel fingers.
        nf_p:   Integer. PMOS number of channel fingers.
        rf:     Float. Value of feedback resistor.
        rdc:    Float. Expected DC transimpedance.
        fbw:    Float. Expected bandwidth (Hz).
        pm:     Float. Expected phase margin.
        ibias:  Float. Expected DC bias current.
    """
    ibn_fun = db_n.get_function('ibias', env=sim_env)
    ibp_fun = db_p.get_function('ibias', env=sim_env)    

    # Get sweep values (Vg, Vd)
    vg_min = 0
    vg_max = vdd
    vg_vec = np.arange(vg_min, vg_max, vg_res)
    nf_n_vec = np.arange(1, 20, 1)  # DEBUGGING: Is there a non-brute force way of setting this?

    # Find the best operating point
    best_ibias = float('inf')
    best_op = None

    for vg in vg_vec:
        vdd_vd_ratio = vdd/vg
        print("\nVD/VG: {}".format(vg))
        n_op_info = db_n.query(vgs=vg, vds=vg, vbs=vb_n-0)
        p_op_info = db_p.query(vgs=vg-vdd, vds=vg-vdd, vbs=vb_p-vdd)
        # Find ratio of fingers to get desired output common mode
        ibias_n = n_op_info['ibias']
        ibias_p = p_op_info['ibias']
        pn_match = abs(ibias_n/ibias_p)
        pn_ratio = pn_match/(vdd_vd_ratio - 1)  # DEBUGGING: Won't be exact
        if pn_ratio == 0:
            continue
        # Sweep the number of fingers to minimize power
        for nf_n in nf_n_vec:
            nf_p = int(round(nf_n * pn_ratio))
            if nf_p <= 0:
                continue
            print("N/P: {}/{} fingers".format(nf_n, nf_p))
            # Extracting FET ss parameters and scaling by width
            gds_n = n_op_info['gds'] * nf_n
            cdd_n = ### YOUR CODE HERE ###
            gm_n  = ### YOUR CODE HERE ###
            cgd_n = ### YOUR CODE HERE ###
            cgs_n = ### YOUR CODE HERE ###
            cgg_n = ### YOUR CODE HERE ###

            gds_p = ### YOUR CODE HERE ###
            cdd_p = ### YOUR CODE HERE ###
            gm_p  = ### YOUR CODE HERE ###
            cgd_p = ### YOUR CODE HERE ###
            cgs_p = ### YOUR CODE HERE ###
            cgg_p = ### YOUR CODE HERE ###
            # Extracting amp ss parameters
            gm = abs(gm_n) + abs(gm_p)
            ro = 1/abs(gds_n + gds_p)
            A0 = gm*ro
            cout = cload + cdd_n + cdd_p
            # Assume Rdc is negative, bound Rf
            rf_min = max(rdc_min*(1+A0)/A0 + 1/gm, 0)
            rf_vec = np.arange(rf_min, rf_min*3, rf_res)
            # Sweep values of Rf to check f3dB and PM spec
            for rf in rf_vec:
                rdc = ### YOUR CODE HERE ###
                miller = (1-gm*rf)/(ro+rf)*ro
                cin = cpd + (1-miller)*(cgd_n + cgd_p) + cgs_n + cgs_p # cpd + cgg_p + cgg_n
                if abs(rdc) < rdc_min-1e-8:
                    print("RDC: {0:.2f} (FAIL)\n".format(rdc))
                    continue
                else:
                    print("RDC: {0:.2f}".format(rdc))
                wn = np.sqrt((1+A0)/(ro*rf*cin*cout))
                zeta = 1/(2*np.sqrt(1+A0)) * \
                    (np.sqrt(ro/rf * cin/cout) \
                    + np.sqrt(ro/rf * cout/cin) \
                    + np.sqrt(rf/ro * cin/cout))
                wbw = ### YOUR CODE HERE ###
                fbw = wbw/(2*np.pi)
                if fbw < fbw_min or isnan(fbw):
                    print("BW: {} (FAIL)\n".format(fbw))
                    break
                else:
                    print("BW: {}".format(fbw))
                pm = np.degrees(### YOUR CODE HERE ###)
                if pm < pm_min or isnan(pm):
                    print("PM: {0:.2f} (FAIL)\n".format(pm))
                    continue
                else:
                    print("PM: {0:.2f}\n".format(pm))
                if ibias_n*nf_n < best_ibias:
                    best_ibias = ibias_n*nf_n
                    best_op = dict(
                    vg=vg,
                    nf_n=nf_n,
                    nf_p=nf_p,
                    rf=rf,
                    rdc=rdc,
                    fbw=fbw,
                    pm=pm,
                    ibias=best_ibias)
                
    if best_op == None:
        raise ValueError("No solutions.")
    return best_op

def design_inverter_tia_lti(db_n, db_p, sim_env,
        vg_res, rf_res,
        vdd, cpd, cload, 
        rdc_min, fbw_min, pm_min,
        vb_n, vb_p):
    """
    Designs a transimpedance amplifier with an inverter amplifier
    in resistive feedback. Uses the LTICircuit functionality.
    Inputs:
        db_n/p:     Databases for NMOS and PMOS device characterization data, 
                    respectively.
        sim_env:    Simulation corner.
        vg_res:     Float. Step resolution when sweeping gate voltage.
        rf_res:     Float. Step resolution when sweeping feedback resistance.
        vdd:        Float. Supply voltage.
        cpd:        Float. Input parasitic capacitance.
        cload:      Float. Output load capacitance.
        rdc_min:    Float. Minimum DC transimpedance.
        fbw_min:    Float. Minimum bandwidth (Hz).
        pm_min:     Float. Minimum phase margin.
        vb_n/p:     Float. Back-gate/body voltage of NMOS and PMOS, respectively.
    Raises:
        ValueError: If unable to meet the specification requirements.
    Returns:
        A dictionary with the following key:value pairings:
        vg:     Float. Input bias voltage.
        nf_n:   Integer. NMOS number of channel fingers.
        nf_p:   Integer. PMOS number of channel fingers.
        rf:     Float. Value of feedback resistor.
        rdc:    Float. Expected DC transimpedance.
        fbw:    Float. Expected bandwidth (Hz).
        pm:     Float. Expected phase margin.
        ibias:  Float. Expected DC bias current.
    """
    ibn_fun = db_n.get_function('ibias', env=sim_env)
    ibp_fun = db_p.get_function('ibias', env=sim_env)    

    # Get sweep values (Vg, Vd)
    vg_min = 0
    vg_max = vdd
    vg_vec = np.arange(vg_min, vg_max, vg_res)
    nf_n_vec = np.arange(1, 20, 1)  # DEBUGGING: Is there a non-brute force way of setting this?

    # Find the best operating point
    best_ibias = float('inf')
    best_op = None

    for vg in vg_vec:
        vdd_vd_ratio = vdd/vg
        #print("\nVD/VG: {}".format(vg))
        n_op_info = db_n.query(vgs=vg, vds=vg, vbs=vb_n-0)
        p_op_info = db_p.query(vgs=vg-vdd, vds=vg-vdd, vbs=vb_p-vdd)
        # Find ratio of fingers to get desired output common mode
        ibias_n = n_op_info['ibias']
        ibias_p = p_op_info['ibias']
        pn_match = abs(ibias_n/ibias_p)
        pn_ratio = pn_match/(vdd_vd_ratio - 1)  # DEBUGGING: Won't be exact
        if pn_ratio == 0:
            continue
        # Sweep the number of fingers to minimize power
        for nf_n in nf_n_vec:
            nf_p = int(round(nf_n * pn_ratio))
            if nf_p <= 0:
                continue
            ibias_error = abs(abs(ibias_p)*nf_p-abs(ibias_n)*nf_n)/(abs(ibias_n)*nf_n)
            if ibias_error > 0.05:
                continue
            print("N/P: {}/{} fingers".format(nf_n, nf_p))
            # Finding amplifier ss parameters
            inv = LTICircuit()
            inv.add_transistor(n_op_info, 'out', 'in', 'gnd', fg=nf_n)
            inv.add_transistor(p_op_info, 'out', 'in', 'gnd', fg=nf_p)
            inv_num, inv_den = inv.get_num_den(in_name='in', out_name='out', in_type='v')
            A0 = abs(inv_num[-1]/inv_den[-1]) 

            gds_n = n_op_info['gds'] * nf_n
            gm_n  = n_op_info['gm']  * nf_n            
            cgs_n = n_op_info['cgs'] * nf_n            
            cgd_n = n_op_info['cgd'] * nf_n            
            cds_n = n_op_info['cds'] * nf_n
            cgb_n = n_op_info.get('cgb', 0) * nf_n
            cdb_n = n_op_info.get('cdb', 0) * nf_n
            cdd_n = n_op_info['cdd'] * nf_n
            cgg_n = n_op_info['cgg'] * nf_n

            gds_p = p_op_info['gds'] * nf_p
            gm_p  = p_op_info['gm']  * nf_p                        
            cgs_p = p_op_info['cgs'] * nf_p       
            cgd_p = p_op_info['cgd'] * nf_p            
            cds_p = p_op_info['cds'] * nf_p
            cgb_p = p_op_info.get('cgb', 0) * nf_p
            cdb_p = p_op_info.get('cdb', 0) * nf_p
            cdd_p = p_op_info['cdd'] * nf_p
            cgg_p = p_op_info['cgg'] * nf_p

            gm = abs(gm_n) + abs(gm_p)
            gds = abs(gds_n) + abs(gds_p)
            ro = 1/gds
            cgs = cgs_n + cgs_p
            cds = cds_n + cds_p
            cgb = cgb_n + cgb_p
            cdb = cdb_n + cdb_p
            cgd = cgd_n + cgd_p
            cdd = cdd_n + cdd_p
            cgg = cgg_n + cgg_p

            # Assume Rdc is negative, bound Rf
            rf_min = max(rdc_min*(1+A0)/A0 + ro/A0, 0)
            rf_vec = np.arange(rf_min, rdc_min*5, rf_res)
            # Sweep values of Rf to check f3dB and PM spec
            for rf in rf_vec:
                # Circuit for GBW
                circuit = LTICircuit()
                ######################
                ### YOUR CODE HERE ###
                ######################
                # Determining if it meets spec
                num, den = circuit.get_num_den(in_name='in', out_name='out', in_type='i')
                rdc = num[-1]/den[-1]
                if abs(rdc) < rdc_min-1e-8:
                    print("RDC: {0:.2f} (FAIL)\n".format(rdc))
                    continue
                else:
                    print("RDC: {0:.2f}".format(rdc))
                fbw = get_w_3db(num, den)/(2*np.pi)
                if fbw < fbw_min or isnan(fbw):
                    print("BW: {} (FAIL)\n".format(fbw))
                    break   # Increasing Rf isn't going to help
                else:
                    print("BW: {}".format(fbw))

                # Circuit for phase margin
                # miller = (1-gm*rf)/(ro+rf)*ro
                circuit2 = LTICircuit()
                circuit2.add_conductance(gds, 'out', 'gnd')
                circuit2.add_cap(cgg+cpd, 'in', 'gnd')
                circuit2.add_cap(cdd+cload, 'out', 'gnd')
                circuit2.add_cap(cgd, 'in', 'out')
                circuit2.add_res(rf, 'in', 'out')
                loopBreak = circuit2.get_transfer_function(in_name='out', out_name='in', in_type='i')
                pm, gainm = get_stability_margins(loopBreak.num*gm, loopBreak.den)
                if pm < pm_min or isnan(pm):
                    print("PM: {} (FAIL)\n".format(pm))
                    continue
                else:
                    print("PM: {}\n".format(pm))
                if ibias_n*nf_n < best_ibias:
                    best_ibias = ibias_n*nf_n
                    best_op = dict(
                    vg=vg,
                    nf_n=nf_n,
                    nf_p=nf_p,
                    rf=rf,
                    rdc=rdc,
                    fbw=fbw,
                    pm=pm,
                    ibias=best_ibias)
                
    if best_op == None:
        raise ValueError("No solutions.")
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
        vg_res=0.01,
        rf_res=100,
        vdd=1.0,
        cpd=5e-15,
        cload=20e-15,
        rdc_min=1e3,
        fbw_min=5e9,
        pm_min=45,
        vb_n=0,
        vb_p=0
        )
    ### YOUR CODE HERE ###
    ### Uncomment out the appropriate line depending on which function you want to run ###
    # amp_specs = design_inverter_tia_eqn(**specs)
    # amp_specs = design_inverter_tia_lti(**specs)
    pprint.pprint(amp_specs)
    print('done')

if __name__ == '__main__':
    run_main()
