# -*- coding: utf-8 -*-

import numpy as np

from bag.util.search import FloatBinaryIterator
from bag.data.lti import LTICircuit, get_w_3db, get_stability_margins
from helper_funcs import verify_ratio, cond_print
from math import isnan

def design_LPF_AMP(db_n, db_p, db_bias, sim_env,
    vin, vdd_nom, cload,
    vtail_res,
    gain_min, fbw_min, pm_min,
    vb_n, vb_p, error_tol=0.1, ibias_max=20e-6, debugMode=False):
    '''
    Designs an amplifier with an N-input differential pair.
    Uses the LTICircuit functionality.
    Inputs:
        db_n/p:     Databases for non-biasing NMOS and PMOS device 
                    characterization data, respectively.
        db_bias:    Database for tail NMOS device characterization data.
        sim_env:    Simulation corner.
        vin:        Float. Input (and output and tail) bias voltage in volts.
        vtail_res:  Float. Step resolution in volts when sweeping tail voltage.
        vdd_nom:    Float. Nominal supply voltage in volts.
        cload:      Float. Output load capacitance in farads.
        gain_min:   Float. Minimum DC voltage gain in V/V.
        fbw_min:    Float. Minimum bandwidth (Hz) of open loop amp.
        pm_min:     Float. Minimum phase margin in degrees.
        vb_n/p:     Float. Back-gate/body voltage (V) of NMOS and PMOS, 
                    respectively (nominal).
        error_tol:  Float. Fractional tolerance for ibias error when computing 
                    the p-to-n ratio.
        ibias_max:  Float. Maximum bias current (A) allowed with nominal vdd.
    Raises:
        ValueError: If unable to meet the specification requirements.
    Outputs:
        A dictionary with the following key:value pairings:
        nf_n:
        nf_p:
        nf_tail:
        gain:           Float. DC voltage gain of both stages combined (V/V).
        fbw:            Float. Bandwdith (Hz).
        pm:             Float. Phase margin (degrees) for unity gain.
    '''
    possibilities = []
    vout = vin
    vstar_min = 0.15
    vtail_vec = np.arange(vstar_min, vout, vtail_res)
    for vtail in vtail_vec:
        cond_print("VTAIL: {0}".format(vtail), debugMode)
        n_op = db_n.query(vgs=vin-vtail, vds=vout-vtail, vbs=vb_n-vtail)
        p_op = db_p.query(vgs=vout-vdd_nom, vds=vout-vdd_nom, vbs=vb_p-vdd_nom)
        tail_op = db_bias.query(vgs=vout, vds=vtail, vbs=vb_n)
        
        idn_base = n_op['ibias']
        idp_base = p_op['ibias']
        idtail_base = tail_op['ibias']
        
        p_to_n = abs(idn_base/idp_base)
        tail_to_n = abs(idn_base/idtail_base)
        
        nf_n_max = int(round(abs(ibias_max/(idn_base*2))))
        nf_n_vec = np.arange(2, nf_n_max, 2)
        for nf_n in nf_n_vec:
            cond_print("\tNF_N: {0}".format(nf_n), debugMode)
            # Verify that sizing is feasible and gets sufficiently
            # close current matching
            p_good, nf_p = verify_ratio(idn_base, idp_base, p_to_n, 
                nf_n, error_tol)
                
            tail_good, nf_tail = verify_ratio(idn_base, idtail_base, tail_to_n,
                nf_n*2, error_tol)

            if not (p_good and tail_good):
                cond_print("\t\tP_BIAS: {0}\n\t\tT_BIAS: {1}".format(p_good, tail_good), debugMode)
                cond_print("\t\tP_SIZE: {0}\n\t\tT_SIZE: {1}".format(nf_p, nf_tail), debugMode)
                tail_error = abs(abs(idtail_base*nf_tail) - abs(idn_base*nf_n*2))/abs(idn_base*nf_n*2)
                cond_print("\t\tTAIL ERROR: {0}".format(tail_error), debugMode)
                continue
            # Devices are sized, check open loop amp SS spec
            openLoop_good, openLoop_params = verify_openLoop(n_op, p_op, tail_op, 
                                        nf_n, nf_p, nf_tail,
                                        gain_min, fbw_min,
                                        cload)
            if not openLoop_good:
                # Gain is set by the bias point
                if openLoop_params['gain'] < gain_min:
                    cond_print("\t\tGAIN: {0} (FAIL)".format(openLoop_params['gain']), debugMode)
                    break
                
                cond_print("\t\tBW: {0} (FAIL)".format(openLoop_params['fbw']), debugMode)
                # Bandwidth isn't strictly set by biasing
                continue
            
            # Check PM in feedback
            closedLoop_good, closedLoop_params = verify_closedLoop(n_op, p_op, tail_op,
                                        nf_n, nf_p, nf_tail, pm_min, 
                                        cload)
            if closedLoop_good:
                viable = dict(nf_n=nf_n,
                              nf_p=nf_p,
                              nf_tail=nf_tail,
                              gain=openLoop_params['gain'],
                              fbw=openLoop_params['fbw'],
                              pm=closedLoop_params['pm'],
                              ibias=abs(idtail_base*nf_tail),
                              vtail=vtail)
                possibilities.append(viable)
    if len(possibilities) == 0:
        return ValueError("No viable solutions.")
    else:
        print("{0} viable solutions".format(len(possibilities)))
        
    best_ibias = float('inf')
    best_op = None
    for candidate in possibilities:
        if candidate['ibias'] < best_ibias:
            best_op = candidate
            best_ibias = candidate['ibias']
    
    return best_op
            
def construct_openLoop(n_op, p_op, tail_op, 
    nf_n, nf_p, nf_tail,
    cload):
    '''
    Inputs:
        p/n/tail_op:    Operating point information for a given device.
                        NMOS (input), PMOS (active load), or tail device.
        nf_n/p/tail:    Integer. Number of minimum channel width/length
                        devices in parallel.
        cload:          Float. Load capacitance in farads.
    Outputs:
        Returns the LTICircuit constructed for open loop analysis of 
        the amplifier.
    '''
    ckt = LTICircuit()
    # Left side
    ckt.add_transistor(n_op, 'out_copy', 'in', 'tail', fg=nf_n)
    ckt.add_transistor(p_op, 'out_copy', 'out_copy', 'gnd', fg=nf_p)
    
    # Right side
    ckt.add_transistor(n_op, 'out', 'gnd', 'tail', fg=nf_n)
    ckt.add_transistor(p_op, 'out', 'out_copy', 'gnd', fg=nf_p)
    
    # Tail
    ckt.add_transistor(tail_op, 'tail', 'gnd', 'gnd', fg=nf_tail)
    
    # Adding additional load
    ckt.add_cap(cload, 'out', 'gnd')
    
    return ckt
    
def construct_feedback(n_op, p_op, tail_op,
    nf_n, nf_p, nf_tail,
    cload, cn=165e-15, cp=83e-15, r=70.71e3):
    '''
    Inputs:
        p/n/tail_op:    Operating point information for a given device.
                        NMOS (input), PMOS (active load), or tail device.
        nf_n/p/tail:    Integer. Number of minimum channel width/length
                        devices in parallel.
        cload:          Float. Load capacitance in farads.
        cn:             Float. Capacitance in farads attached to the negative input
                        terminal of the amplifier for the filter.
        cp:             Float. Capacitance in farads attached to the positive input
                        terminal of the amplifier for the filter.
        r:              Float. Resistance in ohms used external to the amplifier
                        for the filter.
    Outputs:
        Returns the LTICircuit constructed for closed-loop analysis of 
        the amplifier with the loop broken.
    '''
    ckt = LTICircuit()
    # Left side
    ckt.add_transistor(n_op, 'out_copy', 'inp', 'tail', fg=nf_n)
    ckt.add_transistor(p_op, 'out_copy', 'out_copy', 'gnd', fg=nf_p)
    
    # Right side
    ckt.add_transistor(n_op, 'out', 'inn', 'tail', fg=nf_n)
    ckt.add_transistor(p_op, 'out', 'out_copy', 'gnd', fg=nf_p)
    
    # Tail
    ckt.add_transistor(tail_op, 'tail', 'gnd', 'gnd', fg=nf_tail)
    
    # Adding additional passives
    ckt.add_cap(cload, 'out', 'gnd')
    ckt.add_cap(cn, 'inn', 'rmid')
    ckt.add_cap(cp, 'inp', 'gnd')
    ckt.add_res(r, 'in', 'rmid')
    ckt.add_res(r, 'rmid', 'inp')
    
    return ckt
    
def verify_openLoop(n_op, p_op, tail_op, nf_n, nf_p, nf_tail,
    gain_min, fbw_min, cload):
    '''
    Inputs:
    Outputs:
    '''
    ckt = construct_openLoop(n_op, p_op, tail_op, nf_n, nf_p, nf_tail, cload)
            
    num, den = ckt.get_num_den(in_name='in', out_name='out', in_type='v')
    gain = abs(num[-1]/den[-1])
    # Check gain
    if gain < gain_min:
        # Biasing sets the gain
        return False, dict(gain=gain, fbw=0)

    # Check bandwidth
    fbw = get_w_3db(num, den)/(2*np.pi)
    if fbw < fbw_min:
        return False, dict(gain=gain, fbw=fbw)
    return True, dict(gain=gain, fbw=fbw)
    
def verify_closedLoop(n_op, p_op, tail_op,
    nf_n, nf_p, nf_tail, pm_min,
    cload, cn=165e-15, cp=83e-15, r=70.71e3):
    '''
    Inputs:
    Outputs:
    '''
    ckt = construct_feedback(n_op, p_op, tail_op,
                nf_n, nf_p, nf_tail,
                cload)
            
    loopBreak = ckt.get_transfer_function(in_name='inn', out_name='out', 
                                        in_type='v')
                                        
    pm, gainm = get_stability_margins(loopBreak.num, loopBreak.den)
    if pm < pm_min or isnan(pm):
        return False, dict(pm=pm)
    return True, dict(pm=pm)
