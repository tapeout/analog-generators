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
        db_n:       Database for NMOS device characterization data.
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
    
def design_preamp(db_n, db_p lch, w_finger,
        vdd, cload, vin_cm, voutcm_res, vtail_res, 
        vtrim_p, vtrim_n,
        vos, gain_min, bw_min,
        rload_res, 
        vb_p, vb_n, 
        itail_max=10e-6, error_tol=.01):
    """
    Designs a preamplifier to go between the comparator and TIA.
    Inputs:
        db_n/p:     Database with NMOS/PMOS info
        lch:        Float. Channel length (m)
        w_finger:   Float. Width of a single finger (m)
        vdd:        Float. Supply voltage (V)
        cload:      Float. Load capacitance (F)
        vin_cm:     Float. Input bias voltage (V)
        voutcm_res: Float. Resolution for output common mode voltage sweep
                    (V).
        vtail_res:  Float. Resolution for tail voltage of amplifying
                    and offset devices for sweep (V).
        vtrimp/n:   Float. Gate voltages of the trimming MOSFETs (V).
                    p is the higher of the two.
        vos:        Float. Intended peak-to-peak offset voltage at the output (V)
        gain_min:   Float. Minimum gain (V/V)
        bw_min:     Float. Minimum bandwidth (Hz)
        rload_res:  Float. Load resistor resolution for sweep (ohms)
        vb_n/p:     Float. Body/back-gate voltage (V) of NMOS and PMOS devices.
        itail_max:  Float. Max allowable tail current of the main amplifier (A).
        error_tol:  Float. Fractional tolerance for ibias error when computing
                    device ratios.
    
    Outputs:
        nf_in:          Integer. Number of fingers for input devices.
        nf_tail:        Integer. Number of fingers for tail device.
        nf_mirrn:       Integer. Number of fingers for mirror reference.
        nf_dum:         Integer. Number of fingers for dummy device.
        nf_in_trim:     Integer. Number of fingers for trimming devices.
        nf_tail_trim:   Integer. Number of fingers for tail device of
                        trimming element.
        itrim:          Float. Trimming current (A).
        itail:          Float. Tail current of main amp (A).
        rload:          Float. Load resistance (ohm).
        gain:           Float. Amp gain (V/V).
        fbw:            Float. Bandwidth (Hz).
        voutcm:         Float. Output common mode voltage (V).
        cin:            Float. Approximation of input cap (F).
    Notes:
        Under construction.
    """
    # Hardcoded/calculated relevant values
    itail_tol = 0.05
    vthn = estimate_vth(db_n, vdd, 0)
    
    voutcm_min = vin_cm-vthn
    voutcm_max = vdd
    voutcm_vec = np.arange(voutcm_min, voutcm_max, voutcm_res)
    
    vtail_min = .15 # TODO: Avoid hardcoding?
    vtail_max = vin_cm-vthn
    vtail_vec = np.arange(vtail_min, vtail_max, vtail_res)
    
    # Sweep output common mode voltage
    for voutcm in voutcm_vec:
        # Sweep tail voltage
        for vtail in vtail_vec:
            op_tail = db_n.query(vgs=voutcm,
                                vds=vtail,
                                vbs=vb_n)
            op_in = db_n.query(vgs=vin_cm-vtail,
                                vds=voutcm-vtail,
                                vbs=vb_n-vtail)
            op_trim_on = db_n.query(vgs=vtrimp-vtail,
                                    vds=voutcm-vtail,
                                    vbs=vb_n-vtail)
            op_trim_off = db_n.query(vgs=vtrimn-vtail,
                                    vds=voutcm-vtail,
                                    vbs=vb_n-vtail)
            
            # Calculate the max tail size based on current limit
            # and size input devices accordingly
            nf_tail_min = 2
            nf_tail_max = int(round(itail_max/op_tail['ibias'])
            nf_tail_vec = np.arange(nf_tail_min, nf_tail_max, 2)
            for nf_tail in nf_tail_vec:
                amp_ratio_good, nf_in = verify_ratio(op_tail['ibias'],
                                            op_in['ibias'],
                                            nf_tail, error_tol)
                if amp_ratio_good:
                    # Size offset devices
            
            
    
    ####################
    ####################
    # If there's no need for shifting or trim
    if vos_targ < 1e-6:
        vtail_trim = vout_cm
        op_trim_on = db_n.query(vgs=0, vds=0, vbs=vb_n-0)
        op_trim_off = db_n.query(vgs=0, vds=0, vbs=vb_n-0)
    # If there's need for shifting
    else:
        op_trim_on = db_n.query(vgs=vtrim_cm+vdiff_trim/2-vtail_trim, 
                                vds=vout_cm-vtail_trim, 
                                vbs=vb_n-0)
        op_trim_off = db_n.query(vgs=vtrim_cm-vdiff_trim/2-vtail_trim, 
                                vds=vout_cm-vtail_trim, 
                                vbs=vb_n-0)
        if vtrim_cm-vdiff_trim/2 < vtail_trim:
            print("WARNING: Trim gate voltage lower than source")
    
    vmirrn = vout_cm
    op_p_mirr = db_p.query(vgs=vmirrp-vdd, vds=vmirrn-vdd, vbs=0)
    iref = op_p_mirr['ibias']*nf_p_mirr

    # Best operating point
    best_ibias = np.inf
    best_op = None
        
    vtail_amp_min = .1
    vtail_amp_max = min(vout_cm, vin_cm-vthn)
    vtail_amp_vec = np.arange(vtail_amp_min, vtail_amp_max, 0.01)

    # Sweep tail voltage
    for vtail_amp in vtail_amp_vec:
        # Getting device characterization data
        op_tail = db_n.query(vgs=vmirrn, vds=vtail_amp, vbs=0)
        op_in = db_n.query(vgs=vin_cm-vtail_amp, vds=vout_cm-vtail_amp, vbs=0)
        op_mirr = op_tail
        op_dummy = op_in
        
        # Stop when self-loading starts to dominate
        nf_in_vec = np.arange(1, 50, 1)
        # int(round(cload/op_in['cdd'])*2
        # Quick sanity check
        if op_in['ibias'] < 0:
            continue
        
        # Size mirror NMOS
        nf_mirr = round(iref/op_mirr['ibias'])
        if nf_mirr < 1:
            continue
        if abs(nf_mirr*op_mirr['ibias']-iref)/iref > itail_tol:
            continue
        
        # Sizing the dummy device atop the mirror for matching
        nf_dummy = max(1, round(iref/op_dummy['ibias']))

        for nf_in in nf_in_vec:
            cond_print("\nNF,IN: {}".format(nf_in), debugMode)
            cout = cload + op_in['cdd']*nf_in # Underestimates, but it's okay
            gm_in = op_in['gm']*nf_in
            
            # Constrain rload to meet bandwidth, min/max gain
            rload_max = 1/(2*np.pi*bw_min*cout)
            rload_min = gain_min/gm_in
            
            itail = 2*op_in['ibias']*nf_in
            
            nf_tail = round(itail/op_tail['ibias'])
            cond_print("NF,TAIL: {}".format(nf_tail), debugMode)
            if abs(nf_tail*op_tail['ibias'] - itail)/itail > itail_tol:
                cond_print("Current tolerance off", debugMode)
                continue
            rload = (vdd-vout_cm)/(itail/2)
            cond_print("RLOAD: {}".format(rload), debugMode)
            if rload < rload_min and rload > rload_max:
                cond_print("RLOAD {} out of range: {} to {}".format(rload, rload_min, rload_max), debugMode)
                continue
            
            cond_print("ITAIL: {}".format(itail), debugMode)
            # cond_print("INPUT DEVICE CURRENT: {}".format(op_in['ibias']))
            
            # Include trimming devices
            idiff_trim_base = abs(op_trim_on['ibias']-op_trim_off['ibias'])
            if idiff_trim_base == 0:
                nf_trim = 1
                itrim = 0
            else:
                vos_base = idiff_trim_base*rload
                nf_trim = ceil(vos_targ/vos_base)
                itrim = (op_trim_on['ibias']+op_trim_off['ibias'])*nf_trim

            # LTICircuit simulation
            circuit = LTICircuit()
            circuit.add_transistor(op_in, 'outp', 'gnd', 'tail', fg=nf_in)
            circuit.add_transistor(op_in, 'outn', 'inp', 'tail', fg=nf_in)
            circuit.add_transistor(op_trim_on, 'outn', 'gnd', 'gnd', fg=nf_trim)
            circuit.add_transistor(op_trim_off, 'outp', 'gnd', 'gnd', fg=nf_trim)
            circuit.add_transistor(op_tail, 'tail', 'vmirr', 'gnd', fg=nf_tail)
            circuit.add_transistor(op_mirr, 'vmirr', 'vmirr', 'gnd', fg=nf_mirr)
            circuit.add_transistor(op_dummy, 'gnd', 'gnd', 'vmirr', fg=nf_dummy)
            circuit.add_res(rload, 'outp', 'gnd')
            circuit.add_res(rload, 'outn', 'gnd')
            circuit.add_cap(cload, 'outp', 'gnd')
            circuit.add_cap(cload, 'outn', 'gnd')
            circuit.add_cap(dac_cap, 'tail', 'gnd')
            num, den = circuit.get_num_den(in_name='inp', out_name='outn', in_type='v')
            num_unintentional, den_unintentional = circuit.get_num_den(in_name='inp', out_name='outp', in_type='v')
            gain_intentional = num[-1]/den[-1]
            gain_unintentional = num_unintentional[-1]/den_unintentional[-1]
            gain = abs(gain_intentional-gain_unintentional)
            wbw = get_w_3db(num, den)
            if wbw == None:
                wbw = 0
            fbw = wbw/(2*np.pi)
            if fbw < bw_min:
                cond_print("Bandwidth fail: {}".format(fbw), debugMode)
                cond_print("NF,IN: {}".format(nf_in), debugMode)
                cond_print("NF,TAIL: {}".format(nf_tail), debugMode)
                cond_print("NF,MIRR: {}".format(nf_mirr), debugMode)
                cond_print("NF,DUMMY: {}".format(nf_dummy), debugMode)
                continue
            elif gain < gain_min:
                cond_print("Gain Low: {}".format(gain), debugMode)
                continue
            else:
                cond_print("(SUCCESS)")
                if itail < best_ibias:
                    best_ibias = itail
                    best_op = dict(nf_dummy=nf_dummy,
                                   nf_in=nf_in,
                                   nf_mirr=nf_mirr,
                                   nf_tail=nf_tail,
                                   nf_trim=nf_trim,
                                   rload=rload,
                                   itrim=itrim,
                                   itail=itail,
                                   gain=gain,
                                   f3dB=fbw,
                                   vtail_amp=vtail_amp)
    if best_op == None:
        raise ValueError("No solution")
    return best_op
