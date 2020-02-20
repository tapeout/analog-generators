# -*- coding: utf-8 -*-

import numpy as np

from bag.util.search import FloatBinaryIterator
from bag.data.lti import LTICircuit, get_w_3db, get_stability_margins
from helper_funcs import verify_ratio, estimate_vth

def design_preamp_chain(db_n, db_p, lch, w_finger,
    vdd, cload, vin_signal, vin_shift, vtail_res,
    gain_min, fbw_min,
    vb_p, vb_n,
    itail_max=10e-6, error_tol=0.01):
    """
    Designs a preamplifier to go between the comparator and TIA.
    N-input differential pair with an active PMOS load.
    Inputs:
        db_n/p:     Database with NMOS/PMOS info
        lch:        Float. Channel length (m)
        w_finger:   Float. Width of a single finger (m)
        vdd:        Float. Supply voltage (V)
        cload:      Float. Load capacitance (F)
        vin_signal: Float. Input bias voltage (V).
        vin_shift:  Float. Gate bias voltage for the non-signal-facing input
                    device (V).
        vtail_res:  Float. Resolution for tail voltage of amplifying
                    and offset devices for sweep (V).
        gain_min:   Float. Minimum gain (V/V)
        fbw_min:    Float. Minimum bandwidth (Hz)
        vb_n/p:     Float. Body/back-gate voltage (V) of NMOS and PMOS devices.
        itail_max:  Float. Max allowable tail current of the main amplifier (A).
        error_tol:  Float. Fractional tolerance for ibias error when computing
                    device ratios.
    Raises:
        ValueError if there is no solution given the requirements.
    Outputs:
        nf_in:          Integer. Number of fingers for input devices.
        nf_tail:        Integer. Number of fingers for tail device.
        nf_load:        Integer. Number of fingers for active load devices.
        itail:          Float. Tail current of main amp (A).
        gain:           Float. Amp gain (V/V).
        fbw:            Float. Bandwidth (Hz).
        cin:            Float. Approximation of input cap (F).
        vtail:          Float. Tail voltage (V).
    """
    vincm = (vin_signal + vin_shift)/2
    voutcm = vincm
    best_op = None
    best_itail = np.inf
    
    vthn = estimate_vth(db_n, vdd, 0)
    
    vtail_min = 0.15 # TODO: Avoid hardcoding?
    vtail_max = vincm
    vtail_vec = np.arange(vtail_min, vtail_max, vtail_res)
    for vtail in vtail_vec:
        print('VTAIL:\t{0}'.format(vtail))
        op_tail = db_n.query(vgs=voutcm, vds=vtail, vbs=vb_n)
        op_in = db_n.query(vgs=vincm-vtail, vds=voutcm-vtail, vbs=vb_n-vtail)
        op_load = db_p.query(vgs=voutcm-vdd, vds=voutcm-vdd, vbs=vb_p-vdd)
        
        nf_in_min = 2
        nf_in_max = min(100, int(round(.5 * itail_max/op_in['ibias'])))
        nf_in_vec = np.arange(nf_in_min, nf_in_max, 2)
        for nf_in in nf_in_vec:
            
            # Check if those bias points are feasible given the 
            # device sizing quantization
            load_ratio_good, nf_load = verify_ratio(op_in['ibias'], 
                                                op_load['ibias'], 
                                                nf_in,
                                                error_tol)
            if not load_ratio_good:
                continue
            
            tail_ratio_good, nf_tail_half = verify_ratio(op_in['ibias'],
                                                    op_tail['ibias'],
                                                    nf_in,
                                                    error_tol)
            nf_tail = nf_tail_half * 2
            
            if not tail_ratio_good:
                continue
            
            # Check if it's burning more power than previous solutions
            itail = op_tail['ibias'] * nf_tail
            if itail > best_itail:
                break
            
            # Check the half circuit for small signal parameters
            half_circuit = LTICircuit()
            half_circuit.add_transistor(op_in, 'out', 'in', 'tail', fg=nf_in)
            half_circuit.add_transistor(op_load, 'out', 'gnd', 'gnd', fg=nf_load)
            half_circuit.add_transistor(op_tail, 'tail', 'gnd', 'gnd', fg=nf_tail_half)
            half_circuit.add_cap(cload, 'out', 'gnd')
            num, den = half_circuit.get_num_den(in_name='in', out_name='out',
                                                in_type='v')
            gain = abs(num[-1]/den[-1])
            if gain < gain_min:
                print("(FAIL) GAIN:\t{0}".format(gain))
                break
            
            wbw = get_w_3db(num,den)
            if wbw == None:
                wbw = 0
            fbw = wbw/(2*np.pi)
            if fbw < fbw_min:
                print("(FAIL) BW:\t{0}".format(fbw))
                continue
            
            cin = nf_in*(op_in['cgs'] + op_in['cgd']*(1+gain))
            
            best_itail = itail
            best_op = dict(nf_in=nf_in,
                            nf_tail=nf_tail,
                            nf_load=nf_load,
                            itail=itail,
                            gain=gain,
                            fbw=fbw,
                            vtail=vtail,
                            cin=cin)
            
    if best_op == None:
        raise ValueError("No viable solutions.")
    return best_op
            
    
   
def design_preamp(db_n, db_p, lch, w_finger,
    vdd, cload, vin_signal, vin_shift, voutcm_res, vtail_res,
    gain_min, fbw_min,
    vb_p, vb_n,
    itail_max=10e-6, error_tol=.01):
    """
    Designs a preamplifier to go between the comparator and TIA.
    N-input differential pair with an active PMOS load.
    Inputs:
        db_n/p:     Database with NMOS/PMOS info
        lch:        Float. Channel length (m)
        w_finger:   Float. Width of a single finger (m)
        vdd:        Float. Supply voltage (V)
        cload:      Float. Load capacitance (F)
        vin_signal: Float. Input bias voltage (V).
        vin_shift:  Float. Gate bias voltage for the non-signal-facing input
                    device (V).
        voutcm_res: Float. Resolution for output common mode voltage sweep
                    (V).
        vtail_res:  Float. Resolution for tail voltage of amplifying
                    and offset devices for sweep (V).
        gain_min:   Float. Minimum gain (V/V)
        fbw_min:    Float. Minimum bandwidth (Hz)
        vb_n/p:     Float. Body/back-gate voltage (V) of NMOS and PMOS devices.
        itail_max:  Float. Max allowable tail current of the main amplifier (A).
        error_tol:  Float. Fractional tolerance for ibias error when computing
                    device ratios.
    Raises:
        ValueError if there is no solution given the requirements.
    Outputs:
        nf_in:          Integer. Number of fingers for input devices.
        nf_tail:        Integer. Number of fingers for tail device.
        nf_load:        Integer. Number of fingers for active load devices.
        itail:          Float. Tail current of main amp (A).
        gain:           Float. Amp gain (V/V).
        fbw:            Float. Bandwidth (Hz).
        voutcm:         Float. Output common mode voltage (V).
        cin:            Float. Approximation of input cap (F).
    """
    vthn = estimate_vth(db_n, vdd, 0)
    vthp = estimate_vth(db_p, vdd, 0, mos_type="pmos")
    vincm = (vin_signal + vin_shift)/2
    
    vstar_min = .15 # TODO: Avoid hardcoding?
    
    vtail_min = vstar_min
    vtail_max = vin_signal - vstar_min
    vtail_vec = np.arange(vtail_min, vtail_max, vtail_res)
    
    best_op = None
    best_itail = np.inf
    
    for vtail in vtail_vec:
        voutcm_min = max(vin_signal-vthn, vin_shift-vthn, vtail)
        voutcm_max = vdd-vstar_min
        voutcm_vec = np.arange(voutcm_min, voutcm_max, voutcm_res)
        
        for voutcm in voutcm_vec:
            op_tail = db_n.query(vgs=voutcm, vds=vtail, vbs=vb_n)
            # The "mean" of the input and load devices
            op_in = db_n.query(vgs=vincm-vtail,
                                vds=voutcm-vtail,
                                vbs=vb_n-vtail)
            op_load = db_p.query(vgs=vtail-vdd, vds=voutcm-vdd, vbs=vb_p-vdd)
            
            # Calculate the max tail size based on current limit
            # and size input devices accordingly
            nf_in_min = 2
            nf_in_max = int(round(itail_max/2 / op_in['ibias']))
            nf_in_vec = np.arange(nf_in_min, nf_in_max, 1)
            for nf_in in nf_in_vec:
                # Matching device ratios to sink the same amount of current
                # given the bias voltages
                tail_ratio_good, nf_tail = verify_ratio(op_in['ibias']/2,
                                            op_tail['ibias'],
                                            nf_in, error_tol)
                if not tail_ratio_good:
                    continue
                    
                itail = op_tail['ibias']*nf_tail
                
                load_ratio_good, nf_load = verify_ratio(op_in['ibias'],
                                            op_load['ibias'],
                                            nf_in, error_tol)
                
                if not load_ratio_good:
                    continue
                
                # Check small signal parameters with symmetric circuit
                ckt_sym = LTICircuit()
                ckt_sym.add_transistor(op_in, 'outp', 'gnd', 'tail', fg=nf_in)
                ckt_sym.add_transistor(op_in, 'outn', 'inp', 'tail', fg=nf_in)
                ckt_sym.add_transistor(op_tail, 'tail', 'gnd', 'gnd', fg=nf_tail)
                ckt_sym.add_transistor(op_load, 'outp', 'gnd', 'gnd', fg=nf_load)
                ckt_sym.add_transistor(op_load, 'outn', 'gnd', 'gnd', fg=nf_load)
                ckt_sym.add_cap(cload, 'outp', 'gnd')
                ckt_sym.add_cap(cload, 'outn', 'gnd')
                num, den = ckt_sym.get_num_den(in_name='inp', 
                                            out_name='outn', 
                                            in_type='v')
                num_unintent, den_unintent = ckt_sym.get_num_den(in_name='inp', 
                                                                out_name='outp', 
                                                                in_type='v')
                gain_intentional = num[-1]/den[-1]
                gain_unintentional = num_unintent[-1]/den_unintent[-1]
                gain = abs(gain_intentional - gain_unintentional)
                wbw = get_w_3db(num,den)
                if wbw == None:
                    wbw = 0
                fbw = wbw/(2*np.pi)
                if fbw < fbw_min:
                    # print("(FAIL) BW:\t{0}".format(fbw))
                    continue
                if gain < gain_min:
                    # print("(FAIL) GAIN:\t{0}".format(gain))
                    break
                print("(SUCCESS1)")
                
                if itail > best_itail:
                    break
                
                ##############################
                if False:
                    # Check once again with asymmetric circuit
                    vin_diff = vin_signal - vin_shift
                    voutn = voutcm - gain*vin_diff/2
                    voutp = voutcm + gain*vin_diff/2
                    op_signal = db_n.query(vgs=vin_signal-vtail,
                                            vds=voutn-vtail,
                                            vbs=vb_n-vtail)
                    op_shift = db_n.query(vgs=vin_shift-vtail,
                                            vds=voutp-vtail,
                                            vbs=vb_n-vtail)
                    op_loadsignal = db_p.query(vgs=vtail-vdd,
                                                vds=voutn-vdd,
                                                vbs=vb_p-vdd)
                    op_loadshift = db_p.query(vgs=vtail-vdd,
                                                vds=voutp-vdd,
                                                vbs=vb_p-vdd)
                    ckt = LTICircuit()
                    ckt.add_transistor(op_shift, 'outp', 'gnd', 'tail', fg=nf_in)
                    ckt.add_transistor(op_signal, 'outn', 'inp', 'tail', fg=nf_in)
                    ckt.add_transistor(op_tail, 'tail', 'gnd', 'gnd', fg=nf_tail)
                    ckt.add_transistor(op_loadsignal, 'outn', 'gnd', 'gnd', nf_load)
                    ckt.add_transistor(op_loadshift, 'outp', 'gnd', 'gnd', nf_load)
                    ckt.add_cap(cload, 'outn', 'gnd')
                    ckt.add_cap(cload, 'outp', 'gnd')
                    num, den = ckt.get_num_den(in_name='inp', 
                                                out_name='outn', 
                                                in_type='v')
                    num_unintent, den_unintent = ckt.get_num_den(in_name='inp',
                                                                out_name='outp',
                                                                in_type='v')
                    gain_intentional = num[-1]/den[-1]
                    gain_unintentional = num_unintent[-1]/den_unintent[-1]
                    gain = abs(gain_intentional - gain_unintentional)
                    
                    wbw = get_w_3db(num,den)
                    if wbw == None:
                        wbw = 0
                    fbw = wbw/(2*np.pi)
                    if fbw < fbw_min:
                        print("(FAIL) BW:\t{0}".format(fbw))
                        continue
                    if gain < gain_min:
                        print("(FAIL) GAIN:\t{0}".format(gain))
                        break
                    print("(SUCCESS2)")
                ################################
                op_signal = op_in
                
                if itail < best_itail:
                    best_itail = itail
                    
                    best_op = dict(nf_in=nf_in,
                                    nf_tail=nf_tail,
                                    nf_load=nf_load,
                                    itail=itail,
                                    gain=gain,
                                    fbw=fbw,
                                    voutcm=voutcm,
                                    vtail=vtail,
                                    cin=op_signal['cgg']*nf_in)
    if best_op == None:
        raise ValueError("No viable solutions.")
    return best_op
