# -*- coding: utf-8 -*-

import numpy as np

from bag.util.search import FloatBinaryIterator
from bag.data.lti import LTICircuit, get_w_3db, get_stability_margins
from helper_funcs import verify_ratio, estimate_vth

def design_preamp_string(db_n, lch, w_finger,
        vdd, cload, vin_signal, vin_shift, voutcm, vtail_res,
        gain_min, fbw_min,
        rload_res,
        vb_n, itail_max=10e-6, error_tol=.01):
    """
    Designs a preamplifier to go between the comparator and TIA, stringing them
    together.
    Inputs:
    Outputs:
    """
    # Estimate vth for use in calculating biasing ranges
    vthn = estimate_vth(db_n, vdd, 0)
    vincm = (vin_signal + vin_shift)/2
    
    vtail_min = .15 # TODO: Avoid hardcoding?
    vtail_max = vin_signal-vthn
    vtail_vec = np.arange(vtail_min, vtail_max, vtail_res)
    
    best_op = None
    best_itail = np.inf
    
    for vtail in vtail_vec:
        op_tail = db_n.query(vgs=voutcm,
                            vds=vtail,
                            vbs=vb_n)
        # The "mean" of the input devices
        op_in = db_n.query(vgs=vincm-vtail,
                            vds=voutcm-vtail,
                            vbs=vb_n-vtail)
        
        # Calculate the max tail size based on current limit
        # and size input devices accordingly
        nf_in_min = 2
        nf_in_max = int(round(itail_max/2 / op_in['ibias']))
        nf_in_vec = np.arange(nf_in_min, nf_in_max, 1)
        for nf_in in nf_in_vec:
            amp_ratio_good, nf_tail = verify_ratio(op_in['ibias']/2,
                                        op_tail['ibias'],
                                        nf_in, error_tol)
            if not amp_ratio_good:
                continue
            
            itail = op_tail['ibias']*nf_tail
            rload = (vdd-voutcm)/(itail/2)
            
            # Input devices too small
            if op_in['gm']*nf_in*rload < gain_min:
                continue
            
            # Check small signal parameters with symmetric circuit
            ckt_sym = LTICircuit()
            ckt_sym.add_transistor(op_in, 'outp', 'gnd', 'tail', fg=nf_in)
            ckt_sym.add_transistor(op_in, 'outn', 'inp', 'tail', fg=nf_in)
            ckt_sym.add_transistor(op_tail, 'tail', 'gnd', 'gnd', fg=nf_tail)
            ckt_sym.add_res(rload, 'outp', 'gnd')
            ckt_sym.add_res(rload, 'outn', 'gnd')
            ckt_sym.add_cap(cload, 'outp', 'gnd')
            ckt_sym.add_cap(cload, 'outn', 'gnd')
            num, den = ckt_sym.get_num_den(in_name='inp', out_name='outn', in_type='v')
            num_unintent, den_unintent = ckt_sym.get_num_den(in_name='inp', out_name='outp', in_type='v')
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
                continue
                
            print("(SUCCESS)")
            if itail > best_itail:
                continue
            
            # Check once again with asymmetric circuit
            vin_diff = vin_signal-vin_shift
            voutn = voutcm - gain*vin_diff/2
            voutp = voutcm + gain*vin_diff/2
            op_signal = db_n.query(vgs=vin_signal-vtail,
                                    vds=voutn-vtail,
                                    vbs=vb_n-vtail)
            op_shift = db_n.query(vgs=vin_shift-vtail,
                                    vds=voutp-vtail,
                                    vbs=vb_n-vtail)
            ckt = LTICircuit()
            ckt.add_transistor(op_shift, 'outp', 'gnd', 'tail', fg=nf_in)
            ckt.add_transistor(op_signal, 'outn', 'inp', 'tail', fg=nf_in)
            ckt.add_transistor(op_tail, 'tail', 'gnd', 'gnd', fg=nf_tail)
            ckt.add_res(rload, 'outp', 'gnd')
            ckt.add_res(rload, 'outn', 'gnd')
            ckt.add_cap(cload, 'outp', 'gnd')
            ckt.add_cap(cload, 'outn', 'gnd')
            num, den = ckt.get_num_den(in_name='inp', out_name='outn', in_type='v')
            num_unintent, den_unintent = ckt.get_num_den(in_name='inp', out_name='outp', in_type='v')
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
                continue
            
            print("(SUCCESS)")
            if itail < best_itail:
                best_itail = itail
                best_op = dict(nf_in=nf_in,
                                nf_tail=nf_tail,
                                itail=itail,
                                rload=rload,
                                gain=gain,
                                fbw=fbw,
                                voutcm=voutcm,
                                vtail=vtail,
                                cin=op_signal['cgg']*nf_in)
    if best_op == None:
        raise ValueError("No viable solutions.")
    return best_op
                

def design_preamp(db_n, db_p, lch, w_finger,
        vdd, cload, vin_signal, vin_shift, voutcm_res, vtail_res, 
        gain_min, fbw_min,
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
        vin_signal: Float. Input bias voltage (V).
        vin_shift:  Float. Gate bias voltage for the non-signal-facing input
                    device (V).
        voutcm_res: Float. Resolution for output common mode voltage sweep
                    (V).
        vtail_res:  Float. Resolution for tail voltage of amplifying
                    and offset devices for sweep (V).
        gain_min:   Float. Minimum gain (V/V)
        fbw_min:    Float. Minimum bandwidth (Hz)
        rload_res:  Float. Load resistor resolution for sweep (ohms)
        vb_n/p:     Float. Body/back-gate voltage (V) of NMOS and PMOS devices.
        itail_max:  Float. Max allowable tail current of the main amplifier (A).
        error_tol:  Float. Fractional tolerance for ibias error when computing
                    device ratios.
    Raises:
        ValueError if there is no solution given the requirements.
    Outputs:
        nf_in:          Integer. Number of fingers for input devices.
        nf_tail:        Integer. Number of fingers for tail device.
        itail:          Float. Tail current of main amp (A).
        rload:          Float. Load resistance (ohm).
        gain:           Float. Amp gain (V/V).
        fbw:            Float. Bandwidth (Hz).
        voutcm:         Float. Output common mode voltage (V).
        cin:            Float. Approximation of input cap (F).
    Notes:
        Under construction.
    """
    # Estimate vth for use in calculating biasing ranges
    vthn = estimate_vth(db_n, vdd, 0)
    vincm = (vin_signal + vin_shift)/2
    
    voutcm_min = vin_signal-vthn
    voutcm_max = vdd
    voutcm_vec = np.arange(voutcm_min, voutcm_max, voutcm_res)
    
    vtail_min = .15 # TODO: Avoid hardcoding?
    vtail_max = vin_signal-vthn
    vtail_vec = np.arange(vtail_min, vtail_max, vtail_res)
    
    best_op = None
    best_itail = np.inf
    
    # Sweep output common mode voltage
    for voutcm in voutcm_vec:
        # Sweep tail voltage
        for vtail in vtail_vec:
            op_tail = db_n.query(vgs=voutcm,
                                vds=vtail,
                                vbs=vb_n)
            # The "mean" of the input devices
            op_in = db_n.query(vgs=vincm-vtail,
                                vds=voutcm-vtail,
                                vbs=vb_n-vtail)
            
            # Calculate the max tail size based on current limit
            # and size input devices accordingly
            nf_in_min = 2
            nf_in_max = int(round(itail_max/2 / op_in['ibias']))
            nf_in_vec = np.arange(nf_in_min, nf_in_max, 2)
            for nf_in in nf_in_vec:
                amp_ratio_good, nf_tail = verify_ratio(op_in['ibias']/2,
                                            op_tail['ibias'],
                                            nf_in, error_tol)
                if not amp_ratio_good:
                    continue
                
                itail = op_tail['ibias']*nf_tail
                rload = (vdd-voutcm)/(itail/2)
                
                # Check small signal parameters with symmetric circuit
                ckt_sym = LTICircuit()
                ckt_sym.add_transistor(op_in, 'outp', 'gnd', 'tail', fg=nf_in)
                ckt_sym.add_transistor(op_in, 'outn', 'inp', 'tail', fg=nf_in)
                ckt_sym.add_transistor(op_tail, 'tail', 'gnd', 'gnd', fg=nf_tail)
                ckt_sym.add_res(rload, 'outp', 'gnd')
                ckt_sym.add_res(rload, 'outn', 'gnd')
                ckt_sym.add_cap(cload, 'outp', 'gnd')
                ckt_sym.add_cap(cload, 'outn', 'gnd')
                num, den = ckt_sym.get_num_den(in_name='inp', out_name='outn', in_type='v')
                num_unintent, den_unintent = ckt_sym.get_num_den(in_name='inp', out_name='outp', in_type='v')
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
                    continue
                    
                print("(SUCCESS)")
                if itail > best_itail:
                    continue
                
                # Check once again with asymmetric circuit
                vin_diff = vin_signal-vin_shift
                voutn = voutcm - gain*vin_diff/2
                voutp = voutcm + gain*vin_diff/2
                op_signal = db_n.query(vgs=vin_signal-vtail,
                                        vds=voutn-vtail,
                                        vbs=vb_n-vtail)
                op_shift = db_n.query(vgs=vin_shift-vtail,
                                        vds=voutp-vtail,
                                        vbs=vb_n-vtail)
                ckt = LTICircuit()
                ckt.add_transistor(op_shift, 'outp', 'gnd', 'tail', fg=nf_in)
                ckt.add_transistor(op_signal, 'outn', 'inp', 'tail', fg=nf_in)
                ckt.add_transistor(op_tail, 'tail', 'gnd', 'gnd', fg=nf_tail)
                ckt.add_res(rload, 'outp', 'gnd')
                ckt.add_res(rload, 'outn', 'gnd')
                ckt.add_cap(cload, 'outp', 'gnd')
                ckt.add_cap(cload, 'outn', 'gnd')
                num, den = ckt.get_num_den(in_name='inp', out_name='outn', in_type='v')
                num_unintent, den_unintent = ckt.get_num_den(in_name='inp', out_name='outp', in_type='v')
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
                    continue
                
                print("(SUCCESS)")
                if itail < best_itail:
                    best_itail = itail
                    best_op = dict(nf_in=nf_in,
                                    nf_tail=nf_tail,
                                    itail=itail,
                                    rload=rload,
                                    gain=gain,
                                    fbw=fbw,
                                    voutcm=voutcm,
                                    vtail=vtail,
                                    cin=op_signal['cgg']*nf_in)
    if best_op == None:
        raise ValueError("No viable solutions.")
    return best_op
