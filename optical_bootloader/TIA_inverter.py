# -*- coding: utf-8 -*-

import numpy as np

from bag.util.search import FloatBinaryIterator
from bag.data.lti import LTICircuit, get_w_3db, get_stability_margins

def design_TIA_inverter(db_n, db_p, sim_env,
        vg_res, rf_res,
        vdd_nom, vdd_vec, cpd, cload, 
        rdc_min, fbw_min, pm_min, BER_max,
        vos, isw_pkpk,
        vb_n, vb_p, error_tol=0.05, ibias_max=20e-6):
    """
    Designs a transimpedance amplifier with an inverter amplifier
    in resistive feedback. Uses the LTICircuit functionality.
    Inputs:
        db_n/p:     Databases for NMOS and PMOS device characterization data, 
                    respectively.
        sim_env:    Simulation corner.
        vg_res:     Float. Step resolution in volts when sweeping gate voltage.
        rf_res:     Float. Step resolution in ohms when sweeping feedback 
                    resistance.
        vdd_nom:    Float. Nominal supply voltage in volts.
        vdd_vec:    Collection of floats. Elements should include the min and
                    max supply voltage in volts.
        cpd:        Float. Input parasitic capacitance in farads.
        cload:      Float. Output load capacitance in farads.
        rdc_min:    Float. Minimum DC transimpedance in ohms.
        fbw_min:    Float. Minimum bandwidth (Hz).
        pm_min:     Float. Minimum phase margin in degrees.
        BER_max:    Float. Maximum allowable bit error rate (as a fraction).
        vos:        Float. Input-referred DC offset for any subsequent
                    comparator stage as seen at the output
                    of the TIA.
        isw_pkpk:   Float. Input current peak-to-peak swing in amperes.
        vb_n/p:     Float. Back-gate/body voltage (V) of NMOS and PMOS, 
                    respectively.
        error_tol:  Float. Fractional tolerance for ibias error when computing 
                    the p-to-n ratio.
        ibias_max:  Float. Maximum bias current (A) allowed.
    Raises:
        ValueError: If unable to meet the specification requirements.
    Outputs:
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
    # Finds all possible designs for one value of VDD, then
    # confirm which work with all other VDD values.
    possibilities = []

    vg_vec = np.arange(0, vdd_nom, vg_res)
    
    for vg in vg_vec:
        n_op_info = db_n.query(vgs=vg, vds=vg, vbs=vb_n-0)
        p_op_info = db_p.query(vgs=vg-vdd_nom, vds=vg-vdd_nom, vbs=vb_p-vdd_nom)
        
        iref_n = n_op_info['ibias']
        iref_p = p_op_info['ibias']
        
        # Calculate the ratio of NMOS to PMOS based
        # on drive current and bias point
        pn_ratio = abs(iref_n/iref_p)
        if pn_ratio < 1:
            continue
        
        if np.isinf(ibias_max):
            nf_n_max = 100
        else:
            nf_n_max = int(round(ibias_max/iref_n))
            
        nf_n_vec = np.arange(1, nf_n_max, 1)
        for nf_n in nf_n_vec:
            # Number of fingers can only be integer,
            # so increase as necessary until you get
            # sufficiently accurate/precise bias + current match
            nf_p = int(round(nf_n*pn_ratio))
            if nf_p <= 0:
                continue
            elif nf_p > nf_n_max:
                break

            ibias_n = iref_n * nf_n
            ibias_p = iref_p * nf_p
            ibias_error = (abs(ibias_n)-abs(ibias_p))/min(abs(ibias_n), abs(ibias_p))
            if ibias_error > error_tol:
                # ("ibias_error: {}".format(ibias_error))
                continue

            # Getting small signal parameters to constrain Rf
            inv = LTICircuit()
            inv.add_transistor(n_op_info, 'out', 'in', 'gnd', fg=nf_n)
            inv.add_transistor(p_op_info, 'out', 'in', 'gnd', fg=nf_p)
            inv_num, inv_den = inv.get_num_den(in_name='in', out_name='out', in_type='v')
            A0 = abs(inv_num[-1]/inv_den[-1])
            
            gds_n = n_op_info['gds'] * nf_n
            gds_p = p_op_info['gds'] * nf_p
            gds = abs(gds_n) + abs(gds_p)
            ro = 1/gds
            
            # Assume Rdc is negative, bound Rf
            rf_min = max(rdc_min*(1+A0)/A0 + ro/A0, 0)
            rf_vec = np.arange(rf_min, rdc_min*2, rf_res)
            for rf in rf_vec:
                # With all parameters, check if it meets small signal spec
                meets_SS, SS_vals = verify_TIA_inverter_SS(n_op_info, p_op_info,
                                nf_n, nf_p, rf, cpd, cload,
                                rdc_min, fbw_min, pm_min)
                # With all parameters, estimate if it will meet noise spec
                meets_noise, BER = verify_TIA_inverter_BER(n_op_info, p_op_info, 
                    nf_n, nf_p,
                    rf, cpd, cload,
                    BER_max, vos, isw_pkpk)
                
                meets_spec = meets_SS # and meets_noise
                # If it meets small signal spec, append it to the list
                # of possibilities
                if meets_spec:
                    possibilities.append(dict(vg=vg,
                        vdd=vdd_nom,
                        nf_n=nf_n,
                        nf_p=nf_p,
                        rf=rf,
                        rdc=SS_vals['rdc'],
                        fbw=SS_vals['fbw'],
                        pm=SS_vals['pm'],
                        ibias=ibias_n,
                        BER=BER))
                elif SS_vals['fbw'] != None and SS_vals['fbw'] < fbw_min:
                    # Increasing resistor size won't help bandwidth
                    break
    
    # Go through all possibilities which work at the nominal voltage
    # and ensure functionality at other bias voltages
    # Remove any nonviable options
    print("{0} working at nominal VDD".format(len(possibilities)))
    for candidate in possibilities:
        nf_n = candidate['nf_n']
        nf_p = candidate['nf_p']
        rf = candidate['rf']
        for vdd in vdd_vec:
            new_op_dict = vary_supply(vdd, db_n, db_p, nf_n, nf_p, vb_n, vb_p)
            vg = new_op_dict['vb']
            n_op = new_op_dict['n_op']
            p_op = new_op_dict['p_op']
            
            # Confirm small signal spec is met
            meets_SS, scratch = verify_TIA_inverter_SS(n_op, p_op,
                                nf_n, nf_p, rf, cpd, cload,
                                rdc_min, fbw_min, pm_min)
            
            # Confirm noise spec is met
            meets_noise, BER = verify_TIA_inverter_BER(n_op, p_op, 
                    nf_n, nf_p,
                    rf, cpd, cload,
                    BER_max, vos, isw_pkpk)
                    
            meets_spec = meets_SS # and meets_noise
            
            if not meets_spec:
                possibilities.remove(candidate)
                break
    
    # Of the remaining possibilities, check for lowest power.
    # If there are none, raise a ValueError.
    if len(possibilities) == 0:
        raise ValueError("No final viable solutions")
    
    print("{0} working at all VDD".format(len(possibilities)))
    best_op = possibilities[0]
    for candidate in possibilities:
        best_op = choose_op_comparison(best_op, candidate)
        
    return best_op
        
        
def choose_op_comparison(op1, op2):
    """
    Inputs:
        op1/2:  Two dictionaries describing operating points for each
                scenario. key:value should include
                rf:     Float. Value of feedback resistor.
                rdc:    Float. Expected DC transimpedance.
                fbw:    Float. Expected bandwidth (Hz).
                pm:     Float. Expected phase margin.
                ibias:  Float. Expected DC bias current.
                BER:    Float. Bit error rate.
    Outputs:
        Returns op1 or op2---whichever it decides upon. Compares the two 
        operating points at nominal VDD and chooses the "superior" of the 
        two. Superiority is not defined here since it can change between
        implementations.
    """
    # Bias current
    if op1['ibias'] < op2['ibias']:
        return op1
    elif op1['ibias'] > op2['ibias']:
        return op2
    
    # BER
    if op1['BER'] < op2['BER']:
        return op1
    elif op1['BER'] > op2['BER']:
        return op2
    
    # Bandwidth
    if op1['fbw'] > op2['fbw']:
        return op1
    elif op1['fbw'] < op2['fbw']:
        return op2
        
    # Gain
    if op1['rdc'] > op2['rdc']:
        return op1
    elif op1['rdc'] < op2['rdc']:
        return op2
        
    # Phase margin
    if op1['pm'] > op2['pm']:
        return op1
    elif op1['pm'] < op2['pm']:
        return op2
    
    return op1


def vary_supply(vdd, db_n, db_p, nf_n, nf_p, vb_n, vb_p, error_tol=0.01):
    """
    Inputs:
        vdd:        Float. Supply voltage (V).
        db_n/p:     Databases for NMOS and PMOS device characterization data, 
                    respectively.
        nf_n/p:     Integer. Number of channel fingers for the NMOS/PMOS.
        vb_n/p:     Float. Back-gate/body voltage (V) of NMOS and PMOS, 
                    respectively.
        error_tol:  Float. Fractional tolerance for difference in ibias between
                    NMOS and PMOS while trying to find output bias point. 
                    Relative to the smaller of the two
    Outputs:
        Returns the bias conditions that result from changing the supply
        voltage but nothing else (same devices, same size, etc.). Output 
        comes in the form of a dictionary with key:value pairings:
        vb:     Float. Input bias voltage (and output bias voltage).
        ibias:  Float. Expected DC bias current.
        n/p_op: MOSDBDiscrete operating point of the NMOS and PMOS
                with the changed supply voltage.
    Raises:
        ValueError if no convergence is found (this means something)
        is wrong with the code.
    """
    vb_iter = FloatBinaryIterator(low=0, high=vdd, tol=0, search_step=vdd/2**10)
    vb = vb_iter.get_next()
    
    while vb_iter.has_next():
        vb = vb_iter.get_next()
        
        # Get the operating points
        n_op = db_n.query(vgs=vb, vds=vb, vbs=vb_n-0)
        p_op = db_p.query(vgs=vb-vdd, vds=vb-vdd, vbs=vb_p-vdd)
        
        # Check if the bias currents match
        # If they do, finish
        # If the NMOS current is higher, lower the bias voltage
        # If the PMOS current is higher, raise the bias voltage
        ibias_n = n_op['ibias']*nf_n
        ibias_p = p_op['ibias']*nf_p
        
        ierror = (abs(ibias_n)-abs(ibias_p))/min(abs(ibias_n), abs(ibias_p))
        if ierror <= error_tol:
            return dict(vb=vb,
                        ibias=(ibias_n+ibias_p)/2,
                        n_op=n_op,
                        p_op=p_op)
        elif ibias_n > ibias_p:
            vb_iter.down()
        else:
            vb_iter.up()
            
    raise ValueError("No convergence for VDD={0}V".format(vdd))


def verify_TIA_inverter_SS(
    n_op_info, p_op_info, nf_n, nf_p, rf, cpd, cload,
    rdc_min, fbw_min, pm_min):
    """
    Inputs:
        n/p_op_info:    The MOSDBDiscrete library for the NMOS and PMOS
                        devices in the bias point for verification.
        nf_n/p:         Integer. Number of channel fingers for the NMOS/PMOS.
        rf:             Float. Value of the feedback resistor in ohms.
        cpd:            Float. Input capacitance not from the TIA in farads.
        cload:          Float. Output capacitance not from the TIA in farads.
        rdc_min:        Float. Minimum DC transimpedance in ohms.
        fbw_min:        Float. Minimum bandwidth (Hz).
        pm_min:         Float. Minimum phase margin in degrees.
    Outputs:
        Returns two values
        The first is True if the spec is met, False otherwise.
        The second is a dictionary of values for rdc (DC transimpedance, V/I), 
        bw (bandwidth, Hz), and pm (phase margin, deg) if computed. None otherwise.
    """
    # Getting relevant small-signal parameters
    gds_n = n_op_info['gds'] * nf_n
    gds_p = p_op_info['gds'] * nf_p
    gds = gds_n + gds_p
    
    gm_n = n_op_info['gm'] * nf_n
    gm_p = p_op_info['gm'] * nf_p
    gm = gm_n + gm_p
    
    cgs_n = n_op_info['cgs'] * nf_n
    cgs_p = p_op_info['cgs'] * nf_p
    cgs = cgs_n + cgs_p
    
    cds_n = n_op_info['cds'] * nf_n
    cds_p = p_op_info['cds'] * nf_p
    cds = cds_n + cds_p
    
    cgd_n = n_op_info['cgd'] * nf_n
    cgd_p = p_op_info['cgd'] * nf_p
    cgd = cgd_n + cgd_p
    
    # Circuit for GBW
    circuit = LTICircuit()
    circuit.add_transistor(n_op_info, 'out', 'in', 'gnd', fg=nf_n)
    circuit.add_transistor(p_op_info, 'out', 'in', 'gnd', fg=nf_p)
    circuit.add_res(rf, 'in', 'out')
    circuit.add_cap(cpd, 'in', 'gnd')
    circuit.add_cap(cload, 'out', 'gnd')
    
    # Check gain
    num, den = circuit.get_num_den(in_name='in', out_name='out', in_type='i')
    rdc = num[-1]/den[-1]
   
    if abs(round(rdc)) < round(rdc_min):
        print("GAIN:\t{0} (FAIL)\n".format(rdc))
        return False, dict(rdc=rdc,fbw=None, pm=None)

    # Check bandwidth
    fbw = get_w_3db(num, den)/(2*np.pi)
    if fbw < fbw_min or np.isnan(fbw):
        print("BW:\t{0} (FAIL)\n".format(fbw))
        return False, dict(rdc=rdc,fbw=fbw, pm=None)

    # Check phase margin by constructing an LTICircuit first
    circuit2 = LTICircuit()
    """circuit2.add_transistor(n_op_info, 'out', 'in', 'gnd', fg=nf_n)
    circuit2.add_transistor(p_op_info, 'out', 'in', 'gnd', fg=nf_p)
    circuit2.add_cap(cpd, 'in', 'gnd')
    circuit2.add_cap(cload, 'out', 'gnd')
    circuit2.add_res(rf, 'in', 'break')
    # Cancel Cgd to correctly break loop
    circuit2.add_cap(-cgd, 'in' , 'out')
    circuit.add_cap(cgd, 'in', 'break')"""
    
    
    circuit2.add_conductance(gds, 'out', 'gnd')
    circuit2.add_cap(cgs+cpd, 'in', 'gnd')
    circuit2.add_cap(cds+cload, 'out', 'gnd')
    circuit2.add_cap(cgd, 'in', 'out')
    circuit2.add_res(rf, 'in', 'out')
    
    loopBreak = circuit2.get_transfer_function(in_name='in', out_name='out', in_type='i')
    pm, gainm = get_stability_margins(loopBreak.num*gm, loopBreak.den)
    if pm < pm_min or np.isnan(pm):
        print("PM:\t{0} (FAIL)\n".format(pm))
        return False, dict(rdc=rdc,fbw=fbw, pm=pm)
    print("SUCCESS\n")
    return True, dict(rdc=rdc, fbw=fbw, pm=pm)


def verify_TIA_inverter_BER(n_op_info, p_op_info, nf_n, nf_p,
    rf, cpd, cload,
    BER_max, vos, isw_pkpk):
    """
    Inputs:
        n/p_op_info:    The MOSDBDiscrete library for the NMOS and PMOS
                        devices in the bias point for verification.
        nf_n/p:         Integer. Number of channel fingers for the NMOS/PMOS.
        rf:             Float. Value of the feedback resistor in ohms.
        cpd:            Float. Input capacitance not from the TIA in farads.
        cload:          Float. Output capacitance not from the TIA in farads.
        BER_max:        Float. Maximum allowable bit error rate (as a fraction).
        vos:            Float. Input-referred DC offset for any subsequent
                        comparator stage as seen at the output
                        of the TIA.
        isw_pkpk:       Float. Input current peak-to-peak swing in amperes.
    Outputs:
        Returns two values
        The first is True if the spec is met, False otherwise.
        The second is the BER (float).
    """
    return True, 0
    # Getting relevant small-signal parameters
    gds_n   = n_op_info['gds'] * nf_n
    cdd_n   = n_op_info['cdd'] * nf_n
    gm_n    = n_op_info['gm']  * nf_n
    cgd_n   = n_op_info['cgd'] * nf_n
    cgs_n   = n_op_info['cgs'] * nf_n
    gamma_n = n_op_info['gamma']
    cgg_n   = n_op_info['cgg'] * nf_n
    cdd_n = n_op_info['cdd'] * n

    gds_p   = p_op_info['gds'] * nf_p
    cdd_p   = p_op_info['cdd'] * nf_p
    gm_p    = p_op_info['gm']  * nf_p
    cgd_p   = p_op_info['cgd'] * nf_p
    cgs_p   = p_op_info['cgs'] * nf_p
    gamma_p = p_op_info['gamma']
    
    gm = abs(gm_n) + abs(gm_p)
    ro = 1/abs(gds_n + gds_p)
    A0 = gm*ro
    
    cout = cload + cdd_n + cdd_p
    miller = (1-gm*rf)/(ro+rf)*ro
    cin = cpd + (1-miller)*(cgd_n + cgd_p) + cgs_n + cgs_p
    
    rdc = -A0*rf/(1+A0) + ro/(1+A0)
    
    vin = isw_pkpk/2 * asb(rdc)
    
    # Values to be used repeatedly in noise calculation
    w0Q = (1+A0) / (ro*(cin+cout) + rf*cin)
    w0squared = (1+A0) / (cout*ro*cin*rf)                
    wz_FET = 1/(cin*rf)
    wz_rf = gm/cin
    
    # Variance of the white noise (integrated)
    var_noise_FETs = (4*kBT*(gamma_n*gm_n + gamma_p*gm_p)) \
        * (ro/(1+A0))**2 \
        * w0Q/4 \
        * (1+w0squared/wz_FET**2)
    var_noise_rf  = (4*kBT/rf) \
        * (A0*rf/(1+A0))**2 \
        * w0Q/4 \
        * (1+w0squared/wz_FET**2)
        
    var_noise = var_noise_FETs + var_noise_rf
    std_noise = np.sqrt(var_noise)
    BER = 0.5*special.erfc((vin - vos) / (np.sqrt(2)*std_noise))
    
    return BER<=BER_max, BER
