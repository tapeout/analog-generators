# -*- coding: utf-8 -*-

import numpy as np

from bag.util.search import FloatBinaryIterator
from bag.data.lti import LTICircuit, get_w_3db, get_stability_margins

def design_LPF_AMP(db_n, db_p, db_bias, sim_env,
    vin, vdd_nom, vdd_vec, cload,
    gain_min, fbw_min, pm_min,
    vb_n, vb_p, error_tol=0.1, ibias_max=20e-6):
    '''
    Designs an amplifier with an N-input differential pair
    with a source follower. Uses the LTICircuit functionality.
    Inputs:
        db_n/p:     Databases for non-biasing NMOS and PMOS device 
                    characterization data, respectively.
        db_bias:    Database for tail NMOS device characterization data.
        sim_env:    Simulation corner.
        vin:        Float. Input (and output and tail) bias voltage in volts.
        vtail_res:  Float. Step resolution in volts when sweeping tail voltage.
        vdd_nom:    Float. Nominal supply voltage in volts.
        vdd_vec:    Collection of floats. Elements should include the min and
                    max supply voltage in volts.
        cload:      Float. Output load capacitance in farads.
        gain_min:   Float. Minimum DC voltage gain in V/V.
        fbw_min:    Float. Minimum bandwidth (Hz).
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
        nf_cs_n/pB/nB:  Integer. Number of minimum device widths for common source
                        N-input/active P-load/tail device.
        nf_sfN_n/pB/nB: Integer. Number of minimum device widths for source follower
                        N-input/P-bias/N-bias.
        nf_sfP_p/pB/nB: Integer. Number of minimum device widths for source follower
                        P-input/P-bias/N-bias.
        vtail:          Float. DC tail voltage for the common source stage.
                        This value is reused throughout the circuit.
        gain_cs:        Float. DC voltage gain of the CS amplifier (V/V).
        gain_sfN:       Float. DC voltage gain of the first source follower (V/V).
        gain_sfP:       Float. DC voltage gain of the second source follower (V/V).
        gain:           Float. DC voltage gain of both stages combined (V/V).
        fbw:            Float. Bandwdith (Hz).
        pm:             Float. Phase margin (degrees) for unity gain.
    '''
    possibilities = []
    ibias_budget = ibias_max
    
    # Given this amplifier will be unity gain feedback at DC
    vout = vin
    
    vstar_min = 0.15
    vtail_vec = np.arange(vstar_min, vout, vtail_res)
    for vtail in vtail_vec:
        print("\nVTAIL: {0}".format(vtail))
        n_op = db_n.query(vgs=vin-vtail, vds=vout-vtail, vbs=vb_n-vtail)
        p_B_op = db_p.query(vgs=vout-vdd_nom, vds=vout-vdd_nom, vbs=vb_p-vdd_nom)
        n_B_op = db_bias.query(vgs=vout-0, vds=vtail-0, vbs=vb_n-0)
        p_op = db_p.query(vgs=vtail-vout, vds=vtail-vout, vbs=vb_p-vout)
    
        # Finding the ratio between devices to converge to correct biasing
        idn_base = n_op['ibias']
        idp_B_base = p_B_op['ibias']
        idn_B_base = n_B_op['ibias']
        idp_base = p_op['ibias']
                
        pB_to_n = abs(idn_base/idp_B_base)
        nB_to_n = abs(idn_base/idn_B_base)
        pB_to_p = abs(idp_base/idp_B_base)
        nB_to_p = abs(idp_base/idn_B_base)
        
        ### P-input SF ###
        sfP_mult_max = int(round(abs(ibias_budget/idn_base)))
        sfP_mult_vec = np.arange(1, sfP_mult_max, 1)
        for sfP_mult in sfP_mult_vec in 
            nf_sfP_p = sfP_mult
            
            # Verify that the sizing is feasible and gets sufficiently
            # good current matching
            pB_sfP_good, nf_sfP_pB = verify_ratio(idp_base, idp_B_base, pB_to_p, 
                                        nf_sfP_p, error_tol)
            nB_sfP_good, nf_sfP_nB = verify_ratio(idp_base, idn_B_base, nB_to_p,
                                        nf_sfP_p, error_tol)
            if not (pB_sfP_good and nB_sfP_good):
                continue
            
            # Check SF2 bandwidth; size up until it meets
            ckt = construct_sfP_LTI(p_op, p_B_op, n_B_op, 
                    nf_sfP_p, nf_sfP_pB, nf_sfP_nB,
                    cload)
            sfP_num, sfP_den = ckt.get_num_den(in_name='out2', out_name='out', in_type='v')
            fbw_sfP = get_w_3db(sfP_num, sfP_den)/(2*np.pi)
            
            # No need to keep sizing up
            if fbw_sfP >= fbw_min:
                break
        
        # Final stage can never meet bandwidth given biasing
        if fbw_sfP < fbw_min:
            print("SFP: BW failure\n")
            continue
        
        # Sizing ratio failure
        if 0 in [nf_sfP_pB, nf_sfP_nB]:
            print("SFP: sizing ratio failure\n")
            continue
        
        ibias_sfP = idp_base * nf_sfP_p
        ibias_budget = ibias_budget - ibias_sfP
        print("Remaining budget\t:\t {0}".format(ibias_budget))
        
        ### N-input SF ###
        sfN_mult_max = int(round(abs(ibias_budget/idn_base)))
        sfN_mult_vec = np.arange(1, sfN_mult_max, 1)
        for sfN_mult in sfN_mult_vec:
            nf_sfN_n = sfN_mult
            
            # Repeat the same feasibility + current matching check
            pB_sfN_good, nf_sfN_pB = verify_ratio(idn_base, idp_B_base, 
                                            pB_to_n, nf_sfN_n, error_tol)
            nB_sfN_good, nf_sfN_pN = verify_ratio(idn_base, idn_B_base,
                                            nB_to_n, nf_sfN_n, error_tol)
            if not (pB_sfn_good and nB_sfN_good):
                continue
            
            # Check SF1 bandwidth; size up until it meets
            ckt = construct_sfN_LTI(p_op, p_B_op, n_B_op, 
                    nf_sfP_p, nf_sfP_pB, nf_sfP_nB,
                    cload,
                    n_op,
                    nf_sfN_n, nf_sfN_pB, nf_sfN_nB)
            sfN_num, sfN_den = ckt.get_num_den(in_name='out1', out_name='out', in_type='v')
            fbw_sfN = get_w_3db(sfN_num, sfN_den)/(2*np.pi)
            
            # No need to keep sizing up
            if fbw_sfN >= fbw_min:
                break
            
        # Second stage can never meet bandwidth given restrictions
        if fbw_sfN < fbw_min:
            print("SFN: BW failure\n")
            continue
        
        # Sizing ratio failure
        if 0 in [nf_sfN_pB, nf_sfN_nB]:
            print("SFN: sizing ratio failure\n")
            continue
        
        ibias_sfN = idn_base * nf_sfN_n
        ibias_budget = ibias_budget - ibias_sfN
        print("Remaining budget\t:\t {0}".format(ibias_budget))
        
        ### CS input ###
        cs_mult_max = int(round(abs(ibias_budget/idn_base)))
        cs_mult_vec = np.arange(1, cs_mult_max, 1)
        for cs_mult in cs_mult_vec:
            nf_cs_n = cs_mult_vec
            
            # Verify that the sizing is feasible and gets sufficiently
            # good current matching
            pB_cs_good, nf_cs_pB = verify_ratio(idn_base, idp_B_base, pB_to_n, nf_cs_n, 
                                        error_tol)
            nB_cs_good, nf_cs_nB = verify_ratio(idn_base, idn_B_base, nB_to_n, nf_cs_n*2,
                                        error_tol)
            if not (pB_cs_good and nB_cs_good):
                continue 
            
            # Check combined stages' small signal
            ckt = construct_total_LTI(p_op, p_B_op, n_B_op, 
                    nf_sfP_p, nf_sfP_pB, nf_sfP_nB,
                    cload,
                    n_op,
                    nf_sfN_n, nf_sfN_pB, nf_sfN_nB,
                    nf_cs_n, nf_cs_pB, nf_cs_nB)
            total_num, total_den = ckt.get_num_den(in_name='in', out_name='out', in_type='v')
            
            # Check cumulative gain
            gain_total = abs(total_num[-1]/total_den[-1])
            if gain_total < gain_min:
                print("CS: A0 failure {0}\n".format(gain_total))
                break # Biasing sets the gain
            
            # Check cumulative bandwidth
            fbw_total = get_w_3db(total_num, total_den)/(2*np.pi)
            if fbw_total < fbw_min:
                print("CS: BW failure {0}\n".format(fbw_total))
                continue
                
            # Check phase margin (TODO?)
            loopBreak = ckt.get_transfer_function(in_name='out', out_name='in', in_type='v')
            pm, gainm = get_stability_margins(loopBreak.num, loopBreak.den)
            if pm < pm_min or isnan(pm):
                print("CS: PM failure {0}\n".format(pm))
                continue
                
            # If gain, bw, and PM are met, no need to keep sizing up
            break

        # Sizing ratio failure
        if 0 in [nf_cs_pB, nf_cs_nB]:
            print("CS: sizing ratio failure\n")
            continue
        
        # Iterated, biasing condition + constraints can't meet spec
        if fbw_total < fbw_min or gain_total < gain_min or pm < pm_min:
            continue
        else:
            print("HALLELUJAH SUCCESS\n")
            cs_num, cs_den = ckt.get_num_den(in_name='in', out_name='out1', in_type='v')
            sfN_num, sfN_den = ckt.get_num_den(in_name='out1', out_name='out2', in_type='v')
            sfP_num, sfP_den = ckt.get_num_den(in_name='out2', out_name='out', in_type='v')
            
            gain_cs = abs(cs_num[-1]/cs_den[-1])
            gain_sfN = abs(sfN_num[-1]/sfN_den[-1])
            gain_sfP = abs(sfP_num[-1]/sfP_den[-1])
            viable = dict(
                        nf_cs_n     = nf_cs_n,
                        nf_cs_pB    = nf_cs_pB,
                        nf_cs_nB    = nf_cs_nB,
                        nf_sfN_n    = nf_sfN_n,
                        nf_sfN_pB   = nf_sfN_pB,
                        nf_sfN_nB   = nf_sfN_nB,
                        nf_sfP_p    = nf_sfP_p,
                        nf_sfP_pB   = nf_sfP_pB,
                        nf_sfP_nB   = nf_sfP_nB,
                        vtail       = vtail,
                        gain_cs     = gain_cs,
                        gain_sfN    = gain_sfN,
                        gain_sfP    = gain_sfP,
                        gain        = gain_total,
                        fbw         = fbw_total,
                        pm          = pm)
            pprint.pprint(viable)
            print("\n")
            possibilities.append([viable])
        
    # TODO: Check all other VDDs
    return possibilities
    
    
def construct_sfP_LTI(p_op, p_B_op, n_B_op, 
                    nf_sfP_p, nf_sfP_pB, nf_sfP_nB,
                    cload):
    '''
    Inputs:
        *_op:   Operating point information for a given device. A capital "B" 
                indicates a "bias" device, i.e. a device whose gm doesn't really
                factor into the final gain expression.
        nf_*:   Integer. Number of minimum channel width/length devices in parallel.
        cload:  Float. Load capacitance in farads.
    Outputs:
        Returns the LTICircuit constructed for the second-stage P-input 
        source follower.
    '''
    ckt = LTICircuit()
            ckt.add_transistor(p_op, 'tail_copy_sfP', 'out2', 'out', fg=nf_sfP_p)
            ckt.add_transistor(p_B_op, 'out', 'gnd', 'gnd', fg=nf_sfP_pB)
            ckt.add_transistor(n_B_op, 'tail_copy_sfP', 'gnd', 'gnd', fg=nf_sfP_nB)
            ckt.add_cap(cload, 'out', 'gnd')
    return ckt
    
    
def construct_sfN_LTI(p_op, p_B_op, n_B_op, 
                    nf_sfP_p, nf_sfP_pB, nf_sfP_nB,
                    cload,
                    n_op,
                    nf_sfN_n, nf_sfN_pB, nf_sfN_nB):
    '''
    Inputs:
        *_op:   Operating point information for a given device. A capital "B" 
                indicates a "bias" device, i.e. a device whose gm doesn't really
                factor into the final gain expression.
        nf_*:   Integer. Number of minimum channel width/length devices in parallel.
        cload:  Float. Load capacitance in farads.
    Outputs:
        Returns the LTICircuit constructed for the two source followers
        (N-input followed by P-input source follower).
    '''
    ckt = construct_sfP_LTI(p_op, p_B_op, n_B_op, 
                    nf_sfP_p, nf_sfP_pB, nf_sfP_nB,
                    cload)
    ckt.add_transistor(n_op, 'out1_copy_sfN', 'out1', 'out2', fg=nf_sfN_n)
            ckt.add_transistor(p_B_op, 'out1_copy_sfN', 'out1_copy_sfN', 'gnd', fg=nf_sfN_pB)
            ckt.add_transistor(n_B_op, 'out2', 'gnd', 'gnd', fg=nf_sfN_nB)
    return ckt
    
    
def construct_total_LTI(p_op, p_B_op, n_B_op, 
                    nf_sfP_p, nf_sfP_pB, nf_sfP_nB,
                    cload,
                    n_op,
                    nf_sfN_n, nf_sfN_pB, nf_sfN_nB,
                    nf_cs_n, nf_cs_pB, nf_cs_nB):
    '''
    Inputs:
        *_op:   Operating point information for a given device. A capital "B" 
                indicates a "bias" device, i.e. a device whose gm doesn't really
                factor into the final gain expression.
        nf_*:   Integer. Number of minimum channel width/length devices in parallel.
        cload:  Float. Load capacitance in farads.
    Outputs:
        Returns the LTICircuit constructed for the amplifier
        (N-input common source followed by N-input source follower followed
        by P-input source follower).
    '''
    ckt = construct_sfN_LTI()
    # Left side
    ckt.add_transistor(n_op, 'out1_copy', 'gnd', 'tail', fg=nf_cs_n)
    ckt.add_transistor(p_B_op, 'out1_copy', 'out1_copy', 'gnd', fg=nf_cs_pB)
    
    # Right side 
    ckt.add_transistor(n_op, 'out1', 'in', 'tail', fg=nf_cs_n)
    ckt.add_transistor(p_B_op, 'out1', 'out1_copy', 'gnd', fg=nf_cs_pB)
    
    # Tail device
    ckt.add_transistor(n_op, 'tail', 'gnd', 'gnd', fg=nf_cs_nB)
    
    return ckt


def verify_ratio(ibase_A, ibase_B, B_to_A, nf_A,
    error_tol=0.1):
    '''
    Determines if a particular device sizing pairing falls within
    error requirements and is feasible.
    Inputs:
        ibase_A/B:  Float. The drain current of a single A- or B-device.
        B_to_A:     Integer. The factor (W/L of B)/(W/L of A)
        nf_A:       Integer. W of A.
        error_tol:  Float. Fractional tolerance for ibias error when computing 
                    the device sizing ratio.
    Outputs:
        Two arguments.
        (1) Boolean. True if the ratio is possible and meets the error 
            tolerance. False otherwise.
        (2) nf_B. Integer indicating the sizing of the second device. 0 if 
            invalid.
    '''
        # Check if it's possible to achieve the correct ratio with 
        # physical device sizes
        nf_B = int(round(nf_A * B_to_A))
        
        # Is any device now <1 minimum size FET?
        if nf_B < 1:
            return False, 0

        # Check current mismatch given quantization        
        id_A = nf_A * ibase_A
        id_B = nf_B * ibase_B
        
        error = (abs(id_A) - abs(id_B))/abs(id_A)
        
        if abs(error) > error_tol:
            return False, 0
            
        return True, nf_B
