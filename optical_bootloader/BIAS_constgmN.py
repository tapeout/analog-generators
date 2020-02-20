# -*- coding: utf-8 -*-

import numpy as np

from bag.util.search import FloatBinaryIterator
from bag.data.lti import LTICircuit, get_w_3db, get_stability_margins
from helper_funcs import verify_ratio, estimate_vth

def design_constgmN(db_n, db_p, vdd,
    vb_n, vb_p, K, voutn, voutp,
    iref_max=5e-6, error_tol=.1):
    """
    Inputs:
        db_n/p:
        vdd:
        vb_n/p:
        K:
        voutn/p:
        iref_max:
        error_tol:
    """
    op_n1 = db_n.query(vgs=voutn, vds=voutp, vbs=vb_n)
    op_p = db_p.query(vgs=voutp-vdd, vds=voutp-vdd, vbs=vb_p-vdd)
    
    nf_n1_min = 2
    if np.isinf(iref_max):
        nf_n1_max = 200
    else:
        nf_n1_max = int(round(iref_max/op_n1['ibias']))
    
    nf_n1_vec = np.arange(nf_n1_min, nf_n1_max, 2)
    for nf_n1 in nf_n1_vec:
        nf_nK = K * nf_n1
        p_match_good, nf_p = verify_ratio(op_n1['ibias'],
                                        op_p['ibias'],
                                        nf_n1,
                                        error_tol)
        if not p_match_good:
            continue
        
        ibias_1 = op_n1['ibias']*nf_n1
        
        vr_iter = FloatBinaryIterator(low=0, high=voutp, 
                                    tol=0, search_step=voutp/2**10)
        # vr = vr_iter.get_next()
        
        while vr_iter.has_next():
            vr = vr_iter.get_next()
            op_nK = db_n.query(vgs=voutn-vr, vds=voutp-vr, vbs=vb_n-vr)
            
            ibias_K = op_nK['ibias']*nf_nK
            ierror = (abs(ibias_1-ibias_K))/min(abs(ibias_1), abs(ibias_K))
            if ierror <= error_tol:
                return dict(nf_n1=nf_n1,
                            nf_nK=nf_nK,
                            nf_p=nf_p,
                            rsource=vr/ibias_K,
                            ibias_K=ibias_K,
                            ibias_1=ibias_1)
            elif ibias_K > ibias_1:
                vr_iter.up()
            else:
                vr_iter.down()
    raise ValueError("No viable solution.")
