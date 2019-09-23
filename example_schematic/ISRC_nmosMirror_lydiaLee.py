import pprint

import numpy as np
import scipy.optimize as sciopt

from math import isnan
from bag.util.search import BinaryIterator
from verification_ec.mos.query import MOSDBDiscrete
from scipy import signal
from bag.data.lti import LTICircuit, get_w_3db, get_stability_margins
from OTA_diffAmpP_lydiaLee import design_diffAmpP

def get_db(spec_file, intent, interp_method='spline', sim_env='TT'):
    # initialize transistor database from simulation data
    mos_db = MOSDBDiscrete([spec_file], interp_method=interp_method)
    # set process corners
    mos_db.env_list = [sim_env]
    # set layout parameters
    mos_db.set_dsn_params(intent=intent)
    return mos_db

def design_Nmirror(db_n, sim_env,
                vdd, itarget,
                vb_n, error_tol=0.1):
    """
    Inputs:
    Returns:
    """
    vg_min = 0.1
    vg_max = vdd
    vg_vec = np.arange(vg_min, vdd, 10e-3)
    for vg in vg_vec:
        print("VG: {}".format(vg))
        in_op = db_n.query(vgs=vg, vds=vg, vbs=vb_n-0)
        nf_in = round(abs(itarget/in_op['ibias']))
        print("OP IBIAS: {}".format(in_op['ibias']))
        if nf_in <= 0:
            continue
        ireal = abs(in_op['ibias']*nf_in)
        print("IREAL: {}".format(ireal))
        ibias_error = abs((ireal-itarget)/itarget)
        if ibias_error > error_tol:
            continue
        rbias = (vdd-vg)/itarget
        return dict(rbias=rbias,
                    nf_in=nf_in,
                    vg=vg)

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
        sim_env=sim_env,
        vdd=1.0,
        itarget=1e-6,
        vb_n=0,
        error_tol=0.05,
        )

    mirror_specs = design_Nmirror(**specs)
    pprint.pprint(mirror_specs)
    print('done')

if __name__ == '__main__':
    run_main()
