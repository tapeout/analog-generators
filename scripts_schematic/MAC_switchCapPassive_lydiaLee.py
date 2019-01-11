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
    
def design_passiveSCMAC(db_n, db_p, sim_env,
                vdd, cload, T, iinnoise_var, vin_vec, scale_vec:
                fbw_min, verror_max,
                vb_p, vb_n, cap_min, cap_max, error_tol=0.05):
    """
    Inputs:
        db_n/db_p:      Databases for NMOS and PMOS devices, respectively.
        sim_env:        Process corner.
        vdd:            Supply voltage.
        cload:          Capacitance seen at output.
        T:              Temperature in Kelvin.
        iinnoise_var:   Variance of input noise (A^2).
        vin_vec:        Array of input Vin[i] voltages.
        scale_vec:      Array of values with which to compute the inner product
                        with vin_vec.
        fbw_min:        Required bandwidth.
        verror_max:     Maximum allowable error at output. Includes noise,
                        imperfect charge transfer, and mismatch. 
        vb_n/p:         Body bias voltage for NMOS and PMOS devices, respectively.
        cap_min/max:    Min/max allowable capacitance.
        error_tol:      Error tolerance for internal calculation.
    Returns:
        
    """
    kBT = 1.38e-23*T
    # Equations
    
    # TODO: Imperfect charge transfer error
    # TODO: Mismatch effects
    # TODO: Noise effects
    
