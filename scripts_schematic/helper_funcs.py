# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from verification_ec.mos.query import MOSDBDiscrete

# ---------------------------------------
# -------------- Constants --------------
# ---------------------------------------
kB = 1.38e-23

# -----------------------------------------------------
# -------------- BAG-Necessary Functions --------------
# -----------------------------------------------------
def get_db(spec_file, intent, interp_method='spline', sim_env='TT'):
    # initialize transistor database from simulation data
    mos_db = MOSDBDiscrete([spec_file], interp_method=interp_method)
    # set process corners
    mos_db.env_list = [sim_env]
    # set layout parameters
    mos_db.set_dsn_params(intent=intent)
    return mos_db
    
# --------------------------------------------------
# -------------- Various Computations --------------
# --------------------------------------------------
def calculate_Avt_Abeta(db, data_file, lch, wf, nf):
    """
    Calculates the Avt and Abeta of a process technology. This assumes
    that these values are constant.
    
    Inputs:
        db: MOS device data (from get_db)
        data_file: csv file containing vgs, vds, vbs, and sigma_voff for various
            simulation conditions.
        lch: Channel length in meters.
        wf: Single finger width in meters.
        nf: Number of fingers used in the simulation.
    Returns:
        Dictionary with Avt=Avt and Abeta=Abeta parameters.
    """
    # width * length (for clarity later, both in meters)
    WL = wf*nf * lch
    # Initializing the vectors to set up the system of equations.
    A = np.array([[0,0]])
    b = np.array([0])
    # Pulling data from file in a parseable fashion
    sim_data = pd.read_csv(data_file)
    # Uses simulation results and least squares to estimate Avt and Abeta
    for vals in sim_data.itertuples():
        op_info = db.query(vgs=vals.vgs, vds=vals.vds, vbs=vals.vbs)
        vstar = op_info['vstar']
        var_voff = vals.sigma_voff**2
        print("vstar: {}".format(vstar))
        A = np.concatenate((A, np.array([[1/WL, vstar**2/WL]])))
        b = np.concatenate((b, np.array([var_voff])))
        
    x, residual, rank, s = np.linalg.lstsq(A, b)
    Avt = np.sqrt(x[0]) # V/m
    Abeta = np.sqrt(x[1])
    return Avt, Abeta

def estimate_vth(db, vdd, vbs, mos_type="nmos"):
    """
    Estimates the threshold voltage of a MOSFET using the assumption
    Vov = Vstar = Vgs - Vth (for NMOS)
    
    Inputs:
        db: The database file for the appropriate device. See get_db.
        vdd: Approximate supply voltage (will use to put in "saturation")
        vbs: Body-source voltage
        mos_type: "nmos" or "pmos".
    Returns:
        An estimate of the threshold voltage of a device. Note that
        this may heavily vary depending on the vdd value with 
        short-channel devices.
    """
    vgs = vdd
    vds = vdd
    if mos_type == "pmos":
        vgs = -vgs
        vds = -vds
    op = db.query(vgs=vgs, vds=vds, vbs=vbs)
    return vgs-op['vstar']

def parallel(*args):
	"""
	Inputs:
		*args: Unpacked tuple of resistances.
	Returns:
		Float. Parallel combination of all of the arguments in *args.
	"""
	try:
		return 1/sum([1/a for a in args])
	except:
		return 0

def zero_crossing(lst):
	return np.where(np.diff(np.sign(lst)))[0][0]


# -------------------------------------------------
# -------------- Debugging Utilities --------------
# -------------------------------------------------
def cond_print(myStr, yesPrint=True):
	"""
	Prints myStr if yesPrint is true. Used for debugging purposes.
	
	Inputs:
		myStr: String.
		yesPrint: Boolean.
	Returns:
		None.
	"""
	if yesPrint:
		print(myStr)
	return


