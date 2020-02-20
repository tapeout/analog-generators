# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from verification.mos.query import MOSDBDiscrete

# ---------------------------------------
# -------------- Constants --------------
# ---------------------------------------
kB = 1.38e-23 # J/K, 8.617e-5 eV/K

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
    that these values are constant. run_Avt_Abeta (below) provides some
    skeleton of how to use this.
    
    Inputs:
        db: MOS device data (from get_db)
        data_file: csv file containing vgs, vds, vbs, and sigma_voff for various
            simulation conditions. Values should be in volts.
        lch: Channel length in microns.
        wf: Single finger width in microns.
        nf: Number of fingers used in the simulation.
    Returns:
        Two values. Avt=Avt (mV/um) and
        Abeta=Abeta (/um) parameters.
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
        A = np.concatenate((A, np.array([[1/WL, (vstar/2)**2/WL]])))
        b = np.concatenate((b, np.array([var_voff])))
        
    x, residual, rank, s = np.linalg.lstsq(A, b)
    Avt = np.sqrt(x[0])*1e3 # mV/um
    Abeta = np.sqrt(x[1])
    return Avt, Abeta

def estimate_vth(db, vdd, vbs, mos_type="nmos"):
    """
    Estimates the absolute value of the threshold voltage of a
    MOSFET using the assumption
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
        vbs = -vbs
    op = db.query(vgs=vgs, vds=vds, vbs=vbs)
    return abs(vgs) - op['vstar']

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

def verify_ratio(ibase_A, ibase_B, nf_A,
    error_tol):
    '''
    Determines if a particular device sizing pairing falls within
    error requirements and is feasible.
    Inputs:
        ibase_A/B:  Float. The drain current of a single A- or B-device.
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
    # 
    B_to_A = ibase_A/ibase_B
    
    # Check if it's possible to achieve the correct ratio with 
    # physical device sizes
    nf_B = int(round(nf_A * B_to_A))
    
    # Is any device now <1 minimum size FET?
    if nf_B < 1:
        return False, nf_B

    # Check current mismatch given quantization        
    id_A = nf_A * ibase_A
    id_B = nf_B * ibase_B
    
    error = (abs(id_A) - abs(id_B))/abs(id_A)
    
    if abs(error) > error_tol:
        # print("\t\tERROR TOLERANCE: {0}".format(abs(error)))
        return False, nf_B
        
    return True, nf_B

def calculate_Nsigma(cyield):
    '''
    Inputs:
        cyield: Float <= 1. The fractional yield of a system.
    Outputs:
        Returns the (float) number of standard deviations to target for 
        reliability when simulating. 
    '''
    return np.sqrt(2)*special.erfinv(cyield)

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

# -----------------------------------------------------------------
# -------------- Miscellaneous (Modify As Necessary) --------------
# -----------------------------------------------------------------
