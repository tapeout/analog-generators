# -*- coding: utf-8 -*-

# ASSUMPTIONS:
# (1) LVT devices
import os
import pprint

import numpy as np
import scipy.optimize as sciopt
from scipy import signal, special
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import RectBivariateSpline

from math import isnan, ceil, floor
from bag.util.search import BinaryIterator
from verification_ec.mos.query import MOSDBDiscrete

from bag.data.lti import LTICircuit, get_w_3db, get_stability_margins
from bag import BagProject
from bag.io import load_sim_results, save_sim_results, load_sim_file

# Pre-created template testbench
tb_lib = 'zz_scratch_avi'
tb_cell = 'sim_ll_inverter_tia'

# The library to be used for sweeps
impl_lib = 'ZZ_inverter_tia'
data_dir = os.path.join('data', 'demo')

os.makedirs(data_dir, exist_ok=True)

def get_tb_name():
    return '_'.join(['demo', 'TIA_inverter'])

def get_db(spec_file, intent, interp_method='spline', sim_env='tt'):
    # initialize transistor database from simulation data
    mos_db = MOSDBDiscrete([spec_file], interp_method=interp_method)
    # set process corners
    mos_db.env_list = [sim_env]
    # set layout parameters
    mos_db.set_dsn_params(intent=intent)
    return mos_db


def sim_tia_inverter(prj, db_n, db_p, sim_env,
        vdd, cpd, cload, 
        nf_n_vec, nf_p_vec, rf_vec):
    """
    Inputs:
    Outputs:
    """
    tb_name = get_tb_name()
    fname = os.path.join(data_dir, '%s.data' % tb_name)
    
    print("Creating testbench %s..." % tb_name)
    
    # Generate schematic
    tb_sch = prj.create_design_module(tb_lib, tb_cell)
    tb_sch.design()
    tb_sch.implement_design(impl_lib, top_cell_name=tb_name)
    tb_obj = prj.configure_testbench(impl_lib, tb_name) 

    # Simulating nf_n, nf_p, and rf combinations in the vectors
    # In the interest of time, it matches each element in the vectors
    # i.e. nf_n_vec[i] will only be simulated with nf_p_vec[i] and rf_vec[i]
    for i in range(len(nf_n_vec)):
        # Setting the parameters in the testbench
        nf_n = nf_n_vec[i]
        nf_p = nf_p_vec[i]
        rf = rf_vec[i]
        tb_obj.set_parameter('nf_n', nf_n)
        tb_obj.set_parameter('nf_p', nf_p)
        tb_obj.set_parameter('Rf', rf)
        
        tb_obj.set_parameter('CL', cload)
        tb_obj.set_parameter('CPD', cpd)
        tb_obj.set_parameter('VDD', vdd)
        
        tb_obj.set_parameter('FIN', 5e9)
        tb_obj.set_parameter('IIN', 1)
        
        # Update the testbench and run the simulation
        tb_obj.update_testbench()
        print("Simulating testbench %s..." % tb_name)
        save_dir = tb_obj.run_simulation()

        # Load sim results into Python
        print('Simulation done, saving results')
        results = load_sim_results(save_dir)
        
        # Save sim results into data directory
        save_sim_results(results, fname)
        
        pprint.pprint(results)

def run_main(bprj):
    interp_method = 'spline'
    sim_env = 'TT'
    n_spec_file = 'specs_mos_char/nch_w0d5_100nm.yaml'
    p_spec_file = 'specs_mos_char/pch_w0d5_100nm.yaml'
    lch = 100e-9
    intent = 'lvt'

    db_n = get_db(n_spec_file, intent, interp_method=interp_method,
                    sim_env=sim_env)
    db_p = get_db(p_spec_file, intent, interp_method=interp_method,
                    sim_env=sim_env)

    specs = dict(prj=bprj,
                db_n=db_n,
                db_p=db_p,
                sim_env=sim_env,
                vdd=1.0,
                cpd=5e-15,
                cload=20e-15,
                nf_n_vec=[2, 4],
                nf_p_vec=[7, 4],
                rf_vec=[2e3, 2e3]
                )
    
    results = sim_tia_inverter(**specs)

if __name__ == '__main__':
    local_dict = locals()
    if 'bprj' not in local_dict:
        print('Creating BAG project')
        bprj = BagProject()
    else:
        print('Loading BAG project')
        bprj = local_dict['bprj']

    run_main(bprj)
    print("done")
