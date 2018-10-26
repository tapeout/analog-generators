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
    
def design_pga(db_n, db_p, sim_env,
                vdd, cload, T, vref,
                gain_min, gain_max, fbw_min, pm_min, vonoise_std_max,
                vb_p, vb_n, cap_min, cap_max, error_tol=0.05):
    """
    Inputs:
    Returns:
    """
    # Equations
    # Av0 = cs/cf
    # beta = cf/(cf+cs+cgg)
    # vonoise,phi1 = kBT/cf * 1/beta
    # cload,eff = cload + (1-beta)*cf
    # rout = 1/(beta*gm)
    # vonoise,ota = inoise,ota**2 * rout**2 * 1/(4*rout*cload,eff)
    # bandwidth = 1/(rout*cload,eff)
    kBT = 1.38e-23*T
    
    amp_gain_min = gain_max*2
    amp_fbw_min = fbw_min*gain_max/amp_gain_min
    amp_pm_min = 45
    
    specs = dict(db_n=db_n,
                 db_p=db_p,
                 sim_env=sim_env,
                 vdd=vdd, 
                 cload=cload,
                 vincm=vref,
                 T=T,
                 gain_min=amp_gain_min,
                 fbw_min=amp_fbw_min,
                 pm_min=amp_pm_min,
                 inoiseout_std_max=2e-12,
                 error_tol=0.05,
                 vb_p=vb_p,
                 vb_n=vb_n)
    
    amp_specs = design_diffAmpP(**specs)
    pprint.pprint(amp_specs)
    
    amp_in_op = db_p.query(vgs=vref-amp_specs['vtail'], 
                           vds=amp_specs['vout']-amp_specs['vtail'],
                           vbs=vb_p-amp_specs['vtail'])
    cgg = amp_in_op['cgg']*amp_specs['nf_in']
    gm = amp_in_op['gm']*amp_specs['nf_in']
    
    # Constrain and sweep feedback cap size
    cf_min = cap_min
    cf_max = cap_max/gain_max
    cf_vec = np.arange(cf_min, cf_max, 10e-15)
    for cf0 in cf_vec:
        cfM = cf0*gain_max
        cs = cf0*gain_max
        betaM = cfM/(cfM+cs+cgg)
        beta0 = cf0/(cf0+cs+cgg)
        cload_eff0 = cload + (1-beta0)*cf0
        cload_effM = cload + (1-betaM)*cfM
        rout0 = 1/(beta0*gm)
        routM = 1/(betaM*gm)
        
        # Check against bandwidth
        fbw0 = 1/(2*np.pi*rout0*cload_eff0)
        if fbw0 < fbw_min:
            print("BW_0: {} (FAIL)\n".format(fbw0))
            continue
        print("BW_0: {}".format(fbw0))
        
        fbwM = 1/(2*np.pi*routM*cload_effM)
        if fbwM < fbw_min:
            print("BW_M: {} (FAIL)\n".format(fbwM))
            continue
        print("BW_M: {}".format(fbwM))
        
        # Check against noise
        ## From phi1
        vonoise_phi1_var0 = kBT/cf0 * 1/beta0
        if vonoise_phi1_var0 >= vonoise_std_max**2:
            print("PHI1 VNOISE_0: {} (FAIL)\n".format(np.sqrt(vonoise_phi1_var0)))
            continue
        print("PHI1 VNOISE_0: {}".format(np.sqrt(vonoise_phi1_var0)))
        
        vonoise_phi1_varM = kBT/cfM * 1/betaM
        if vonoise_phi1_varM >= vonoise_std_max**2:
            print("PHI1 VNOISE_M: {} (FAIL)\n".format(np.sqrt(vonoise_phi1_varM)))
            continue
        print("PHI1 VNOISE_M: {}".format(np.sqrt(vonoise_phi1_varM)))
        
        ## From the OTA
        inoise_ota_var = amp_specs['inoiseout_std']**2
        vonoise_ota_var0 = inoise_ota_var * rout0**2 * 1/(4*rout0*cload_eff0)
        if vonoise_ota_var0 >= vonoise_std_max**2:
            print("OTA VNOISE_0: {} (FAIL)\n".format(np.sqrt(vonoise_ota_var0)))
            continue
        print("OTA VNOISE_0: {}".format(np.sqrt(vonoise_ota_var0)))
        
        vonoise_ota_varM = inoise_ota_var * routM**2 * 1/(4*routM*cload_effM)
        if vonoise_ota_varM >= vonoise_std_max**2:
            print("OTA VNOISE_M: {} (FAIL)\n".format(np.sqrt(vonoise_ota_varM)))
            continue
        print("OTA VNOISE_M: {}".format(np.sqrt(vonoise_ota_varM)))
        
        ## Combined
        vonoise_std0 = np.sqrt(vonoise_phi1_var0 + vonoise_ota_var0)
        if vonoise_std0 > vonoise_std_max:
            print("TOTAL VNOISE_0: {} (FAIL)\n".format(vonoise_std0))
            continue
        print("TOTAL VNOISE_0: {}".format(vonoise_std0))
        
        vonoise_stdM = np.sqrt(vonoise_phi1_varM + vonoise_ota_varM)
        if vonoise_stdM > vonoise_std_max:
            print("TOTAL VNOISE_M: {} (FAIL)\n".format(vonoise_stdM))
            continue
        print("TOTAL VNOISE_M: {}".format(vonoise_stdM))

        # TODO: Check against phase margin
        pm = 360
        if pm < pm_min:
            print("PM: {} (FAIL)\n".format(pm))
            continue
        print("PM: {}".format(pm))

        print("(SUCCESS)\n")
        return dict(cs=cs,
                    cf_0=cf0,
                    fbw0=fbw0,
                    fbwM=fbwM,
                    pm=pm,
                    vonoise_std0=vonoise_std0,
                    vonoise_stdM=vonoise_stdM,
                    amp_itail=amp_specs['itail'],
                    amp_nf_in=amp_specs['nf_in'],
                    amp_nf_load=amp_specs['nf_load'],
                    amp_vout=amp_specs['vout'],
                    amp_vtail=amp_specs['vtail'],
                    amp_gain=amp_specs['gain'],
                    amp_fbw=amp_specs['fbw'],
                    amp_pm=amp_specs['pm'],
                    amp_nf_tail=amp_specs['nf_tail'],
                    amp_nf_bias_tail=amp_specs['nf_bias_tail'],
                    amp_nf_bias_in=amp_specs['nf_bias_in'],
                    amp_nf_bias_loadM=amp_specs['nf_bias_loadM'],
                    amp_nf_bias_load1=amp_specs['nf_bias_load1'],
                    amp_iref=amp_specs['iref']
                    )
    raise ValueError("No solution for PGA")
    
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
        db_p=pch_db,
        sim_env=sim_env,
        vdd=0.8,
        cload=300e-15,
        T=300,
        vref=0,
        gain_min=1, 
        gain_max=5,
        fbw_min=1e6,
        pm_min=70, 
        vonoise_std_max=1.0e-3,
        vb_p=1.0, 
        vb_n=0,
        cap_min=7e-15,
        cap_max=1e-12,
        error_tol=0.05,
        )

    pga_specs = design_pga(**specs)
    pprint.pprint(pga_specs)
    print('done')

if __name__ == '__main__':
    run_main()
