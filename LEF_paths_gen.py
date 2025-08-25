import sys
import numpy as np
import pyximport; pyximport.install(
    setup_args={"include_dirs":np.get_include()},
    reload_support=True)
from looplib_bacterial_gal import bacterial_no_bypassing, bacterial_bypassing
import os, sys, time, shutil
import h5py
"""
# Variables that we need to feed these hungry babies:
- Bacterium = {
name:               ----
N:                  Number of bins/monomers
terLength:          in kbs
offloadingRates:    ----
loadingRates:       ----
rebindingTimes:     in mins
stepRate:           in kbs per min
backstepRate:       in kbs per min
}

"""

def check_or_create_directory(directory_path):
    if os.path.exists(directory_path):
        user_input = input(f"The directory '{directory_path}' already exists. Overwrite? (y/n): ").strip().lower()
        if user_input == 'y' or user_input=="yes":
            print(f"Overwriting the directory: {directory_path}")
            shutil.rmtree(directory_path)
            os.makedirs(directory_path)
        else:
            print("Operation aborted. Exiting.")
            sys.exit()
    else:
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")

def out_dir_name(bacterium, N, results_dir_label="Test_No_Bypass"):
    return results_dir_label + f"_{bacterium['name']}_L_{bacterium['N']}_SMCs_{N}"


def save_to_h5(l_sites, r_sites, ts, loading_site_probs, lef_lifetimes, filename='data.h5'):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('l_sites', data=l_sites)
        f.create_dataset('r_sites', data=r_sites)
        f.create_dataset('ts', data=ts)
        f.create_dataset('loading_site_probs', data=loading_site_probs)
        f.create_dataset('lef_lifetimes', data=lef_lifetimes)

def run_sims(bacterium, num_smcs, burn_in_time, simulation_time_min, num_sims, output_directory, delta_t_sec = 1, results_dir_label = None):
    
    delta_t = delta_t_sec/60

    if results_dir_label is not None:
        results_dir = out_dir_name(bacterium, num_smcs, results_dir_label = results_dir_label)
    else:
        results_dir = out_dir_name(bacterium, num_smcs)

    results_dir = os.path.join(output_directory, results_dir)
    check_or_create_directory(results_dir)

    for i in range(num_sims):
        p = {}
        p['L']  = bacterium['N']
        p['N']  = num_smcs


        p['BURNIN_TIME'] = burn_in_time
        p['T_MAX']  = simulation_time_min

        p['R_OFF']  = bacterium["offloadingRates"]
        p['R_ON']   = bacterium["loadingRates"]
        p['REBINDING_TIME'] = bacterium["rebindingTimes"]

        p['R_EXTEND'] = bacterium["stepRate"]
        p['R_SHRINK'] = bacterium["backstepRate"]   
        
        p['N_SNAPSHOTS'] = p['T_MAX'] // delta_t  
        p['PROCESS_NAME'] = b'Test'         # b'proc'

        t_start=time.perf_counter()
        l_sites, r_sites, ts, loading_sites_count, lef_lifetimes = bacterial_no_bypassing.simulate(p, verbose=False) #perform a new simulation
        t_end = time.perf_counter()

        print(f"performed sim {i} in {t_end-t_start:0.4f} s")

        save_to_h5(l_sites, r_sites, ts, loading_sites_count, lef_lifetimes, os.path.join(results_dir, f"simulation_{i}.h5"))
    return results_dir

def make_loading_profile(L, ori = 0.5, A = 1.0, d = -50, shift = 0):
    # Generate L evenly spaced values in [0, 1)
    x_vals = np.linspace(0, 1, L, endpoint=False)
    # Periodic distance from ori
    dist_to_ori = np.minimum(np.abs(x_vals - ori), 1 - np.abs(x_vals - ori))
        # Evaluate the function
    prob_profile = A * np.exp(dist_to_ori * d) + shift
    # Normalize so it sums to 1 (for use as probabilities)
    prob_profile /= prob_profile.sum()

    return x_vals, prob_profile

def make_unbinding_rates(L, base_rate=0.01, ter_multiplier=1e6, ter_region_perc = 10):
    """
    Returns an array of unbinding rates for a genome of length L.
    - Ori is at the center.
    - Ter is x% of the genome: first x/2% and last x/2%.
    - Unbinding rate is base_rate everywhere except Ter, where it is base_rate * ter_multiplier.
    """
    rates = np.ones(L) * base_rate
    ter_size = int(float(ter_region_perc/200) * L)
    # Set Ter region at both ends
    rates[:ter_size] = base_rate * ter_multiplier
    rates[-ter_size:] = base_rate * ter_multiplier
    return rates