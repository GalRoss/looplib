import os
from LEF_paths_gen import *

# We assume 4600 kb
L                  = 3000
N                  = 40

kp_per_monomer     = 4600/L     # kb / monomer   
leg_move_speed     = 40         # kb / min
step_rate          = float(leg_move_speed/(4600/L)) # monomer/min
contraction_rate_factor   = 30                      # Extend / Shrink factor (we want faster extension)

burnin_time        = L/2/step_rate * 4              # "Thermalization": Janni recommends n x (Time for one leg to cross half genome)
total_time         = 5 * burnin_time                # Take into account that "sampled time" is total_time - burnin_time
print(f"Burning time is: {burnin_time} minutes")
print(f"Total time is: {total_time} minutes")

life_time = float(500/30) # In minutes
print(f"LEF life time is: {life_time} minutes")

# Load unbinding profile
R_OFF = make_unbinding_rates(L, 
                            base_rate=1./life_time, 
                            ter_multiplier = 1e9,           # Detatchment factor at Ter region
                            ter_region_perc = 10)           # Percentage of genome that is Ter region
xs, loading_probs = make_loading_profile(L, shift=0.02)     # Peaked profile at ori

bacterium = {}
bacterium['name'] = "Bacillus"
bacterium['N'] = L
bacterium["offloadingRates"] = R_OFF
bacterium["loadingRates"] = loading_probs
bacterium["rebindingTimes"] = 2
bacterium["stepRate"] = step_rate
bacterium["backstepRate"] = step_rate/contraction_rate_factor

run_sims(bacterium, N, burn_in_time=burnin_time, simulation_time_min=total_time, num_sims=500, output_directory="LEF_paths/", delta_t_sec=20, results_dir_label= None)
