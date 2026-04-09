import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import global_params as g
from all_functions import synthetic_data, plot_ice_history, plot_concentration_profile


isotope = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
#isotope = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])

#sample_depth = np.array([0]*len(isotope))
sample_depth = np.array([0, 1, 2, 3, 4]*2)
#sample_depth = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
depth_error = np.array([0.025]*len(isotope))

# True parameters
first_degla = 50
last_degla = 10
glaciation = 30

time_ice = np.array([glaciation, 0])
time_degla = np.array([first_degla, last_degla])
block_erosion = np.array([1])
const_erosion = 0.03


# Defining the true model parameters as a global list
g.true_params.extend([first_degla, glaciation, last_degla, block_erosion[0], const_erosion])

# Creating the synthetic samples
N_sample, N_max, N_min, z_sample, z_max, z_min, isotope = synthetic_data(isotope, sample_depth, depth_error, time_ice, time_degla, block_erosion, const_erosion)

# Calculating the error in concentration on the basis of the maximum and minimum concentrations
N_error = ((N_max-N_sample)+(N_sample-N_min))/2


# Creating a DataFrame for the sample data and filling it
sample_data = pd.DataFrame()
sample_data['Concentration [atoms/kg]'] = N_sample
sample_data['Error [atoms/kg]'] = N_error
sample_data['Sample depth [m]'] = z_sample
sample_data['Depth error [m]'] = depth_error
sample_data['Isotope'] = isotope

# Printing the sample data
print(sample_data)

# Producing the figures

ich_fig = plot_ice_history(time_ice, time_degla, block_erosion, const_erosion)
cp_fig = plot_concentration_profile(concentration = N_sample, depth = z_sample, isotope = isotope, lines = '', markers = '*', sample_type = 'synthetic samples', error=None)
ich_fig.savefig('../Figures/Synthetic_tests/Ice_coverage_history', dpi=1000, bbox_inches='tight')
cp_fig.savefig('../Figures/Synthetic_tests/Concentration_profile', dpi=1000, bbox_inches='tight')
plt.show()

# Saving the data into an output file 'Synthetic_data.txt
sample_data.to_csv('Synthetic_data/Synthetic_data_simple_ero/Synthetic_data_simple_surface.txt', sep='\t', index=False)





