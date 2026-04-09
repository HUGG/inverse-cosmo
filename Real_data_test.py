import numpy as np
import pandas as pd
from all_functions import inverseNA, inverseMCMC, forward


# Reading the sample-related data from an input datafile
sample_data = pd.read_csv('Synthetic_data/Synthetic_data_simple_ero/Synthetic_data_simple_reference.txt', delimiter = '\t')

# Printing the number of samples found
print(f'The number of samples: {len(sample_data.index)}')


# Building a dictionary for the bounds (min, max) of the model parameters. To predefine a value, give it an integer/float value.
parameter_bounds = {'First deglaciation': (41, 60),
                    'First glaciation': (21, 40),
                    'Last deglaciation': (5, 20),
                    'Block erosion': (0, 4),
                    'Constant erosion': (0, 2)
                    }


#Defining the inversion mode: MC = pure Monte Carlo, NA = Neighborhood algorithm (neighpy), MCMC = Markov Chain Monte Carlo (emcee)

#inversion_mode = 'MC'
inversion_mode = 'NA'
#inversion_mode = 'MCMC'

# Building a dictionary for the inversion parameters

if inversion_mode == 'MC':
    inversion_params = {'ns' : 1,
                        'nr' : 1,
                        'ni' : 50000,
                        'n' : 1,
                        'n_resample' : 0
                        }

if inversion_mode == 'NA':
    inversion_params = {'ns': 400,
                        'nr': 200,
                        'ni': 2000,
                        'n': 50,
                        'n_resample': 5000
                        }
    
elif inversion_mode == 'MCMC':
    inversion_params = {'nwalkers': 25,
                        'nsteps': 500,
                        'discard': 0,
                        'thin': 1
                        } 
    

# Running the inversion and printing the results

if inversion_mode == 'MC':

    best_params, best_misfit = inverseNA(sample_data, parameter_bounds, inversion_params)


if inversion_mode == 'NA':

    best_params, best_misfit = inverseNA(sample_data, parameter_bounds, inversion_params)


elif inversion_mode == 'MCMC':

    best_params, best_misfit = inverseMCMC(sample_data, parameter_bounds, inversion_params)


print(f'The best set of model parameters: {best_params}')
      
print(f'Misfit of the best model: {best_misfit}')

# Get the original sample data
N_sample = np.array(sample_data.iloc[:,0])    #array containing the nuclide concentrations of the samples [atoms/kg]
N_error = np.array(sample_data.iloc[:,1])     #array containing concentration errors [atoms/kg]
z_sample = np.array(sample_data.iloc[:,2])    #array containing the sample detphs [m]
z_error = np.array(sample_data.iloc[:,3])     #array containing the depth errors [m]
isotope = np.array(sample_data.iloc[:,4])     #isotope array (this shoud match the sample depth and concentration): 1 Be-10, 2 Al-26, 3 C-14 

# Get the parameters from the 'best_params' 
best = list(best_params.values())
glaciation_time = np.zeros(0)
exposure_time = np.zeros(0)
block_erosion = np.zeros(0)

for i in range(len(best)):
    if i == len(best) - 2 and len(best) > 2:
        block_erosion = np.full(len(glaciation_time), best[i]) # as the forward function allows for glaciation specific block erosion
    elif i == len(best) -1 and len(best) > 1:
        const_erosion = best[i]
    elif i % 2 == 0:
        exposure_time = np.append(exposure_time, best[i])
    elif i % 2 != 0:
        glaciation_time = np.append(glaciation_time, best[i])

_, _, _= forward(isotope, glaciation_time, exposure_time, block_erosion, const_erosion, inversion_mode = False, depths = z_sample)

