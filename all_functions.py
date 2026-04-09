# -*- coding: utf-8 -*-
"""
Functions related to my thesis of forward and inverse modelling of terrestrial 
cosmogenic nuclides to detect past glaciations.

The calculations are based on Vermeesch 2007 and Dunai 2010.

Forward function calculates nuclide concentrations with depth.
Find_times function chooses randomly the glaciation and deglaciation times and Rand_erosion function chooses randomly the amount of block erosion
The glaciation & deglaciation times and erosion are tested using Inverse function.

Lotta Ylä-Mella 5.5.2020
"""

import numpy as np
import time
import global_params as g
from operator import itemgetter
import emcee
from neighpy import NASearcher, NAAppraiser
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.spatial import Voronoi, voronoi_plot_2d
import corner



def plot_ice_history(time_ice, time_degla, block_erosion, const_erosion):
    '''
    Function to create an ice/no ice plot visualizing the ice coverage history.
    
    Parameters:
    time_ice (array) -- array for the glaciation times (i.e. start of ice coverage) [ka]
    time_degla (array) -- array for the deglaciation times (i.e. start of exposure = no ice coverage) [ka]
    block_erosion (array/float) -- array or single value of glacial block erosion [m]
    const_erosion (float) -- constant erosion rate during interglacials [mm/a]
    
    Output:
    A step figure showing the ice coverage history.
    
    '''

    # Adding the present (0 ka) as the last glaciation to end the latest exposure
    if time_ice[-1] != 0:
        time_ice = np.concatenate([time_ice, [0]])

    # Creating y-axis values representing 'ice' and 'no ice' conditions
    ice = [0] * len(time_ice)
    no_ice = [1] * len(time_degla)

    # The values 'ice' and 'no ice' are bound inversely to deglaciation and glaciation times
    # as the step function draws the horizontal lines prior to the y-values ('pre'-step)
    exposure = list(zip(ice, time_ice))
    glaciation = list(zip(no_ice, time_degla))
    
    # The dictionary created above is sorted in descending order
    ice_history = sorted((exposure + glaciation), key=itemgetter(1), reverse=True)

    # After sorting, the dictionary is unzipped into y and x values 
    y, x = zip(*ice_history)

    fig = plt.figure(figsize=(10, 7))

    # Adding the erosion values to the plot as text
    plt.text((time_ice[0]-0.5), 0.9, f'Block erosion: {block_erosion[0]:.1f} m', fontsize=15)
    plt.text((time_degla[0]-0.5), 0.9, f'Constant erosion: {const_erosion:.2f} mm/a', fontsize=15)

    # The ice coverage history is plotted with the step function
    plt.step(x, y, 'k', where='pre')
    # The periods of ice coverage are colored with light blue
    plt.fill_between(x, y, step='pre', alpha=0.2)
    plt.xlabel('Time [ka]', fontsize=20)
    plt.xlim(-1, max(x)+1)
    plt.xticks(fontsize=15)
    plt.ylabel('Ice coverage [%]', fontsize=20)
    plt.yticks([0, 1], ['0', '100'], fontsize=15)
    plt.gca().xaxis.set_inverted(True)
    #plt.gca().xaxis.set_major_locator(ticker.MultipleLocator())
    #plt.title ('Ice coverage history', fontsize=25)
    fig.tight_layout()

    return fig


def plot_concentration_profile(concentration, depth, isotope, lines='-', markers='*', sample_type=None, error=None ):
    '''
    Function for plotting the nuclide concentration profile(s).

    The required parameters include the concentration and corresponding depth and isotope. 
    The linestyle and marker type of the figure can be modified from defaults with inputs 'lines' and 'markers'.
    
    The sample type (the type of data on which the profile is based on) can be given as optional input. 
    It should be an array of strings matching the size of the the nuclide concentrations array. If all the samples are of the same type,
    the sample type can be given as a string and is then included in the title. The default of sample type is None.
    
    Another optional parameter is the concentration error. If the error is given,  x-axis errorbars will be plotted for the samples.
    
    Parameters:
    concentration (array) -- array containing the nuclide concentrations [atoms/kg]
    depth (array) -- array containing the sample depths [m]
    isotope (array) -- the corresponding isotope: 1 Be-10, 2 Al-26, 3 C-14
    lines (str) -- the linestyle to be used, default = '-' (uniform)
    markers (str) -- the marker style to be used, default = '*' (stars)
    sample_type (array/str/None) -- the type of data the profile is based on: e.g. 'original samples' or 'best inverse solution', default = None
    error (array/None) -- array containing the error in concentration, default = None
    
    Output:
    A figure showing the concentration profile for each nuclide.

    '''


    # The figure is set up outside the loop so that the different nuclide concentrations can all be shown in the same figure.
    fig = plt.figure(figsize=(10, 7))
    plt.xlabel('Atoms in kg of quartz', fontsize=20)
    plt.xticks(fontsize=15)
    plt.ylabel('Depth [m]', fontsize=20)
    plt.yticks(fontsize=15)
    plt.gca().yaxis.set_inverted(True)

    # Checking if the sample type is given as input
    #if isinstance(sample_type, type(None)):
    #    plt.title(f'Concentration profile', fontsize=25)
    #elif isinstance(sample_type, str):
    #    plt.title(f'Concentration profile based on {sample_type}', fontsize=25)


    # Finding the different nuclides
    nuclide = list(set(isotope))

    # Looping over the different nuclides
    for i in range(len(nuclide)):
        # Finding the indices corresponding to the current isotope
        index = np.where(isotope==nuclide[i])[0]
        # Sample depth and concentration arrays containing only the data related to the current isotope 
        depth_ns = depth[index] # ns = nuclide-specific
        concentration_ns = concentration[index]

        # Creating a nuclide-specific array for concentration error (if given)
        if error is not None:
            error_ns = error[index]

        # Naming the different nuclides
        if nuclide[i] == 1:
            name = 'Be-10'
        elif nuclide[i] == 2:
            name = 'Al-26'
        elif nuclide[i] == 3:
            name = 'C-14'

        if error is None:
            plt.plot(concentration_ns, depth_ns, marker=markers, linestyle=lines, markersize=10, label=name)
        else:
            plt.errorbar(concentration_ns, depth_ns, xerr=error_ns, marker=markers, linestyle=lines, markersize=10, label=name)

    #plt.grid()
    plt.legend(fontsize=15)
    plt.tight_layout()

    return fig


def forward(isotope, time_ice, time_degla ,block_erosion, const_erosion, inversion_mode, sample_type=None, depths=None):  
    '''
    Function to calculate nuclide concentration with depth. 
    The forward solution is calculated for each isotope separately starting from the isotope represented by the smallest integer
    and the final output arrays 'depth' and 'N_final' are constructed accordingly.

    The required inputs include the isotope(s), glaciation and deglaciation times, glacial block erosion and interglacial constant erosion, and 'inversion_mode'.
    If 'inversion_mode' is set to 'False', ice/no ice -plot and conentration profile plot are produced. With 'inversion_mode' as 'True' no plots are produced.
    
    Optional inputs include the sample type (used for plotting and thus only relevant when 'inversion_mode' is 'False') 
    and depth(s) for which to calculate the forward solution. 
    If the depths are given as input, the isotope array should match the given depths. 
    Otherwise, the solution is calculated for different isotopes in the 'isotope' array, for a depth range from 0 to model_depth.
    
    Parameters:
    isotope (array) -- the isotope(s): 1 Be-10, 2 Al-26, 3 C-14 (corresponding in length to 'depths' or including the different isotopes for which to calculate the concentration profile)
    time_ice (array) -- array for the glaciation times (i.e. start of ice coverage) [ka]
    time_degla (array) -- array for the deglaciation times (i.e. start of exposure = no ice coverage) [ka]
    block_erosion (array) -- array for the amount of erosion instantly after glaciation [m]
    const_erosion (float) -- constant erosion rate during interglacial [mm/a]
    inversion_mode (bool) -- set to True if called inside inverse -> does not produce plots, when set to False -> produces plots
    sample_type (str) -- the type of data the forward calculation is based on: e.g. 'original samples' or 'best inverse solution'
    depths (array) -- array containing the depths for which to calculate the forward solution [m]
    
    Output:
    depth (array) -- depth [m]
    N_final (array) -- final number of nuclides [atoms/kg]
    nuclide_specific (dictionary) -- dictionary for nuclide-specific sample data
 
    '''

    # Constants
    rho = 2650              # density [kg/m3]
    depth_m = 10            # model depth [m]

    Ln = 160                # g/cm2  Vertical attenuation length, neutrons, Vermeesch 2007
    Lsm1 = 738              # g/cm2  Vertical attenuation length, slow muons, Vermeesch 2007
    Lsm2 = 2688             # g/cm2  Vertical attenuation length, slow muons, Vermeesch 2007
    Lfm = 4360              # g/cm2  Vertical attenuation length, fast muons, Vermeesch 2007
    
    # Rename variables
    erosion = block_erosion
    ec = const_erosion      # constant erosion [mm/a]
    
    # An empty dictionary for nuclide-specific sample data (concentration and corresponding depth)
    nuclide_specific = {}

    # Identify different isotopes and calculate the forward solution separately for each
    nuclide = list(set(isotope)) 

    # To execute the calculation separately for each isotope loop over the different isotopes
    for i in range(len(nuclide)):

        # Isotope related constants
        if (nuclide[i] == 1):
            # Be-10
            P_0_g = 3.95        # Production rate, atoms/g/a, Stroeven et al.2015 
            t_half = 1.387e6    # half-life, a, Korschinek et al. 2010
            name = 'Be'
            
            # Relative production
            F0 = 0.9724         # Neutrons
            F1 = 0.0186         # Slow muons
            F2 = 0.004          # Slow muons
            F3 = 0.005          # Fast muons
            
        elif (nuclide[i] == 2):
            # Al-26
            P_0_g = 26.71       # Production rate, atoms/g/a, Stroeven et al. 2016,
            t_half = 7.05e5     # half-life, a, Norris 1983
            name = 'Al'
            
            # Relative production
            F0 = 0.9655         # Neutrons
            F1 = 0.0233         # Slow muons
            F2 = 0.005          # Slow muons
            F3 = 0.0062         # Fast muons
            
        elif (nuclide[i] == 3):
            # C-14
            P_0_g = 15.5        # Production rate, atoms/g/a, Miller 2006
            t_half = 5730       # half-life, a, Dunai 2010 
            name = 'C'
            
            # Relative production
            F0 = 0.83           # Neutrons
            F1 = 0.0691         # Slow muons
            F2 = 0.0809         # Slow muons
            F3 = 0.02           # Fast muons
        
            
        # Time arrays from ka to years
        ti = time_ice*1e3       # a
        td = time_degla*1e3     # a

        
        # Unit conversions to SI
        P_0 = P_0_g * 1000      # atoms/g/a --> atoms/kg/a
        L0 = Ln*10              # g/cm2 --> kg/m2
        L1 = Lsm1*10
        L2 = Lsm2*10
        L3 = Lfm*10
        ec = ec/1000            # mm/a --> m/a

        # Decay constant
        lambda1 = np.log(2)/t_half         


        # Arrays 
        spacing = 0.001                                  # Spacing for arrays
        z = np.arange(0.0,depth_m+spacing,spacing)       # Depth [m], from 0m (the surface) to the model depth
        N = np.zeros(len(z))                             # Number of nuclides
        N_decay = np.zeros(len(z))                       # Decay during glaciation
        N_final_ns = np.zeros(len(z))                    # After every step
        N_erosion = np.zeros(len(z))                     # After erosion and glaciation
        N_ex = np.zeros(len(z))                          # After exposure
        
        neu = np.zeros(len(z))                           # Neutrons
        slow_muon1 = np.zeros(len(z))                    # Slow muons
        slow_muon2 = np.zeros(len(z))                    # Slow muons
        fast_muon = np.zeros(len(z))                     # Fast muons

        
        
        # If the first timestep is glaciation > no nuclides formed > remove the first step
        if (len(ti)>len(td)):
            ti = np.delete(ti, 0)
        # Similarly, if there are more values given for block erosion than there are exposure periods, delete the firs value
        if (len(erosion)>len(td)):
            erosion = np.delete(erosion, 0)
        

        # Loop over glacial cycle: exposure, decay, erosion
        for j in range(len(ti)-1):

            # Exposure
            t_ex = td[j] - ti[j]
            # Glaciation
            t_gla = ti[j] - td[j+1] 
            
            # Production paths
            neu = F0/(lambda1 + ec*rho/L0) * np.exp(-z*rho/L0) * \
            (1 - np.exp(-(lambda1 + ec*rho/L0)*t_ex))
            
            slow_muon1 = F1/(lambda1 + ec*rho/L1) * np.exp(-z*rho/L1) * \
            (1 - np.exp(-(lambda1 + ec*rho/L1)*t_ex))
            
            slow_muon2 = F2/(lambda1 + ec*rho/L2) * np.exp(-z*rho/L2) * \
            (1 - np.exp(-(lambda1 + ec*rho/L2)*t_ex))
            
            fast_muon = F3/(lambda1 + ec*rho/L3) * np.exp(-z*rho/L3) * \
            (1 - np.exp(-(lambda1 + ec*rho/L3)*t_ex))
            
            # Total concentration after exposure
            N_ex = P_0 * (neu + slow_muon1 + slow_muon2 + fast_muon) + \
            N*np.exp(-lambda1*t_ex)        
            
            for k in range(len(z)):
                # Number of nuclides after glaciation
                N_decay[k] = N_ex[k]*np.exp(-lambda1*t_gla)
                # Index of last value
                N_idx = k
            
            #Index of erosion depth
            idx = 0
            
            #Erosion
            # Do not calculate if there is no block erosion
            if erosion[j] != 0:
                # Find the index of erosion depth (rounded to 3 decimals). Depth rounded to 4 decimals
                a = np.where(np.around(z, 4)==np.around(erosion[j],3))
                idx = a[0][0]
                for k in range(len(z)):    
                    if ((k+idx) <= N_idx): 
                        #Inherited nuclides are transferred 
                        new_idx = k+idx 
                        N_erosion[k] = N_decay[new_idx]
                    else:                 
                        #If no inheritance, set to 0
                        N_erosion[k] = 0
            else:
                N_erosion = N_decay
            
            # Rename for the next loop
            N = N_erosion

        # Final exposure
        t_ex = td[-1]

        # If the depths for which to calculate the concentrations were not given the depth array is defined based on a model depth and spacing:
        if isinstance(depths, type(None)):
            depth = z
            N_depth_ns = N
            # Modify the isotope array to match in length of the depth and concentration arrays.
            if i == 0:
                isotope = np.full(len(z), nuclide[i])
            else:
                isotope = np.append(isotope, np.full(len(z), nuclide[i]))
            markers = '' # this prevents use of markers in the concentration profile figure
            lines = '-' # this sets the linestyle to uniform
        else:
            # Find the indices corresponding to the current isotope
            indices = np.where(isotope==nuclide[i])[0]
            depth = depths[indices]  # the depths
            # Based on the depths, find the corresponding concentrations
            N_depth_ns = np.zeros(len(depth)) 
            for l in range(len(depth)):
                index = np.where(np.around(z,4)==depth[l])[0][0]
                N_depth_ns[l] = N[index]
            markers = '*' # this sets the marker type in the concentration profile figure to stars
            lines = '' # this removes lines

        # Calculate the last exposure and the final concentration only for the depths given as input
        
        # Production pathways
        neu = F0/(lambda1 + ec*rho/L0) * np.exp(-depth*rho/L0) * \
        (1 - np.exp(-(lambda1 + ec*rho/L0)*t_ex))
        
        slow_muon1 = F1/(lambda1 + ec*rho/L1) * np.exp(-depth*rho/L1) * \
        (1 - np.exp(-(lambda1 + ec*rho/L1)*t_ex))
        
        slow_muon2 = F2/(lambda1 + ec*rho/L2) * np.exp(-depth*rho/L2) * \
        (1 - np.exp(-(lambda1 + ec*rho/L2)*t_ex))
        
        fast_muon = F3/(lambda1 + ec*rho/L3) * np.exp(-depth*rho/L3) * \
        (1 - np.exp(-(lambda1 + ec*rho/L3)*t_ex))
        

        # Final concentration
        N_final_ns = P_0 * (neu + slow_muon1 + slow_muon2 + fast_muon) +\
        N_depth_ns*np.exp(-lambda1*t_ex)

        # The nuclide-specific concentrations and corresponding depths into dictionary
        if nuclide[i] == 1:
            nuclide_specific['N_Be10'] = N_final_ns
            nuclide_specific['z_Be10'] = depth
        elif nuclide[i] == 2:
            nuclide_specific['N_Al26'] = N_final_ns
            nuclide_specific['z_Al26'] = depth
        elif nuclide[i] == 3:
            nuclide_specific['N_C14'] = N_final_ns
            nuclide_specific['z_C14'] = depth


        # For the first nuclide, create the arrays N_final and z_final
        if i == 0:
            N_final = np.copy(N_final_ns)
            z_final = np.copy(depth)
        # for the rest of the nuclides, add the values to the existing arrays
        else:
            N_final = np.append(N_final, N_final_ns)
            z_final = np.append(z_final, depth)


    #Plot (if inversion_mode = False)
    if inversion_mode == False:
        ich_fig = plot_ice_history(time_ice, time_degla, block_erosion, const_erosion) # this plots the ice coverage history as an 'ice/no ice' plot
        ich_fig.savefig('../Figures/Synthetic_tests/Ice_coverage_history', dpi=1000, bbox_inches='tight')
        cp_fig = plot_concentration_profile(N_final, z_final, isotope, lines, markers, sample_type) # this plots the concentration profiles of each nuclide
        cp_fig.savefig('../Figures/Synthetic_tests/Concentration_profile', dpi=1000, bbox_inches='tight')
        
        plt.show()
    
    return N_final, z_final, nuclide_specific



def synthetic_data(isotope, sample_depth, depth_error, time_ice, time_degla, block_erosion, const_erosion):
    '''
    Function to create synthetic data (samples) -> synthetic reference model. 
    The samples are created separately for each isotope starting from the one represented by the smallest integer
    and the final output arrays are constructed accordingly.
    
    Parameters:
    isotope (array) -- array containing the isotope: 1 = Be-10, 2 = Al-26, 3 = C-14
    sample_depth (array) -- the depth of the synthetic sample(s) [m]
    depth_error (array) -- the error in sample depth(s) (if this is an array, the length should be the same as for the sample depth array) [m]
    
    time_ice (array) -- array for ice coverage in the reference model[ka]
    time_degla(array) -- array for exposure times in the reference model [ka]
    block_erosion (array) -- amount of erosion after glaciation in the reference model [m]
    const_erosion (float) -- constant erosion in the reference model [cm/a]
    
    Output:
    N_sample (array) -- array containing the nuclide concentrations of the synthetic samples
    N_max (array) -- array containing the minimum nuclide concentrations (the concentrations at maximum sample depths)
    N_min (array) -- array containing the maximum nuclide concentrations (the concentrations at minimum sample depths)
    z_sample (array) -- array containing the sample depths [m]
    z_max (array) -- array containing the maximum sample depths (sample depth + depth error) [m]
    z_min (array) -- array containing the minimum sample depths (sample depth - depth error) [m]
    isotope (array) -- array containing the isotopes matching the sample depths and concentrations (in ascending order): 1 = Be-10, 2 = Al-26, 3 = C-14

    '''


    # Forward solution
    _, _, nuclide_specific = forward(isotope, time_ice, time_degla,
                              block_erosion, const_erosion, inversion_mode=True, sample_type='synthetic samples')
    
    
    # If the depth error is specified for each sample individually:
    if len(depth_error) == len(sample_depth):
        depth_err = depth_error
    else: # If the depth error is not specified for each sample individually, use the first value of given depth errors for all the samples
        depth_err = np.full((len(sample_depth)), depth_error[0])


    # For a sample closer to the surface than the error in depth, change the sample depth to be equal to the depth error
    z_sample = np.array([y if x < y else x for x, y in zip(sample_depth, depth_err)])

    # Calculate the maximum and minimum depths with the changed sample depth to avoid negative values of minimum depth 
    z_min = z_sample - depth_err
    z_max = z_sample + depth_err


    # Identify different isotopes and create the synthetic samples separately for each
    nuclide = list(set(isotope))  # this creates a list of the different isotopes in ascending order


    # Loop over the different isotopes
    for i in range(len(nuclide)):

        # Find the indices corresponding to the current isotope
        indices = np.where(isotope==nuclide[i])[0]
        # Create the nuclide-specific arrays
        z_sample_ns = z_sample[indices]
        z_min_ns = z_min[indices]
        z_max_ns = z_max[indices]

        # Extract the nuclide-specific forward solutions from 'nuclide_specific' dictionary
        if nuclide[i] ==1:
            N_orig_ns = nuclide_specific['N_Be10']
            z_orig_ns = nuclide_specific['z_Be10']
        elif nuclide[i] == 2:
            N_orig_ns = nuclide_specific['N_Al26']
            z_orig_ns = nuclide_specific['z_Al26']
        elif nuclide[i] == 3:
            N_orig_ns = nuclide_specific['N_C14']
            z_orig_ns = nuclide_specific['z_C14']

        # Create the nuclide-specific sample concentrations using the whole profile and the given sample depths
        N_sample_ns = np.interp(z_sample_ns, z_orig_ns, N_orig_ns)    

        # Nuclide-specific arrays for max and min values of N
        N_max_ns = np.interp(z_min_ns, z_orig_ns, N_orig_ns)
        N_min_ns = np.interp(z_max_ns, z_orig_ns, N_orig_ns)


        # For the first nuclide, create the concentration arrays: N_sample, N_max, N_min
        if i == 0:
            N_sample = np.copy(N_sample_ns)
            N_max = np.copy(N_max_ns)
            N_min = np.copy(N_min_ns)
        # for the rest of the nuclides, add the values to the existing arrays
        else:
            N_sample = np.append(N_sample, N_sample_ns)
            N_max = np.append(N_max, N_max_ns)
            N_min = np.append(N_min, N_min_ns)

    # Finally sort the isotope array in ascending order to match the other sample-related arrays
    isotope.sort()

    return N_sample, N_max, N_min, z_sample, z_max, z_min, isotope



def find_times(number_of_times, rand_max, rand_min=0, min_dur=0.5, tied=None):
    '''
    Function to choose random times
    
    Parameters:
    number_of_times (int) -- how many exposures the solution must have (=complexity of the solution)
    rand_max (float/int) -- the upper limit (maximum model time) [ka]
    rand_min (float/int) -- lower limit (minimum model time) [ka], default value 0 
    mind_dur (float/int) -- the minimum duration between two glaciations and minimum duration of one glaciation [ka], default value 0.5
    tied (float/int) -- time of tied last deglaciation [ka], default None
    
    Output:
    random_times_ice (array) -- array containing values of starting times of glaciations
    random_times_exposure (array) -- array containing values of starting times of exposures

    '''

    # Variable to break to loop when the conditions are filled
    this = True
    while this == True:
        
        # Get rid of the valueError with large number_of_times
        try:
            # Empty lists for times
            random_times_exposure = []
            random_times_ice = []
            
            # Empty dictionaries for  times
            exposure = {}
            burial = {}
            
            #Exposure
            for x in range(number_of_times):
                # Key for the dictionary
                key_ex = "time_exp{0}".format(x)
                
                if x > 0:
                    # Find other exposure ages, which are smaller than the 
                    # final exposure 
                    prev_key_ex = list(exposure.keys())[x-1]
                    # Find the time. Accuracy 100 years
                    value_ex = np.random.randint(rand_min*10, exposure[prev_key_ex]*10)/10
                else:
                    # Final exposure, longest time ago. Accuracy 100 years
                    value_ex = np.random.randint(rand_min*10,rand_max*10)/10
                
                # Connect key and value
                exposure[key_ex] = value_ex
            
            # Change exposure times to NumPy array
            ex = np.array(list(exposure.values()))
            random_times_exposure = ex
            
            # Ice cover
            for x in range(number_of_times):
                # Key for the dictionary
                key_bur = "time_bur{0}".format(x)
                
                if (x < number_of_times -1):
                    # Find other burial ages (ice cover ages)
                    # Keys for corresponding exposures
                    key_ex1 = list(exposure.keys())[x+1]
                    key_ex2 = list(exposure.keys())[x]
 
                    # The value is smaller than the previous exposure, but 
                    # larger than the "coming" exposure
                    value_bur = np.random.randint(exposure[key_ex1]*10, exposure[key_ex2]*10)/10

                else:
                    # The last value is always 0
                    value_bur = 0
                
                # Connect key and value
                burial[key_bur] = value_bur
                
            # Change ice over times to NumPy array
            bur = np.array(list(burial.values()))
            random_times_ice = bur
            
        except ValueError:
            ex = np.zeros(number_of_times)
            bur = ex
        
        # If first value is tied, change that
        if tied:
            ex[-1]=tied
        
        # Calculate durations between exposure time and ice coverage and vice versa
        durations = ex - bur
        durations2 = bur[:-1] - ex[1:]
        
        #Check that all durations are long enough
        if all(i >= min_dur for i in durations) and all(i >= min_dur for i in durations2):
            this = False

    
    return random_times_exposure, random_times_ice



def rand_erosion(complexity, max_erosion):
    '''
    Function to randomly define the amount of erosion during glacial periods (block erosion).
    
    Parameters:
    complexity (int) -- how complicated models are looked for
    max_erosion (float) -- maximum amount of block erosion at the end of a glaciation [m]
    
    Output:
    block_erosion (array) -- array for the amount of erosion instantly after glaciation [m]
    
    '''

    block_erosion = np.zeros(complexity-1)
    for i in range(complexity-1):
        # In dm/10 to get values in 10 cm gaps
        block_erosion[i] = np.random.randint(0,max_erosion*10)/10   
    
    return block_erosion



def misfit_type1(N_comp, z_comp, N_sample, z_sample, N_max, N_min, z_max, z_min):
    '''
    Function to check if a curve passes through an acceptance box and calculate the misfit (type 1). This can be used in the case of synthetic reference models.

    N_comp and z_comp define the curve that is compared to the reference. N_sample and z_sample define the reference model (samples). 
    N_max, N_min, z_max and z_min define the acceptance boxes (sample specific).
    

    Parameters:
    N_comp (array) -- array of nuclide concentrations to compare
    z_comp (array) -- depth array to compare

    N_sample (array) -- array of sample concentrations (reference)
    z_sample (array) -- array of sample depths (reference)

    N_max (array) -- the maximum concentrations
    N_min (array) -- the minimum concentrations
    z_max (array) -- the maximum depths [m]
    z_min (array) -- the minimum depths [m]

    Output:
    misfit (array) -- array of misfit for samples from each depth
    total_misfit (float) -- the total misfit

    '''

    # An array to store the misfit values for each sample
    misfit = np.zeros(len(z_sample))
    misfit[:] = np.nan             #Set all values to nan

    # Temporary lists for misfit values
    gof_temp = []
    gof = []
    
    # Empty list to store the sample depth
    depth_temp = []


    #Test if curve is inside of acceptance box for every sample
    for j in range(len(z_sample)):

        # Empty list to store distances
        z_temp = []
        
        # Temporal list to save accepted concentrations
        N_temp = []

        #Is the value inside the box. If it is add it to a list
        for i in range(len(N_comp)):
            if (N_comp[i] <= N_max[j]) & (N_comp[i] >= N_min[j]):
                if (z_comp[i] <= z_max[j]) & (z_comp[i] >= z_min[j]):
                    N_temp.append(N_comp[i])
                    z_temp.append(z_comp[i])

        # Convert list to array to be able to calculate mean
        N_temp = np.array(N_temp)
        z_temp = np.array(z_temp)

        # If there is something inside the box
        if len(z_temp > 0):
            # Find corresponding z values from the original 
            z_temp_min = z_temp.min()
            z_temp_max = z_temp.max()

            z_mask_orig = (z_sample <= z_temp_max) & (z_sample >= z_temp_min) 
            z_mask_comp = (z_comp <= z_temp_max) & (z_comp >= z_temp_min)

            
            N_o_masked = N_sample[z_mask_orig]
            N_c_masked = N_comp[z_mask_comp]
            
            # Misfit
            # Compare values that are expected at the depth of observed values
            # Expected is the original forward model (o), observed is the compared model (c)
            gof_temp.append(sum(np.abs(N_o_masked-N_c_masked))/len(N_o_masked))
            
            depth_temp.append(z_sample[j])
        

    # Check that every sample has been accepted, if not, then add the misfit
    # value of each sample to array and leave "empty" place to NaNs
    if len(gof_temp) == len(z_sample):
        for i in range(len(z_sample)):
            misfit[i] = gof_temp[i]
    else:
        for i in range(len(gof_temp)):
            for j in range(len(z_sample)):
                if depth_temp[i] == z_sample[j]:
                    misfit[j] = gof_temp[i]
     
    
    #Check that accepted samples don't have NaNs
    if (len(gof_temp) == len(z_sample)):
        for i in range(len(gof_temp)):
            if np.isnan(gof_temp[i]):
                #If there is NaN, don't add it to accepted
                break
            else: 
                #If all values are ok, accept the model
                gof.append(gof_temp[i])
    

    
    # Find final misfit values. If there is no value, set it to NaN.
    # Otherwise calculate the sum of each sample misfit
    array = np.array(gof)
    total_misfit = 0
    if np.isnan(array.mean()):
        total_misfit = np.nan
    else:
        total_misfit = np.sum(array) 


    return misfit, total_misfit


def misfit_type2(N_comp, N_sample, N_error):
    '''
    Function to calculate the misfit based on Braun et al. 2012 (misfit type 2). This can be used for real data. 

    N_comp defines the model to be compared to the observations. 
    N_sample and z_sample are the observed concentration and depth (of the samples) and N_error is the error in observed concentration.
    
    Parameters:    
    N_comp (array) -- array of nuclide concentrations to compare

    N_sample (array) -- array of sample concentrations (observed)

    N_error (array) -- array for errors in sample concentration (observed)

    Output:
    misfit (array) -- array of misfit for samples from each depth
    total_misfit (float) -- the total misfit

    '''

    # An array for the sample misfit values
    misfit = np.zeros(len(N_sample))
        

    for i in range(len(N_sample)):
        # Calculating the misfit based on an individual sample
        misfit[i] = ((N_comp[i] - N_sample[i]) / N_error[i])**2


    # Calculating the total misfit as the sum of sample misfits
    total_misfit = sum(misfit)
    #print(N_comp[0], N_sample[0], total_misfit)


    return misfit, total_misfit



def MCinverse(N_sample, z_sample, isotope, misfit_type, z_error= -1, N_error= -1, N_max=-1, N_min=-1, z_max=-1, z_min= -1, 
            time_ice_inv=None, time_degla_inv=None, erosion_inv=None, complexity=1,
            tied=None, rand_max=50, max_block_erosion=4, const_erosion=0):
    '''
    Function to find possible solutions of glaciation histories that fit to a specific forward model.

    The required parameters are the sample depths and corresponding concentrations and isotopes. The misfit type must also be chosen. 
    If the misfit type is 1: z_min, z_max, N_min, N_max are required. If the type is 2: z_error and N_error are required.
    The array containing isotopes and all error arrays should be of the same length (or a single value) as the sample depth and concentration arrays.
    The parameters time_ice_inv, time_degla_inv, erosion_inv can be used to preset inversion parameters.
    Other optional parameters include complexity (the number misfit_nsof exposures), tied (tied time of the last deglaciation), 
    rand_max (maximum time of models), max_erosion and const_erosion.
    
    Parameters:
    N_sample (array) -- the nuclide concentrations of the synthetic samples
    z_sample (array) -- the sample depths [m]
    isotope (array) -- the isotope: 1 = Be-10, 2 = Al-26, 3 = C-14
    misfit_type (int) -- the type of misfit calculation: 1 = acceptance box, 2 = Braun et al. 2012 (eq. 10)

    N_error (array) -- the error in sample concentration (may contain only non-negative values)

    N_max (array) -- the minimum nuclide concentrations (may contain only non-negative values)
    N_min (array) -- the maximum nuclide concentrations (may contain only non-negative values)
    z_max (array) -- the maximum sample depths [m] (may contain only non-negative values)
    z_min (array) -- the minimum sample depths [m] (may contain only non-negative values)
    
    time_ice_inv (array) -- array for ice coverage of inverse model [ka]
    time_degla_inv (array) -- array for exposure times of inverse model [ka]
    erosion_inv (array) -- amount of erosion after glaciation in inverse model [m]
    
    #complexity is essentially the number of exposures -> the number of glacial periods and block erosion events is thus complexity-1
    complexity (int) -- how complicated models are looked for, value between 1 and 4, default 1
    tied (float) -- tied time of the last deglaciation, default None
    rand_max (float) -- maximum time of models, default 50 [ka] 
    max_erosion (float) -- maximum erosion [m]
    const_erosion (float) -- constant erosion [cm/a]
    
    Output:
    free_para (array)-- array of free parameters (erosion, isotope, coverage, snow)
    all_info (array) -- array of exposures and glaciations [ka]
    misfit (float) -- total misfit value
    misfit_arr (array) -- array of misfit for samples from each depth (first values are misfits, next ones are depths and last one is the total misfit)
    block_erosion (array) -- array the amount of erosion instantly after glaciation [m]

    '''

    
    # Save other parameters to new array    
    free_para = isotope
    
    # If there is no preset value for inverse solution, find one
    if time_degla_inv is None:
        time_degla_inv, time_ice_inv = find_times(complexity, rand_max=rand_max, 
                                                  tied=tied)

    
    # If erosion value is not preset, find one
    if erosion_inv[0] is None:
        block_erosion = rand_erosion(complexity, max_block_erosion)
    else: 
        # Fill the array with the first value of erosion, if the length is incorrect
        if len(erosion_inv) != complexity-1:
            block_erosion = np.zeros(complexity-1)
            for i in range(len(block_erosion)):
                block_erosion[i] = erosion_inv[0]
        else:     
            block_erosion = erosion_inv

            
    # Store exposures, glaciations and erosion
    all_info = np.zeros((2*complexity + complexity-1))
    
    # Add values to the same array       
    for i in range(complexity):
        all_info[i] = time_degla_inv[i]
        all_info[complexity+i] = time_ice_inv[i]
        if i < complexity-1 :
            all_info[2*complexity+i] = block_erosion[i]
    
    # If the isotope information was not given for each sample, the array is filled with the first given value
    if len(isotope) != len(N_sample):
        isotope = np.full(len(N_sample), isotope[0])

    # Identify different isotopes and calculate the forward solution separately for each
    nuclide = list(set(isotope)) 

    # To execute the calculation separately for each isotope loop over the different isotopes
    for i in range(len(nuclide)):

        # Find the indices corresponding to the current isotope
        index = np.where(isotope==nuclide[i])[0]
        # Sample depth and concentration arrays containing only the data related to the current isotope 
        sample_depth_ns = z_sample[index] # ns = nuclide-specific
        concentration_ns = N_sample[index]


        # Calculate solution with the inverse times
        N_pred, z_pred, _ = forward(nuclide[i], time_ice_inv, time_degla_inv ,
                              block_erosion, const_erosion, inversion_mode= True, depths = sample_depth_ns)
 
        # For the first nuclide, create the arrays N_inverse and z_inverse to store the solutions
        if i == 0:
            N_inverse = np.copy(N_pred)
            z_inverse = np.copy(z_pred)
        # For the rest of the nuclides, add the values to the existing arrays
        else:
            N_inverse = np.append(N_inverse, N_pred)
            z_inverse = np.append(z_inverse, z_pred)

        # Calculate misfit (type 1 or type 2)

        # misfit type 1:
        if misfit_type == 1:
            # check that the needed arrays are given as input and are all non-negative
            if all(x >= 0 for array in (N_max, N_min, z_max, z_min) for x in array):
                # min and max arrays (defining the acceptance box) containing only the data related to the current isotope
                N_max_ns = N_max[index]
                N_min_ns = N_min[index]
                z_max_ns = z_max[index]
                z_min_ns = z_min[index]
                # calculate misfit
                misfit_ns, total_misfit_ns = misfit_type1(N_pred, z_pred, N_sample=concentration_ns, z_sample=sample_depth_ns, N_max=N_max_ns, N_min=N_min_ns, z_max=z_max_ns, z_min=z_min_ns)
            else: # if the needed arrays are not given or any of them is negative, print following message
                print("To calculate misfit as misfit type 1 (acceptance box), N_max, N_min, z_max and z_min are a required input and must all be non-negative.")
        
        # misfit type 2:
        elif misfit_type == 2:
            if all(x >= 0 for x in N_error): # check that the error array is given and non-negative
                # the N_error array (needed for misfit type 2 calculation) containing only the data related to the current isotope
                N_err_ns = N_error[index]
                # calculate misfit
                misfit_ns, total_misfit_ns = misfit_type2(N_pred, concentration_ns, N_err_ns)
            else: #if the needed array if not given or is negative, print the following message
                print("To calculate misfit as 'misfit type 2' (after Braun et al. 2012), N_error is a required input and must be non-negative.")
  
        # For the first nuclide, create an array for sample misfit and set the total misfit to be the nuclide specific misfit
        if i == 0:
            misfit = np.copy(misfit_ns)
            total_misfit = total_misfit_ns
        # for the rest of the nuclides, add to the existing array and sum the nuclide specific total misfit to the current total misfit
        else:
            misfit = np.append(misfit, misfit_ns)
            total_misfit += total_misfit_ns


    # Sort the sample concentration and depth according to the isotope
    N_sample = [N_sample[x] for x in np.argsort(isotope)]
    z_sample = [z_sample[x] for x in np.argsort(isotope)]


    N_concentration = N_inverse
    z_depth = z_inverse


    return N_concentration, z_depth, free_para, all_info, total_misfit, misfit, block_erosion



def inverse(sampled_params):
    '''
    Function to calculate the forward solution and corresponding misfit to the sample data for the sampled model parameters.
    
    Parameters:
    sampled_params (list) -- a list containing the sampled model parameters needed for inversion: first and last deglaciation time, glaciation time, block_erosion and const_erosion
    
    Output:
    total_misfit (float) -- the total misfit of the model
    
    '''


    # Define the sample-related parameters and predefined values as global to be able to use them inside the function
    global N_sample, N_error, z_sample, isotope, parameter_names, sampled_params_names, predefined_values


    glaciation_time = np.zeros(0)
    exposure_time = np.zeros(0)
    block_erosion = np.zeros(0)
    const_erosion = 0

    # Build a dictionary, 'sampled_params_dict', from the 'bounds_keys' and 'sampled_params' lists
    sampled_params_dict = dict(zip(sampled_params_names, sampled_params))

    # Merge the 'sampled_params' and 'predefined_values' dictionaries
    model_params_dict = {**sampled_params_dict, **predefined_values}

    # Extract the values from 'model_params_dict' in the original order of 'parameter_names'
    model_params = [model_params_dict[k] for k in parameter_names]

    # Get the inverse parameters from the 'model_params' list
    for i in range(len(model_params)):
        if i == len(model_params) - 2 and len(model_params) > 2:
            block_erosion = np.full(len(glaciation_time), model_params[i]) # as the forward function allows for glaciation specific block erosion
        elif i == len(model_params) -1 and len(model_params) > 1:
            const_erosion = model_params[i]
        elif i % 2 == 0:
            exposure_time = np.append(exposure_time, model_params[i])
        elif i % 2 != 0:
            glaciation_time = np.append(glaciation_time, model_params[i])

    # Set the last glaciation as 0
    glaciation_time = np.append(glaciation_time, 0)


    # Print the model parameters
    #print(f' \nExposure times: {exposure_time}')
    #print(f'Glaciation time: {glaciation_time}')
    #print(f'Block erosion: {block_erosion}')
    #print(f'Constant erosion: {const_erosion}')

    # The total exposure
    total_exposure = sum(exposure_time) - sum(glaciation_time)
    #print(f'Total exposure: {total_exposure}')



    # Calculate solution with the inverse times
    N_pred, _, nuclide_specific = forward(isotope, glaciation_time, exposure_time,
                             block_erosion, const_erosion, inversion_mode= True, depths = z_sample)


    # Calculate misfit:


    # If the error in sample concentration was not given for each sample, the array is filled with the first given value
    if len(N_error) != len(N_sample):
        N_error = np.full(len(N_sample), N_error[0])
    

    # Identify different isotopes to sort the arrays in the same way as the concentration calculated by forward
    nuclide = list(set(isotope)) 
    order = []
    N_sample_ordered = np.zeros(0)
    N_error_ordered = np.zeros(0)
    # Find the indices corresponding to each isotope and sort
    for i in range(len(nuclide)):
        order = np.where(isotope==nuclide[i])[0]
        N_sample_ordered = np.append(N_sample_ordered, N_sample[order])
        N_error_ordered = np.append(N_error_ordered, N_error[order])
        
    # Misfit type 2:
    if all(x >= 0 for x in N_error_ordered): # check that the error array is given and non-negative
        # calculate misfit
        _, total_misfit = misfit_type2(N_pred, N_sample_ordered, N_error_ordered)
    else: #if the needed array is not given or is negative, print the following message
        print("To calculate misfit as 'misfit type 2' (after Braun et al. 2012), N_error is a required input and must be non-negative.")


    # If there is data for different isotopes, calculate the concentration ratio of different isotopes

    ns_keys = list(nuclide_specific.keys())

    if 'N_Al26' in ns_keys and 'N_Be10' in ns_keys:
        z_AlBe = []
        AlBe_ratio = []
        for i in range(len(nuclide_specific['N_Al26'])):
            for j in range(len(nuclide_specific['N_Be10'])):
                if nuclide_specific['z_Al26'][i] == nuclide_specific['z_Be10'][j]:
                    z_AlBe.append(nuclide_specific['z_Al26'][i])
                    AlBe_ratio.append(nuclide_specific['N_Al26'][i] /nuclide_specific['N_Be10'][j])
        #if len(AlBe_ratio) != 0:
        #    print(f'Aluminium-26/Beryllium-10 -ratios: {", ".join(str(x) for x in AlBe_ratio)}')
        #    print(f'at depths [m]: {", ".join(str(x) for x in z_AlBe)}')

    if 'N_C14' in ns_keys and 'N_Be10' in ns_keys:
        z_CBe = []
        CBe_ratio = []
        for i in range(len(nuclide_specific['N_C14'])):
            for j in range(len(nuclide_specific['N_Be10'])):
                if nuclide_specific['z_C14'][i] == nuclide_specific['z_Be10'][j]:
                    z_CBe.append(nuclide_specific['z_C14'][i])
                    CBe_ratio.append(nuclide_specific['N_C14'][i]/nuclide_specific['N_Be10'][j])
        #if len(CBe_ratio) != 0:
        #   print(f'Carbon-14/Beryllium-10 -ratios: {", ".join(str(x) for x in CBe_ratio)}')
        #   print(f'at depths [m]: {", ".join(str(x) for x in z_CBe)}')

    if 'N_C14' in ns_keys and 'N_Al26' in ns_keys:
        z_CAl = []
        CAl_ratio = []
        for i in range(len(nuclide_specific['N_C14'])):
            for j in range(len(nuclide_specific['N_Al26'])):
                if nuclide_specific['z_C14'][i] == nuclide_specific['z_Al26'][j]:
                    z_CAl.append(nuclide_specific['z_C14'][i])
                    CAl_ratio.append(nuclide_specific['N_C14'][i]/nuclide_specific['N_Al26'][j])
        #if len(CAl_ratio) != 0:
        #   print(f'Carbon-14/Aluminium-26 - ratios: {", ".join(str(x) for x in CAl_ratio)}')
        #   print(f'at depths [m]: {", ".join(str(x) for x in z_CAl)}')



    #print(f'Mean difference between predicticted and sample concentrations: {np.mean(abs(N_pred-N_sample))}')
    #print(f'Total misfit: {total_misfit}')
    

    return total_misfit



def inverseNA(sample_data, parameter_bounds, inversion_params):
    
    '''
    Function to find possible solutions of glaciation histories (with two exposures and one glaciation) utilizing the neighborhood algorithm (NA).

    Required input include the sample-related data in 'sample_data', bounds for the model parameters in 'parameter_bounds' 
    and parameters needed for setting up the model (/the sampler) in 'inversion_params'
        
    Parameters:

    sample_data (dataframe) -- dataframe for the sample-related data (sample depth, concentration, error and isotope)
 
    parameter_bounds (dictionary) -- dictionary for the bounds of the model parameters: values should be either tuples (min, max) or for predefined values int/float

    inversion_params (dictionary) -- dictionary for the inversion parameters (used for setting up the model)

    Output:

    best_params (dictionary) -- dictionary containing the best set of model parameters

    best_misfit (float) -- the total misfit of the best model
    
    '''

    # Start the timer to get the execution time of the inverse model
    exec_start = time.time()

    # Create global arrays for sample-related data
    global N_sample, N_error, z_sample, isotope, parameter_names, sampled_params_names, predefined_values
            

    # Extract the sample-related data from the dataframe to separate arrays
    N_sample = np.array(sample_data.iloc[:,0])    #array containing the nuclide concentrations of the samples [atoms/kg]
    N_error = np.array(sample_data.iloc[:,1])     #array containing concentration errors [atoms/kg]
    z_sample = np.array(sample_data.iloc[:,2])    #array containing the sample detphs [m]
    isotope = np.array(sample_data.iloc[:,4])     #isotope array (this shoud match the sample depth and concentration): 1 Be-10, 2 Al-26, 3 C-14 


    # Parameter names and values from the parameter_bounds dictionary
   
    print(f'Bounds dictionary: {parameter_bounds}')
    # Pick the parameter names from the dictionary to a list
    parameter_names = list(parameter_bounds.keys())
    #print(f'Parameter names: {parameter_names}')
    # Pick the min and max values of the parameters from the dictionary to a list
    bounds = [] # empty list for the parameter bounds with min an max values
    sampled_params_names = [] # empty list for the corresponding keys (/parameter names)
    predefined_values = {} # empty dictionary for the parameters with a single given value (predefined)
    for key, value in parameter_bounds.items():
        if isinstance(value, tuple): # 'bounds' will contain only parameters that were given min and max values
            bounds.append(value) # this list of values is used as bounds for the Neighbourhood Algorithm
            sampled_params_names.append(key) # this list contains the corresponding keys
        elif isinstance(value, (int, float)):
            predefined_values.update({key: value}) # this dictionary will contain the predefined parameters that were given only one value
        else:
            raise TypeError('All the values in "parameter_bounds" should be either tuples for (min, max) or ints/floats for predefined values.')
    print(f'Bounds for NA: {dict(zip(sampled_params_names, bounds))}')
    print(f'Predefined parameters: {predefined_values}')

    def objective(params):

        misfit = inverse(params)

        return misfit

    # Initialize the parameters for the searcher using the dictionary inversion_params
    ns =inversion_params['ns'] # number of samples per iteration
    nr = inversion_params['nr'] # number of cells to resample
    ni = inversion_params['ni'] # size of initial random search
    n =  inversion_params['n'] # number of iterations

    # Initialize the NA searcher
    searcher = NASearcher(
        objective,
        ns, nr, ni, n,
        bounds,
    )

    # Run the direct search phase
    searcher.run()

    # If the 'n_resample' parameter is set at zero, skip the appraiser phase
    if inversion_params['n_resample'] != 0:

        # Initialize the NA appraiser
        appraiser = NAAppraiser(
            initial_ensemble=searcher.samples,
            log_ppd= -searcher.objectives,
            bounds=bounds,
            n_resample= inversion_params['n_resample'],
            n_walkers=1, # the number of walkers is always set to 1 as multiple walkers does not seem to work
            )

        # Run the appraiser phase, print the results
        appraiser.run()

        print(f'Appraiser mean: {appraiser.mean}')
        print(f'Appraiser mean error: {appraiser.sample_mean_error}')
        print(f'Appraiser covariance: {appraiser.covariance}')
        print(f'Appraiser covariance error: {appraiser.sample_covariance_error}')


    # Extract the best set of parameters and build a dictionary of them

    best_idx = np.argmin(searcher.objectives) # the index of best solution (minimum misfit)
    best_sampled = searcher.samples[best_idx] # the best sampled parameters

    best_dict = dict(zip(sampled_params_names, best_sampled)) # Build a dictionary of the best sampled params

    best_dict.update(predefined_values) # Merge the predefined values into 'best_dict'

    best = [best_dict[k] for k in parameter_names] # Extract the values from 'best_dict' in the original order accroding to 'parameter_names'

    best_params = dict(zip(parameter_names, best)) # Build a dictionary of the best parameters in the correct orded with 'parameter_names' and 'best'

    # Extract the misfit of the best model

    best_misfit = searcher.objectives[best_idx]

    # Stop the timer to get the execution time of the inverse model
    exec_end = time.time()
    minutes = (exec_end-exec_start)//60
    seconds = ((exec_end-exec_start)/60 - minutes) * 60
    print(f'Inversion completed in {int(minutes)} minutes and {int(seconds)} seconds.')

    # Produce figures

    # Misfit plot
    plt.figure(figsize=(10, 7))
    plt.plot(searcher.objectives, '.')
    plt.scatter(best_idx, searcher.objectives[best_idx], c='r', s=10, zorder=10)
    plt.axvline(searcher.ni, c='k', ls='--')
    plt.yscale('log')
    plt.ylabel('Misfit', fontsize=20)
    plt.yticks(fontsize=15)
    plt.xlabel('Number of samples', fontsize=20)
    plt.xticks(fontsize=15)
    plt.title('Misfit evolution', fontsize=25)
    plt.text(0.05, 0.95, 'Initial search', fontsize=15, transform=plt.gca().transAxes, ha='left')
    plt.text(0.95, 0.95, 'Neighborhood search', fontsize=15, transform=plt.gca().transAxes, ha='right')
    plt.savefig('../Figures/Synthetic_tests/NA/NA_misfit', dpi=1000, bbox_inches='tight')

    # Voronoi plots

    # The true parameters (if the data is synthetic) from global_params.py
    if len(g.true_params) != 0:
        true_params = g.true_params
        units = g.true_params_units

    samples = searcher.samples
    nparams = samples.shape[1]

    fig, axs = plt.subplots(nparams-1, nparams-1, figsize=(3*nparams, 2.5*nparams), tight_layout=True)
    plt.suptitle('Voronoi diagram for each parameter pair', fontsize=25)
    for i in range(nparams-1):
        for j in range(nparams-1):
            if j <= i:
                vor = Voronoi(samples[:, [j, i+1]])
                voronoi_plot_2d(vor, ax=axs[i, j], show_vertices=False, show_points=False, line_width=0.5)
                axs[i, j].scatter(best_sampled[j], best_sampled[i+1], c='#DC3220', marker='x', s=100, label='Best model', zorder=10)
                if 'true_params' in dir():
                    axs[i, j].scatter(true_params[j], true_params[i+1], c='#005AB5', marker='x', s=100, label='True model', zorder=10)
                axs[i, j].set_xlim(searcher.bounds[j])
                axs[i, j].set_ylim(searcher.bounds[i+1])
                axs[i,j].tick_params(axis='both', which='major', labelsize=10)
                if j == 0:
                    axs[i, j].set_ylabel(f'{sampled_params_names[i+1]} {units[i+1]}', fontsize=15)
                if i == nparams -2:
                    axs[i, j].set_xlabel(f'{sampled_params_names[j]} {units[j]}', fontsize=15)
            else:
                axs[i, j].set_visible(False)
        handles, labels = axs[1, 0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(
            by_label.values(),
            by_label.keys(),
            fontsize=13,
            loc='lower left',
            bbox_to_anchor=(0.55, 0.5))
    fig.savefig('../Figures/Synthetic_tests/NA/Voronoi_diagrams', dpi=1000, bbox_inches='tight')

    
    # Histograms
    fig, axs = plt.subplots(nparams, 1, figsize=(5, 2.5 * nparams), tight_layout=True)
    plt.suptitle(f'Distributions of the sampled parameters', fontsize=20)    
    for i in range(nparams):
        axs[i].hist(searcher.samples[:, i], bins=15, color='grey', edgecolor='black')
        axs[i].set_xlim(searcher.bounds[i])
        axs[i].set_xlabel(sampled_params_names[i], fontsize=15)
        axs[i].set_ylabel('Count', fontsize=15)
        axs[i].tick_params(axis='both', which='major', labelsize=10)
    fig.savefig('../Figures/Synthetic_tests/NA/NA_histograms', dpi=1000, bbox_inches='tight')
  
  
    plt.show()



    return best_params, best_misfit


def inverseMCMC(sample_data, parameter_bounds, inversion_params):

    '''
    Function to find possible solutions of glaciation histories (with two exposures and one glaciation) utilizing the Markov chain Monte Carlo (MCMC) sampling method.

    Required input include the sample-related data in 'sample_data', bounds for the model parameters in 'parameter_bounds' 
    and parameters needed for setting up the model (/the sampler) in 'inversion_params'

    
    Parameters:

    sample_data (dataframe) -- dataframe for the sample-related data (sample depth, concentration, error and isotope)
 
    parameter_bounds (dictionary) -- dictionary for the bounds of the model parameters: the values should be either (min, max) or [predefined]

    inversion_params (dictionary) -- dictionary for the inversion parameters (used for setting up the model)


    Output:

    best_params (dictionary) -- dictionary containing the best set of model parameters

    best_misfit (float) -- the total misfit of the best model

    '''

    # Start the timer to get the execution time of the inverse model
    exec_start = time.time()

    # Creating global arrays for sample-related data
    global N_sample, N_error, z_sample, isotope, parameter_names, sampled_params_names, predefined_values


    # Extracting the sample-related data from the dataframe to separate arrays
    N_sample = np.array(sample_data.iloc[:,0])    #array containing the nuclide concentrations of the samples [atoms/kg]
    N_error = np.array(sample_data.iloc[:,1])     #array containing concentration errors
    z_sample = np.array(sample_data.iloc[:,2])    #array containing the sample detphs [m]
    isotope = np.array(sample_data.iloc[:,4])     #isotope array (this shoud match the sample depth and concentration): 1 Be-10, 2 Al-26, 3 C-14 


    # Parameter names and values from the parameter_bounds dictionary
   
    print(f'Bounds dictionary: {parameter_bounds}')
    # Pick the parameter names from the dictionary to a list
    parameter_names = list(parameter_bounds.keys())
    #print(f'Parameter names: {parameter_names}')
    # Pick the min and max values of the parameters from the dictionary to a list
    bounds = [] # empty list for the parameter bounds with min an max values
    sampled_params_names = [] # empty list for the corresponding keys (/parameter names)
    predefined_values = {} # empty dictionary for the parameters with a single given value (predefined)
    for key, value in parameter_bounds.items():
        if isinstance(value, tuple): # 'bounds' will contain only parameters that were given min and max values
            bounds.append(value) # this list of values is used as bounds for the MCMC
            sampled_params_names.append(key) # this list contains the corresponding keys
        elif isinstance(value, (int, float)):
            predefined_values.update({key: value}) # this dictionary will contain the predefined parameters that were given only one value
        else:
            raise TypeError('All the values in "parameter_bounds" should be either tuples for (min, max) or ints/floats for predefined values.')
    print(f'Predefined parameters: {predefined_values}')
    print(f'Bounds for MCMC: {dict(zip(sampled_params_names, bounds))}')



    def log_prior(x):
        for val, (low,high) in zip(x, bounds):
            if not (low <= val <= high):
                return -np.inf
        return 0.0
    
    def log_likelihood(x):
        misfit = inverse(x)
        return -misfit
    
    def log_probability(x):
        lp = log_prior(x)
        if not np.isfinite(lp):
            return -np.inf
        ll = log_likelihood(x)
        return lp + ll
    
    # Initializing MCMC setup

    nwalkers = inversion_params['nwalkers']
    nsteps = inversion_params['nsteps']
    discard = inversion_params['discard']
    thin = inversion_params['thin']
    ndim = len(sampled_params_names)

    # Defining the initial state

    p0 = [[np.random.uniform(low, high) for (low, high) in bounds] for _ in range(nwalkers)]

    # Creating the sampler

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_prob_fn = log_probability,
    )

    # Running the MCMC

    sampler.run_mcmc(initial_state = p0, nsteps = nsteps, progress = True)

    # Extracting the results

    chain = sampler.chain
    log_probs = sampler.lnprobability

    # Flattening the chains
    
    flat_samples = chain[:, discard::thin, :].reshape(-1, ndim)
    flat_log_probs = log_probs[:, discard::thin].reshape(-1)
    
    # Extracting the best set of model parameters

    best_idx = np.argmax(flat_log_probs)
    best_sampled = flat_samples[best_idx]

    best_dict = dict(zip(sampled_params_names, best_sampled)) # Build a dictionary of the best sampled params

    best_dict.update(predefined_values) # Merge the predefined values into 'best_dict'

    best = [best_dict[k] for k in parameter_names] # Extract the values from 'best_dict' in the original order accroding to 'parameter_names'

    best_params = dict(zip(parameter_names, best)) # Build a dictionary of the best parameters in the correct orded with 'parameter_names' and 'best'

    # Extracting the misfit of the best model

    best_misfit = - flat_log_probs[best_idx]

    # Stop the timer to get the execution time of the inverse model
    exec_end = time.time()
    minutes = (exec_end-exec_start)//60
    seconds = ((exec_end-exec_start)/60 - minutes) * 60
    print(f'Inversion completed in {int(minutes)} minutes and {int(seconds)} seconds.')



    # Producing figures

    # The true parameters (if the data is synthetic) from global_params.py
    if len(g.true_params) != 0:
        true_params = np.array(g.true_params)
        units = np.array(g.true_params_units)

    # A corner plot
    if len(best_sampled) > 1:
        fig = corner.corner(flat_samples, labels = f'{sampled_params_names} {units}', label_kwargs = {'fontsize': 11},
                            show_titles= True, title_fmt='.2f', title_kwargs={'fontsize':11})
        corner.overplot_lines(fig, best_sampled,c='#DC3220', label='Best model')
        corner.overplot_points(fig, best_sampled[None, :], markersize=10, c='#DC3220')
        corner.overplot_lines(fig, true_params, c='#005AB5', label='True model')
        corner.overplot_points(fig, true_params[None, :], markersize=10, c='#005AB5')
        fig.set_size_inches(13, 11)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), fontsize=11, loc='lower left', bbox_to_anchor=(0.8, 0.5))
        fig.savefig('../Figures/Synthetic_tests/MCMC/Corner_plot', dpi=1000, bbox_inches='tight')

    # Evolution of misfit values
    plt.figure(figsize=(10, 7))
    neg_log_probs = - flat_log_probs
    plt.plot(neg_log_probs, '.')
    plt.scatter(best_idx, neg_log_probs[best_idx], c='#DC3220', label='Best model')
    plt.xlabel('Number of samples', fontsize=20)
    plt.xticks(fontsize=15)
    plt.ylabel('Misfit', fontsize=20)
    plt.yticks(fontsize=15)
    plt.title('Misfit evolution', fontsize=25)
    plt.yscale('log')
    plt.legend(fontsize=15, loc='upper right')
    plt.savefig('../Figures/Synthetic_tests/MCMC/MCMC_misfit', dpi=1000, bbox_inches='tight')

    # MCMC chains -> how succesful is the parameter convergence
    #'seaborn-v=_8-colorblind'
    with plt.style.context('tableau-colorblind10'):
        plt.figure(figsize=(12, ndim * 2))
        for i in range(ndim):
            plt.subplot(ndim, 1, i+1)
            for walker in chain[:, :, i]:
                plt.plot(walker, alpha = 0.4)
            plt.ylabel(sampled_params_names[i], fontsize=12)
            if i == 0:
                plt.title('MCMC chains for each parameter', fontsize=20)
        plt.xlabel('Step', fontsize=12)
        plt.tight_layout()
        plt.savefig('../Figures/Synthetic_tests/MCMC/MCMC_chains', dpi=1000, bbox_inches='tight')

    plt.show()

    return best_params, best_misfit

