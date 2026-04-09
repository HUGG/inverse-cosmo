# -*- coding: utf-8 -*-
"""
Functions related to my thesis of forward and inverse modelling of terrestrial 
cosmogenic nuclides to detect past glaciations.

The calculations are based on Vermeesch 2007.

Forward function calculates nuclide concentrations with depth.
Find_times function chooses randomly times that are testes in Inverse function.

Lotta Ylä-Mella 5.5.2020
"""

import numpy as np


def forward(isotope, time_ice, time_degla ,block_erosion, const_erosion):  
    '''
    Function to calculate nuclide concentration with depth.
    
    Parameters:
    isotope -- 1 Be-10, 2 Al-26, 3 C-14
    time_ice -- array for ice coverage [ka]
    time_degla -- array for no ice coverage [ka]
    block_erosion -- array the amount of erosion instantly after glaciation [m]
    const_erosion -- float, constant erosion rate during interglacial [cm/a]
    
    Output:
    z -- depth [m]
    N_final -- final number of nuclides [kg of quartz]


    '''
    # Constants
    rho = 2650              # kg/m3
    depth_m = 10            # model depth, m

    Ln = 160                # g/cm2  Vertical attenuation length, neutrons, Gosse 2001, Vermeesch 2007
    Lsm1 = 738              # g/cm2  Vertical attenuation length, slow muons, Vermeesch 2007
    Lsm2 = 2688             # g/cm2  Vertical attenuation length, slow muons, Vermeesch 2007
    Lfm = 4360              # g/cm2  Vertical attenuation length, fast muons, Vermeesch 2007
    
    # Remname variables
    erosion = block_erosion
    ec = const_erosion      # constant erosion cm/a
    
    # Isotope related constants
    if (isotope == 1):
        # Be-10
        P_0_g = 3.95        # Production rate, atoms/g, Stroeven et al.2015 
        t_half = 1.387e6    # half-life, a, Korschinek et al. 2010
        name = 'Be'
        
        # Relative production
        F0 = 0.9724         # Neutrons
        F1 = 0.0186         # Slow muons
        F2 = 0.004          # Slow muons
        F3 = 0.005          # Fast muons
        
    elif (isotope == 2):
        # Al-26
        P_0_g = 26.71       # Production rate, atoms/g, Stroeven et al. 2016,
        t_half = 7.05e5     # half-life, a, Norris 1983
        name = 'Al'
        
        # Relative production
        F0 = 0.9655         # Neutrons
        F1 = 0.0233         # Slow muons
        F2 = 0.005          # Slow muons
        F3 = 0.0062         # Fast muons
        
    elif (isotope == 3):
        # C-14
        P_0_g = 15.5        # Production rate, atoms/g, Miller 2006
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

    
    #If the first timestep is glaciation > no nuclides formed > remove the first step
    if (len(ti)>len(td)):
        ti = np.delete(ti,0)
    
    # Unit conversions to SI
    P_0 = P_0_g * 1000      # atoms/kg/a
    L0 = Ln*10              # kg/m2
    L1 = Lsm1*10
    L2 = Lsm2*10
    L3 = Lfm*10

    # Decay constant
    lambda1 = np.log(2)/t_half         
    
    # Arrays 
    spacing = 0.001                                  # Spacing for arrays
    z = np.arange(-0.1,depth_m,spacing)              # Depth (m)
    N = np.zeros(len(z))                             # Number of nuclides
    N_decay = np.zeros(len(z))                       # Decay during glaciation
    N_final = np.zeros(len(z))                       # After every step
    N_erosion = np.zeros(len(z))                     # After erosion and glaciation
    N_ex = np.zeros(len(z))                          # After exposure
    
    neu = np.zeros(len(z))                           # Neutrons
    slow_muon1 = np.zeros(len(z))                    # Slow muons
    slow_muon2 = np.zeros(len(z))                    # Slow muons
    fast_muon = np.zeros(len(z))                     # Fast muons
      
    
    # Loop for glacial cycle: exposure, decay, erosion
    for i in range(len(ti)-1):

        # Exposure
        t_ex = td[i] - ti[i]
        # Glaciation
        t_gla = ti[i] - ti[i+1]
        
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
        N_ex = P_0 * (neu + slow_muon1 + slow_muon2 + fast_muon) - \
        (N-N*np.exp(-lambda1*t_ex))        
        
        for j in range(len(z)):
            # Number of nuclides after glaciation
            N_decay[j] = N_ex[j]*np.exp(-lambda1*t_gla)
            # Index of last value
            N_idx = j
        
        #Index of erosion
        idx = 0
         
        #Erosion
        # Do not calculate if there is no erosion
        if erosion[i] != 0:
            # FFind the index of erosion depth. Depth rounded to 4 decimals
            a = np.where(np.around(z,4)==erosion[i])
            idx = a[0][0]
            for j in range(len(z)):    
                if ((j+idx) <= N_idx): 
                    #Inherited nuclides are transferred 
                    new_idx = j+idx 
                    N_erosion[j] = N_decay[new_idx]
                else:                 
                    #If no inheritance, set to 0
                    N_erosion[j] = 0
        else:
            N_erosion = N_decay
        
        # Rename for the next loop
        N = N_erosion

    # Final exposure
    t_ex = td[-1]
    
    # Production pathways
    neu = F0/(lambda1 + ec*rho/L0) * np.exp(-z*rho/L0) * \
    (1 - np.exp(-(lambda1 + ec*rho/L0)*t_ex))
    
    slow_muon1 = F1/(lambda1 + ec*rho/L1) * np.exp(-z*rho/L1) * \
    (1 - np.exp(-(lambda1 + ec*rho/L1)*t_ex))
    
    slow_muon2 = F2/(lambda1 + ec*rho/L2) * np.exp(-z*rho/L2) * \
    (1 - np.exp(-(lambda1 + ec*rho/L2)*t_ex))
    
    fast_muon = F3/(lambda1 + ec*rho/L3) * np.exp(-z*rho/L3) * \
    (1 - np.exp(-(lambda1 + ec*rho/L3)*t_ex))
    
    # Final concentration
    N_final = P_0 * (neu + slow_muon1 + slow_muon2 + fast_muon) +\
    N*np.exp(-lambda1*t_ex)
    
    return N_final, z



def find_times(number_of_times, rand_max, rand_min=0, min_dur=0.5, tied=None):
    '''
    Function to choose random times
    
    Parameters:
    number_of_times -- how many exposures the solution must have
    rand_max -- the upper limit [ka]
    rand_min -- lower limit, default value 0
    mind_dur -- the minimum duration between two glaciations and minimum 
    duration of one glaciation
    tied -- time if last deglaciation is tied [ka]
    
    Output:
    random_times_ice -- array that contains values of starting times of glaciations
    random_times_exposure -- array that contains values of starting times of exposures
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

def inverse(z_chosen, z_error, isotope, time_ice_fwd, time_degla_fwd, erosion_fwd, 
            time_ice_inv=None, time_degla_inv=None, erosion_inv=None, complexity=1,
            tied=None, rand_max=50,max_erosion=4, const_erosion=0):
    '''
    Function to find possible solutions of glaciation histories that fit to 
    a specific forward model.
    
    Parameters:
    z_chosen -- array of sample depths [m]
    z_error -- depth error of the samples [m]
    isotope -- 1 = Be-10, 2 = Al-26, 3 = C-14
    
    time_ice_fwd -- array for ice coverage of forward model [ka]
    time_degla_fwd -- array for exposure times of forward model [ka]
    erosion_fwd -- amount of erosion after glaciation in forward model [m]
    
    time_ice_inv -- array for ice coverage of inverse model [ka]
    time_degla_inv -- array for exposure times of inverse model [ka]
    erosion_inv -- amount of erosion after glaciation in inverse model [m]
    
    complexity -- how complicated models are looked for, value between 1 and 4, default 1
    tied -- tied time of the last deglaciation, default None
    rand_max -- maximum time of models, default 50 [ka]
    max_erosion -- maximum erosion [m]
    const_erosion -- constant erosion [cm/a]
    
    Output:
    free_para -- array of free parameters (erosion, isotope, coverage, snow)
    all_info -- array of exposures and glaciations [ka]
    misfit -- total misfit value
    misfit_arr -- array of misfit for samples from each depth (first values 
    are misfits, next ones are depths and last one is the total misfit)
    '''
    # Rename
    block_erosion_fwd = erosion_fwd
    
    # Forward solution
    N_orig, z_orig = forward(isotope, time_ice_fwd, time_degla_fwd ,
                              block_erosion_fwd, const_erosion)
    
    # Sample depth and errors
    z_max_chosen = z_chosen + z_error
    z_min_chosen = z_chosen - z_error
    
    # Arrays for max and min values of N for the sample
    N_max_chosen = np.interp(z_min_chosen, z_orig, N_orig)
    N_min_chosen = np.interp(z_max_chosen, z_orig, N_orig)
     
    # Save other parameters to new array    
    free_para = np.zeros(1)
    free_para[0] = isotope
    
    # If there is no preset value for inverse solution, find one
    if time_degla_inv is None:
        time_degla_inv, time_ice_inv = find_times(complexity, rand_max=rand_max, 
                                                  tied=tied)
    
    # If erosion value is not preset, find one
    if erosion_inv[0] is None:
        block_erosion = np.zeros(len(time_degla_inv)-1)
        for i in range(len(time_degla_inv)-1):
            # In dm/10 to get values in 10 cm gaps
            block_erosion[i] = np.random.randint(0,max_erosion*10)/10   
    else: 
        # Fill the array with the first value of erosion, if the length is incorrect
        if len(erosion_inv) != len(time_degla_inv)-1:
            block_erosion = np.zeros(len(time_degla_inv)-1)
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
    
    
    # Calculate solution with the inverse times
    N_comp, z_comp = forward(isotope, time_ice_inv, time_degla_inv ,
                              block_erosion, const_erosion)

    # Temporary lists for misfit values
    gof_temp = []
    gof = []
    
    # Empty list to store the sample depth
    depth_temp = []
    
    
    # New array to store misfit values for each sample + total + sample depths
    misfit_arr = np.zeros(2*len(z_chosen)+1)
    misfit_arr[:] = np.nan             #Set all values to nan
    
    #Test if curve is inside of acceptance box for every sample
    for j in range(len(z_chosen)):
        # Empty list to store distances
        z_temp = []
        
        # Temporal list to save accepted concentrations
        N_temp = []

        #Is the value inside the box. If it is add it to a list
        for i in range(len(N_comp)):
            if (N_comp[i] <= N_max_chosen[j]) & (N_comp[i] >= N_min_chosen[j]):
                if (z_comp[i] <= z_max_chosen[j]) & (z_comp[i] >= z_min_chosen[j]):
                    N_temp.append(N_comp[i])
                    z_temp.append(z_comp[i])
        
        # Convert list to array to be able to calculate mean
        N_temp = np.array(N_temp)
        z_temp = np.array(z_temp)

        # If there is something inside the box
        if len(z_temp > 0):
            # Find corresponding z values from the original
            z_min = z_temp.min()
            z_max = z_temp.max()
            
            z_mask_orig = (z_orig <= z_max) & (z_orig >= z_min)
            z_mask_comp = (z_comp <= z_max) & (z_comp >= z_min)
            
            N_o_masked = N_orig[z_mask_orig]
            N_c_masked = N_comp[z_mask_comp]
            
            # Misfit
            # Compare values that are expected at the depth of observated values
            # Expected is the original forward model (o), observed is the compared model (c)
            gof_temp.append(sum(np.abs(N_o_masked-N_c_masked))/len(N_o_masked))
            
            depth_temp.append(z_chosen[j])
        

    # Check that every sample has been accepted, if not, then add the misfit
    # value of each sample to array and leave "empty" place to NaNs
    if len(gof_temp) == len(z_chosen):
        for i in range(len(z_chosen)):
            misfit_arr[i] = gof_temp[i]
    else:
        for i in range(len(gof_temp)):
            for j in range(len(z_chosen)):
                if depth_temp[i] == z_chosen[j]:
                    misfit_arr[j] = gof_temp[i]
     
    # Combine the sample and the misfit
    for i in range(len(z_chosen)):
        misfit_arr[len(z_chosen)+i] = z_chosen[i]
    
    #Check that accepted samples don't have NaNs
    if (len(gof_temp) == len(z_chosen)):
        for i in range(len(gof_temp)):
            if np.isnan(gof_temp[i]):
                #If there is NaN, don't add it to accepted
                break
            else: 
                #If all values are ok, accept the model
                gof.append(gof_temp[i])
    

    
    # Find  final misfit values. If there is no value, set it to NaN.
    # Otherwise calculate the sum of each sample misfit
    array = np.array(gof)
    misfit = 0
    if np.isnan(array.mean()):
        misfit = np.nan
    else:
        misfit = np.sum(array) 


    # Add the total misfit value to the array that has values of each sample
    misfit_arr[-1] = misfit

    return free_para, all_info, misfit, misfit_arr, block_erosion
