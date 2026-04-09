import numpy as np

global isotope, N_sample, z_sample, N_error

isotope = 0
N_sample = 0
z_sample = 0
N_error = 0

first_degla = 50
last_degla = 10
glaciation = 30
block_erosion = np.array([1])
const_erosion = 0.03

true_params = [first_degla, glaciation, last_degla, block_erosion[0], const_erosion]
true_params_units = ['[ka]', '[ka]', '[ka]', '[m]', '[mm/a]']