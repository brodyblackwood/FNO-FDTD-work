from os.path import supports_unicode_filenames
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sci
from scipy.ndimage import gaussian_filter
import scipy.io

#Basic 1D FDTD simulation with simple absorbing boundaries. Uncomment plotting to show plots of source propogating
def FDTD1D_simulate(jmax, nmax, lambda_min, source_function, eps_r=1.0, mu_r=1.0, resolution_mult=1):
  dx = lambda_min/20
  dt = dx/sci.c
  eps_r1 = eps_r if hasattr(eps_r, '__len__') else np.ones(jmax) * eps_r
  mu_r1 = mu_r if hasattr(mu_r, '__len__') else np.ones(jmax) * mu_r
  eps = eps_r1 * sci.epsilon_0
  mu = mu_r1 * sci.mu_0
  imp0 = np.sqrt(sci.mu_0/sci.epsilon_0)

  #Source center
  jsource = 10

  # Initialize fields for the simulation loop
  Ex = np.zeros(jmax, dtype=np.float32)
  Hz = np.zeros(jmax, dtype=np.float32)
  Ex_prev = np.zeros(jmax, dtype=np.float32)
  Hz_prev = np.zeros(jmax, dtype=np.float32)
  Ex_history = np.zeros((nmax, jmax), dtype=np.float32)
  Hz_history = np.zeros((nmax, jmax), dtype=np.float32)


  for n in range(nmax):
    # Update magnetic field boundaries
    Hz[jmax-1] = Hz_prev[jmax-2]

    #Update magnetic field
    Hz[:jmax-1] = Hz_prev[:jmax-1] + dt/(dx*mu[1:]) * (Ex[1:jmax] - Ex[:jmax-1])
    Hz_prev = Hz

    #Magnetic Field Source
    Hz[jsource-1] -= source_function(n)/imp0
    Hz_prev[jsource-1] = Hz[jsource-1]

    #Update electric field boundaries
    Ex[0] = Ex_prev[1]

    #Update electric field
    Ex[1:] = Ex_prev[1:] + dt/(dx*eps[1:]) * (Hz[1:] - Hz[:jmax-1])
    Ex_prev = Ex

    # Electric Field Source
    Ex[jsource] += source_function(n+1)
    Ex_prev[jsource] = Ex[jsource]

    Ex_history[n] = Ex
    Hz_history[n] = Hz

    #Optional Plotting
    # epsrplt = eps_r/eps_r.max()
    # if n % 10 == 0:
    #   plt.plot(Ex)
    #   plt.plot(epsrplt)
    #   plt.ylim([-1,1])
    #   plt.show()
    #   plt.close()


  return Ex_history, Hz_history

def generate_training_data(n_samples=1, jmax=500, nmax=3000, lambda_min=350e-9, lambda_range=(400e-9, 700e-9),  tau_range=(10, 50), amplitude_range=(0.5, 2.0), phase_range=(0, 2*np.pi), resolution_multiplier=1):
  training_E_fields = []
  training_H_fields = []
  test_E_fields = []
  test_H_fields = []
  material_profiles = []

  # Jmax is the number of cells for the material part (before padding)
  # Nmax is the total number of time steps for the simulation
  jmax_material_part = int(jmax * resolution_multiplier) # This scales the material cells
  nmax_sim_total = int(nmax * resolution_multiplier) #This scales the total simulation steps

  # dx and dt for the current resolution
  dx = lambda_min / (20 * resolution_multiplier) # Smaller resolution_multiplier means larger dx (lower spatial res)
  dt = dx / sci.c # Corresponding dt for stability, standard FDTD

  # tau is related to source temporal pulse width
  # Determines number of reference steps for the delay, so that full pulse can form by input time.
 
  ref_delay_steps_count = 18 * tau_range[1]

  # Scale this reference number of steps by the resolution_multiplier to get the actual steps for the current simulation
  initial_time_to_extract_data = int(ref_delay_steps_count * resolution_multiplier)
  # Ensure it's at least 1 step if scaled to 0
  if initial_time_to_extract_data == 0:
      initial_time_to_extract_data = 1

  # Calculate the physical length of the vacuum region based on a reference dx, and then scale
  # At resolution_multiplier=1, dx_ref = lambda_min/20.
  # The physical length of the vacuum region should be constant.
  
  physical_vacuum_length_ref = ref_delay_steps_count * (lambda_min / 20)

  # Now, calculate how many cells (jvac) are needed for this constant `physical_vacuum_length_ref`
  # given the current dx.
  jvac = int(physical_vacuum_length_ref / dx)
  # Ensure it's at least 1 cell if scaled to 0
  if jvac == 0:
      jvac = 1

  jtotal = 2 * jvac + jmax_material_part

  x_grid = np.linspace(0, dx * jtotal, jtotal)
  t_grid = np.linspace(0, dt * nmax_sim_total, nmax_sim_total)

  print(f"Minimum time steps: {initial_time_to_extract_data}")
  for i in range(n_samples):


    #Generating random source parameters
    source_lambda = np.random.uniform(*lambda_range)
    source_tau = np.random.uniform(*tau_range) # source_tau is in terms of steps
    source_amplitude = np.random.uniform(*amplitude_range)
    source_phase = np.random.uniform(0, 2*np.pi)

    def source_function(t):
      # Derived quantities
      w0 = 2 * np.pi * sci.c / source_lambda
      t0 = source_tau*3 # t0 is in terms of steps
      # Multiply t (step index) by dt (physical time per step) for the sine wave's phase
      wave = source_amplitude * np.exp(-((t-t0)**2) / source_tau**2) * np.sin(w0 * t * dt + source_phase)
      return wave

    #Generating random material properties
    eps_r = 1 + 4*gaussian_filter(np.random.rand(jmax_material_part), sigma = 5)
    film_num = np.random.randint(0,5)
    for i in range(film_num):
      film_width = 50
      left = np.random.randint(1,240)
      eps_r[left:left+film_width] = 10
    eps_r = np.pad(eps_r, pad_width=jvac, mode='constant', constant_values=1)

    mu_r = np.ones(jtotal)




    #Running simulation
    E_history, H_history = FDTD1D_simulate(jtotal, nmax_sim_total, lambda_min, source_function, eps_r, mu_r, resolution_multiplier)

    material_profiles.append(eps_r)
    training_E_fields.append(E_history[initial_time_to_extract_data])
    training_H_fields.append(H_history[initial_time_to_extract_data])
    test_E_fields.append(E_history[nmax_sim_total-1]) # Extract from the last simulated step
    test_H_fields.append(H_history[nmax_sim_total-1]) # Extract from the last simulated step

    if (i + 1) % 100 == 0:
      print(f"Generated {i+1}/{n_samples} samples")

  training_data = np.stack([np.array(training_E_fields), np.array(training_H_fields), np.array(material_profiles)], axis=0)
  test_data = np.stack([np.array(test_E_fields), np.array(test_H_fields), np.array(material_profiles)], axis=0)
  # Return numpy arrays
  return training_data, test_data, np.arange(jtotal)*dx, np.arange(nmax)*dt, jtotal, initial_time_to_extract_data

training_data, test_data, x_grid, t_grid, jtotal, initial_time_end = generate_training_data(n_samples=20)

# Specify the file path to save the data
file_path = r'E:\FDTD_FNO_Project\Generated Data\FDTD_data_2.mat' 

# Save the data to the .mat file
scipy.io.savemat(file_path, {
    'training_data': training_data,
    'test_data': test_data,
    'x_grid': x_grid,
    't_grid': t_grid
})

print(f"Data saved to {file_path}")