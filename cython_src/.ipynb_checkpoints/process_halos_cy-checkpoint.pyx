# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp

"""
Optimized Cython implementation of the process_halos function for astrophysical simulations.
This version eliminates Python overhead, optimizes array access, and parallelizes bin calculations.
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, pow, M_PI
from cython.parallel import prange
import data_io as io

# Import necessary types from astropy
from astropy import units as u
from astropy.constants import k_B, m_p

# Define types for NumPy arrays
ctypedef np.float32_t DTYPE_t
ctypedef np.int64_t ITYPE_t
ctypedef np.uint8_t BTYPE_t

# Global constants
cdef:
    double XH = 0.76
    double Zsun = 0.0198

def process_halos(int start, int end, 
                  double hubble, double scale_factor, double Omega_M, double Omega_L, 
                  double redshift, 
                  object DM_fields, object stellar_fields, object fields, 
                  object pos_h, object vel_h, 
                  object M200c, double boxsize, object radial_bins_inner, 
                  object radial_bins_outer, object T_all, int nbins, int start_halo,
                  object local_DM_density_array, object local_stellar_density_array, 
                  object local_potential_array,
                  object local_density_array, object local_temperature_array, 
                  object local_pressure_array, object local_metallicity_array,
                  object local_radial_gas_velocity_array, object local_rotational_gas_velocity_array,
                  object local_radial_gas_velocity_dispersion_array, object local_gas_velocity_dispersion_array,
                  object local_hot_density_array, object local_hot_temperature_array,
                  object local_hot_pressure_array, object local_hot_metallicity_array,
                  object local_hot_radial_gas_velocity_array, object local_hot_rotational_gas_velocity_array,
                  object local_hot_radial_gas_velocity_dispersion_array, object local_hot_gas_velocity_dispersion_array):
    """
    Process halos from start to end indices, calculating various profiles.
    
    Parameters:
    -----------
    start : int
        Starting halo index
    end : int
        Ending halo index (exclusive)
    hubble : double
        Hubble parameter
    scale_factor : double
        Scale factor
    Omega_M : double
        Matter density parameter
    Omega_L : double
        Dark energy density parameter
    redshift : double
        Redshift
    fields : dict
        Dictionary containing simulation fields
    pos_h : array
        Halo positions
    vel_h : array
        Halo velocities
    M200c : array
        Halo masses (M200c)
    boxsize : double
        Simulation box size
    radial_bins_inner : array
        Inner radii of bins
    radial_bins_outer : array
        Outer radii of bins
    T_all : array or astropy.Quantity
        Temperature of all particles
    nbins : int
        Number of radial bins
    start_halo : int
        Starting halo index for this process
    local_*_array : array
        Output arrays for various profiles
    """
    # Define variables with types
    cdef:
        int ih, halo_index, k, n_particles, idx
        double r1, r2, volume_physical, T_virial_K, T_virial_keV
        double K_to_keV = 8.617333262e-6  # conversion factor from K to keV
        double volume_factor = (4.0 / 3.0) * M_PI
        double kpc_to_cm = 3.085678e21  # kpc to cm conversion
        double msun_to_g = 1.989e33     # Msun to g conversion
        double mass_factor = 1.0e10 / hubble  # conversion for masses
        int num_bins = nbins
        
        # Declare arrays for input data
        np.ndarray[DTYPE_t, ndim=2] pos_h_arr
        np.ndarray[DTYPE_t, ndim=2] vel_h_arr
        np.ndarray[DTYPE_t, ndim=1] M200c_arr
        np.ndarray[DTYPE_t, ndim=1] radial_bins_inner_arr
        np.ndarray[DTYPE_t, ndim=1] radial_bins_outer_arr
        
        # Declare arrays for fields 
        np.ndarray[DTYPE_t, ndim=2] DM_coords_all
        np.ndarray[DTYPE_t, ndim=2] DM_velocities_all
        np.ndarray[DTYPE_t, ndim=1] DM_masses_all

        np.ndarray[DTYPE_t, ndim=2] stellar_coords_all
        np.ndarray[DTYPE_t, ndim=2] stellar_velocities_all
        np.ndarray[DTYPE_t, ndim=1] stellar_masses_all

        np.ndarray[DTYPE_t, ndim=2] coords_all
        np.ndarray[DTYPE_t, ndim=2] velocities_all
        np.ndarray[DTYPE_t, ndim=1] masses_all
        np.ndarray[DTYPE_t, ndim=1] potential_all
        np.ndarray[DTYPE_t, ndim=1] internal_e_all
        np.ndarray[DTYPE_t, ndim=1] electron_abundance_all
        np.ndarray[DTYPE_t, ndim=1] metallicity_all
        
        # Declare output arrays
        np.ndarray[DTYPE_t, ndim=2] local_DM_density_arr
        np.ndarray[DTYPE_t, ndim=2] local_stellar_density_arr
        np.ndarray[DTYPE_t, ndim=2] local_potential_arr
        np.ndarray[DTYPE_t, ndim=2] local_density_arr
        np.ndarray[DTYPE_t, ndim=2] local_temperature_arr
        np.ndarray[DTYPE_t, ndim=2] local_pressure_arr
        np.ndarray[DTYPE_t, ndim=2] local_metallicity_arr
        np.ndarray[DTYPE_t, ndim=2] local_radial_gas_velocity_arr
        np.ndarray[DTYPE_t, ndim=2] local_rotational_gas_velocity_arr
        np.ndarray[DTYPE_t, ndim=2] local_radial_gas_velocity_dispersion_arr
        np.ndarray[DTYPE_t, ndim=2] local_gas_velocity_dispersion_arr
        
        np.ndarray[DTYPE_t, ndim=2] local_hot_density_arr
        np.ndarray[DTYPE_t, ndim=2] local_hot_temperature_arr
        np.ndarray[DTYPE_t, ndim=2] local_hot_pressure_arr
        np.ndarray[DTYPE_t, ndim=2] local_hot_metallicity_arr
        np.ndarray[DTYPE_t, ndim=2] local_hot_radial_gas_velocity_arr
        np.ndarray[DTYPE_t, ndim=2] local_hot_rotational_gas_velocity_arr
        np.ndarray[DTYPE_t, ndim=2] local_hot_radial_gas_velocity_dispersion_arr
        np.ndarray[DTYPE_t, ndim=2] local_hot_gas_velocity_dispersion_arr
        
        # Declare T_all
        np.ndarray[DTYPE_t, ndim=1] T_all_val
        
        # Create arrays for current halo
        np.ndarray[DTYPE_t, ndim=2] DM_coords
        np.ndarray[DTYPE_t, ndim=2] stellar_coords
        np.ndarray[DTYPE_t, ndim=2] coords
        np.ndarray[DTYPE_t, ndim=1] DM_r
        np.ndarray[DTYPE_t, ndim=1] stellar_r
        np.ndarray[DTYPE_t, ndim=1] r
        np.ndarray[DTYPE_t, ndim=1] vhalo  # Changed to 1D
        np.ndarray[BTYPE_t, ndim=1, cast=True] bin_mask, superv_mask
        
        # For bin calculations
        np.ndarray[DTYPE_t, ndim=2] bin_pos, bin_vel, hot_bin_pos, hot_bin_vel
        np.ndarray[DTYPE_t, ndim=1] bin_r, bin_mass, bin_internal_e, bin_X_e, bin_metallicity, bin_potential
        np.ndarray[DTYPE_t, ndim=1] hot_bin_r, hot_bin_mass, hot_bin_internal_e, hot_bin_X_e, hot_bin_metallicity
        double bin_total_mass, hot_bin_total_mass, mu, hot_mu
        np.ndarray[DTYPE_t, ndim=1] bin_T, hot_bin_T
        double mean_T, hot_mean_T, volume_cm3
        
        # Results for normal bins
        double DM_density_profile, stellar_density_profile, potential_profile
        double density_profile, temperature_profile, pressure_profile, metallicity_profile
        double mean_radial_vel, mean_rot_vel, rad_vel_disp, vel_disp_1D
   
        double bin_DM_mass, bin_stellar_mass
     
        # Results for hot bins
        double hot_density_profile, hot_temperature_profile, hot_pressure_profile, hot_metallicity_profile
        double hot_mean_radial_vel, hot_mean_rot_vel, hot_rad_vel_disp, hot_vel_disp_1D
    
    # Initialize arrays with proper type conversion
    pos_h_arr = np.asarray(pos_h, dtype=np.float32)
    vel_h_arr = np.asarray(vel_h, dtype=np.float32)
    M200c_arr = np.asarray(M200c, dtype=np.float32)
    radial_bins_inner_arr = np.asarray(radial_bins_inner, dtype=np.float32)
    radial_bins_outer_arr = np.asarray(radial_bins_outer, dtype=np.float32)
    
    # Extract fields data
    coords_all = np.asarray(fields['Coordinates'], dtype=np.float32)
    velocities_all = np.asarray(fields['Velocities'], dtype=np.float32)
    masses_all = np.asarray(fields['Masses'], dtype=np.float32)
    potential_all = np.asarray(fields['Potential'], dtype=np.float32)
    internal_e_all = np.asarray(fields['InternalEnergy'], dtype=np.float32)
    electron_abundance_all = np.asarray(fields['ElectronAbundance'], dtype=np.float32)
    metallicity_all = np.asarray(fields['GFM_Metallicity'], dtype=np.float32)
 
    DM_coords_all = np.asarray(DM_fields['Coordinates'], dtype=np.float32)
    DM_velocities_all = np.asarray(DM_fields['Velocities'], dtype=np.float32)
    DM_masses_all = np.asarray(DM_fields['Masses'], dtype=np.float32)
 
    stellar_coords_all = np.asarray(stellar_fields['Coordinates'], dtype=np.float32)
    stellar_velocities_all = np.asarray(stellar_fields['Velocities'], dtype=np.float32)
    stellar_masses_all = np.asarray(stellar_fields['Masses'], dtype=np.float32)
   
    # Initialize output arrays
    local_DM_density_arr = np.asarray(local_DM_density_array, dtype=np.float32)
    local_stellar_density_arr = np.asarray(local_stellar_density_array, dtype=np.float32)
    local_potential_arr = np.asarray(local_potential_array, dtype=np.float32)
    local_density_arr = np.asarray(local_density_array, dtype=np.float32)
    local_temperature_arr = np.asarray(local_temperature_array, dtype=np.float32)
    local_pressure_arr = np.asarray(local_pressure_array, dtype=np.float32)
    local_metallicity_arr = np.asarray(local_metallicity_array, dtype=np.float32)
    local_radial_gas_velocity_arr = np.asarray(local_radial_gas_velocity_array, dtype=np.float32)
    local_rotational_gas_velocity_arr = np.asarray(local_rotational_gas_velocity_array, dtype=np.float32)
    local_radial_gas_velocity_dispersion_arr = np.asarray(local_radial_gas_velocity_dispersion_array, dtype=np.float32)
    local_gas_velocity_dispersion_arr = np.asarray(local_gas_velocity_dispersion_array, dtype=np.float32)
    
    local_hot_density_arr = np.asarray(local_hot_density_array, dtype=np.float32)
    local_hot_temperature_arr = np.asarray(local_hot_temperature_array, dtype=np.float32)
    local_hot_pressure_arr = np.asarray(local_hot_pressure_array, dtype=np.float32)
    local_hot_metallicity_arr = np.asarray(local_hot_metallicity_array, dtype=np.float32)
    local_hot_radial_gas_velocity_arr = np.asarray(local_hot_radial_gas_velocity_array, dtype=np.float32)
    local_hot_rotational_gas_velocity_arr = np.asarray(local_hot_rotational_gas_velocity_array, dtype=np.float32)
    local_hot_radial_gas_velocity_dispersion_arr = np.asarray(local_hot_radial_gas_velocity_dispersion_array, dtype=np.float32)
    local_hot_gas_velocity_dispersion_arr = np.asarray(local_hot_gas_velocity_dispersion_array, dtype=np.float32)
    
    # Handle T_all which might be an astropy Quantity
    if hasattr(T_all, 'value'):
        T_all_val = np.asarray(T_all.value, dtype=np.float32)
    else:
        T_all_val = np.asarray(T_all, dtype=np.float32)
    
    # Process each halo
    for ih in range(start, end):
        halo_index = ih - start_halo
        
        # Calculate particle positions relative to halo center
        DM_coords = io.periodic_bcs(DM_coords_all, pos_h_arr[ih], box=boxsize) - pos_h_arr[ih]
        # Calculate radial distance for each particle
        DM_r = np.sqrt(np.sum(DM_coords * DM_coords, axis=1))
 
        # Process each bin
        for k in range(num_bins):
            # Calculate the radius bounds
            r1 = radial_bins_inner_arr[k] / scale_factor * hubble
            r2 = radial_bins_outer_arr[k] / scale_factor * hubble
            
            # Calculate volume in physical units
            volume_physical = volume_factor * (r2**3 - r1**3) * (scale_factor**3 / hubble**3)
            volume_cm3 = volume_physical * kpc_to_cm**3
            
            # Create masks for DM particles
            bin_mask = (DM_r >= r1) & (DM_r < r2)
            
            # --- Process normal gas bin ---
            n_particles = np.sum(bin_mask)
            if n_particles > 0:
                # Extract data for particles in this bin
                sum_mass = np.sum(DM_masses_all[bin_mask])
                bin_DM_mass = sum_mass * mass_factor
                 # Density in g/cm^3
                DM_density_profile = bin_DM_mass / volume_physical
                local_DM_density_arr[halo_index, k] = DM_density_profile
            else :
                local_DM_density_arr[halo_index, k] = 0.0

        # Calculate particle positions relative to halo center
        stellar_coords = io.periodic_bcs(stellar_coords_all, pos_h_arr[ih], box=boxsize) - pos_h_arr[ih]
        # Calculate radial distance for each particle
        stellar_r = np.sqrt(np.sum(stellar_coords * stellar_coords, axis=1))
 
        # Process each bin
        for k in range(num_bins):
            # Calculate the radius bounds
            r1 = radial_bins_inner_arr[k] / scale_factor * hubble
            r2 = radial_bins_outer_arr[k] / scale_factor * hubble
            
            # Calculate volume in physical units
            volume_physical = volume_factor * (r2**3 - r1**3) * (scale_factor**3 / hubble**3)
            volume_cm3 = volume_physical * kpc_to_cm**3
            
            # Create masks for stars
            bin_mask = (stellar_r >= r1) & (stellar_r < r2)
            
            # --- Process normal gas bin ---
            n_particles = np.sum(bin_mask)
            if n_particles > 0:
                # Extract data for particles in this bin
                bin_stellar_mass = np.sum(stellar_masses_all[bin_mask]) * mass_factor 
                stellar_density_profile = bin_stellar_mass / volume_physical
                local_stellar_density_arr[halo_index, k] = stellar_density_profile
            else :
                local_stellar_density_arr[halo_index, k] = 0.0

        # Calculate gas  particle positions relative to halo center
        coords = io.periodic_bcs(coords_all, pos_h_arr[ih], box=boxsize) - pos_h_arr[ih]
        
        # Calculate radial distance for each particle
        r = np.sqrt(np.sum(coords * coords, axis=1))
        
        # Compute halo peculiar velocity - extract the row from the 2D array
        vhalo = vel_h_arr[ih,:] / scale_factor
        
        # Calculate virial temperature
        #T_virial_K = io.virial_temperature(M200c_arr[ih], redshift, mdef='200c',
        #                                  Omega_M=Omega_M, Omega_L=Omega_L, hubble=hubble)
        T_virial_keV = 1.0e6 * K_to_keV
        
        # Process each bin
        for k in range(num_bins):
            # Calculate the radius bounds
            r1 = radial_bins_inner_arr[k] / scale_factor * hubble
            r2 = radial_bins_outer_arr[k] / scale_factor * hubble
            
            # Calculate volume in physical units
            volume_physical = volume_factor * (r2**3 - r1**3) * (scale_factor**3 / hubble**3)
            volume_cm3 = volume_physical * kpc_to_cm**3
            
            # Create masks for normal and hot particles
            bin_mask = (r >= r1) & (r < r2)
            superv_mask = bin_mask & (T_all_val > T_virial_keV)
            
            # --- Process normal gas bin ---
            n_particles = np.sum(bin_mask)
            if n_particles > 0:
                # Extract data for particles in this bin
                bin_pos = coords[bin_mask]
                bin_r = r[bin_mask]
                bin_vel = velocities_all[bin_mask] * np.sqrt(scale_factor) - vhalo
                bin_mass = masses_all[bin_mask] * mass_factor
                bin_potential = potential_all[bin_mask] * scale_factor
                bin_internal_e = internal_e_all[bin_mask]
                bin_X_e = electron_abundance_all[bin_mask]
                bin_metallicity = metallicity_all[bin_mask]
                
                # Calculate temperatures
                bin_T = io.temperature(bin_X_e, bin_internal_e * (u.km / u.s) ** 2).value  # Extract the value
                mu = 4.0 / (1.0 + 3.0 * XH + 4.0 * XH * np.mean(bin_X_e))
                
                # Calculate total mass
                bin_total_mass = np.sum(bin_mass)
                
                # Potential in (km/s)^2
                potential_profile = np.sum(bin_potential)/ n_particles

                # Density in g/cm^3
                density_profile = bin_total_mass * msun_to_g / volume_cm3
                    
                # Temperature profile (mass-weighted average)
                temperature_profile = np.sum(bin_T * bin_mass) / bin_total_mass
                    
                # Pressure profile (P = nkT)
                pressure_profile = temperature_profile * density_profile / (mu * m_p.to(u.g).value)
                    
                # Metallicity profile
                metallicity_profile = (np.sum(bin_metallicity * bin_mass) / bin_total_mass) / Zsun
                    
                # Calculate radial and rotational velocities
                mean_radial_vel, rad_vel_disp = io.calculate_radial_velocity_mean_and_dispersion(
                        bin_pos, bin_vel, bin_mass)
                    
                # Calculate 3D velocity dispersion and convert to 1D
                vel_disp_3D = io.calculate_3D_velocity_dispersion(bin_vel, bin_mass)
                vel_disp_1D = vel_disp_3D / np.sqrt(3.0)
                    
                # Calculate mean rotational velocity
                mean_rot_vel = io.calculate_mean_rotational_velocity(bin_pos, bin_vel, bin_mass)
                    
                # Store profiles in output arrays
                local_potential_arr[halo_index, k] = potential_profile
                local_density_arr[halo_index, k] = density_profile
                local_temperature_arr[halo_index, k] = temperature_profile
                local_pressure_arr[halo_index, k] = pressure_profile
                local_metallicity_arr[halo_index, k] = metallicity_profile
                local_radial_gas_velocity_arr[halo_index, k] = mean_radial_vel
                local_rotational_gas_velocity_arr[halo_index, k] = mean_rot_vel
                local_radial_gas_velocity_dispersion_arr[halo_index, k] = rad_vel_disp
                local_gas_velocity_dispersion_arr[halo_index, k] = vel_disp_1D

            else:
                # No particles in this bin
                local_potential_arr[halo_index, k] = 0
                local_density_arr[halo_index, k] = 0
                local_temperature_arr[halo_index, k] = 0
                local_pressure_arr[halo_index, k] = 0
                local_metallicity_arr[halo_index, k] = 0
                local_radial_gas_velocity_arr[halo_index, k] = 0
                local_rotational_gas_velocity_arr[halo_index, k] = 0
                local_radial_gas_velocity_dispersion_arr[halo_index, k] = 0
                local_gas_velocity_dispersion_arr[halo_index, k] = 0
            
            # --- Process hot bin ---
            n_particles = np.sum(superv_mask)
            if n_particles > 0:
                # Extract data for particles in this bin
                hot_bin_pos = coords[superv_mask]
                hot_bin_r = r[superv_mask]
                hot_bin_vel = velocities_all[superv_mask] * np.sqrt(scale_factor) - vhalo
                hot_bin_mass = masses_all[superv_mask] * mass_factor
                hot_bin_internal_e = internal_e_all[superv_mask]
                hot_bin_X_e = electron_abundance_all[superv_mask]
                hot_bin_metallicity = metallicity_all[superv_mask]
                
                # Calculate temperatures
                hot_bin_T = io.temperature(hot_bin_X_e, hot_bin_internal_e * (u.km / u.s) ** 2).value
                hot_mu = 4.0 / (1.0 + 3.0 * XH + 4.0 * XH * np.mean(hot_bin_X_e))
                
                # Calculate total mass
                hot_bin_total_mass = np.sum(hot_bin_mass)
                
                # Density in g/cm^3
                hot_density_profile = hot_bin_total_mass * msun_to_g / volume_cm3
                    
                # Temperature profile (mass-weighted average)
                hot_temperature_profile = np.sum(hot_bin_T * hot_bin_mass) / hot_bin_total_mass
                    
                # Pressure profile (P = nkT)
                hot_pressure_profile = hot_temperature_profile * hot_density_profile / (hot_mu * m_p.to(u.g).value)
                    
                # Metallicity profile
                hot_metallicity_profile = (np.sum(hot_bin_metallicity * hot_bin_mass) / hot_bin_total_mass) / Zsun
                    
                # Calculate radial and rotational velocities
                hot_mean_radial_vel, hot_rad_vel_disp = io.calculate_radial_velocity_mean_and_dispersion(
                        hot_bin_pos, hot_bin_vel, hot_bin_mass)
                    
                # Calculate 3D velocity dispersion and convert to 1D
                hot_vel_disp_3D = io.calculate_3D_velocity_dispersion(hot_bin_vel, hot_bin_mass)
                hot_vel_disp_1D = hot_vel_disp_3D / np.sqrt(3.0)
                    
                # Calculate mean rotational velocity
                hot_mean_rot_vel = io.calculate_mean_rotational_velocity(hot_bin_pos, hot_bin_vel, hot_bin_mass)
                    
                # Store profiles in output arrays
                local_hot_density_arr[halo_index, k] = hot_density_profile
                local_hot_temperature_arr[halo_index, k] = hot_temperature_profile
                local_hot_pressure_arr[halo_index, k] = hot_pressure_profile
                local_hot_metallicity_arr[halo_index, k] = hot_metallicity_profile
                local_hot_radial_gas_velocity_arr[halo_index, k] = hot_mean_radial_vel
                local_hot_rotational_gas_velocity_arr[halo_index, k] = hot_mean_rot_vel
                local_hot_radial_gas_velocity_dispersion_arr[halo_index, k] = hot_rad_vel_disp
                local_hot_gas_velocity_dispersion_arr[halo_index, k] = hot_vel_disp_1D
            else:
                # No particles in this bin
                local_hot_density_arr[halo_index, k] = 0
                local_hot_temperature_arr[halo_index, k] = 0
                local_hot_pressure_arr[halo_index, k] = 0
                local_hot_metallicity_arr[halo_index, k] = 0
                local_hot_radial_gas_velocity_arr[halo_index, k] = 0
                local_hot_rotational_gas_velocity_arr[halo_index, k] = 0
                local_hot_radial_gas_velocity_dispersion_arr[halo_index, k] = 0
                local_hot_gas_velocity_dispersion_arr[halo_index, k] = 0
    
    return
