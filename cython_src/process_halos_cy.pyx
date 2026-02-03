# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, nonecheck=False

"""
Optimized Cython implementation of the process_halos.
"""

import cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, pow, M_PI
#from cython.parallel import prange
import data_io as io

# Import necessary types from astropy
from astropy import units as u
from astropy.constants import k_B, m_p

# Define types for NumPy arrays
ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t ITYPE_t
ctypedef np.uint8_t BTYPE_t

# Global constants
cdef:
    double XH = 0.76
    double Zsun = 0.0198

#@cython.nogil
#@cython.cfunc
#@cython.noexcept
def process_halos(int start, int end, 
                  double hubble, double scale_factor, double Omega_M, double Omega_L, 
                  double redshift, 
                  object DM_fields, object stellar_fields, object fields, 
                  object pos_h, object vel_h, 
                  object M200c, double boxsize, object radial_bins_inner, 
                  object radial_bins_outer, object T_all, int nbins,
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
    Parameters
    ----------
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
        Halo masses
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
    local_*_array : array
        Output arrays for various profiles
    """
    # Define variables with types
    cdef:
        int ih, halo_index, k, n_particles, idx, p, b, Ndm, Nstar, Ngas
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
        np.ndarray[DTYPE_t, ndim=1] DM_masses_all
        
        np.ndarray[DTYPE_t, ndim=2] stellar_coords_all
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
        double mean_T, hot_mean_T, volume_cm3, kpc_to_cm3

        # Precomputed bin geometry
        np.ndarray[DTYPE_t, ndim=1] r1_arr
        np.ndarray[DTYPE_t, ndim=1] r2_arr
        np.ndarray[DTYPE_t, ndim=1] volume_physical_arr
        np.ndarray[DTYPE_t, ndim=1] volume_cm3_arr
        np.ndarray[DTYPE_t, ndim=1] r_edges_arr

        # Per-bin mass accumulators (reused per halo)
        np.ndarray[DTYPE_t, ndim=1] DM_mass_per_bin
        np.ndarray[DTYPE_t, ndim=1] stellar_mass_per_bin

        # Bin index arrays for digitize
        np.ndarray[np.int32_t, ndim=1] bin_index_DM
        np.ndarray[np.int32_t, ndim=1] bin_index_stellar
        np.ndarray[np.int32_t, ndim=1] bin_index_gas
        
        # Results for normal bins
        double DM_density_profile, stellar_density_profile, potential_profile
        double density_profile, temperature_profile, pressure_profile, metallicity_profile
        double mean_radial_vel, mean_rot_vel, rad_vel_disp, vel_disp_1D
   
        double bin_DM_mass, bin_stellar_mass
     
        # Results for hot bins
        double hot_density_profile, hot_temperature_profile, hot_pressure_profile, hot_metallicity_profile
        double hot_mean_radial_vel, hot_mean_rot_vel, hot_rad_vel_disp, hot_vel_disp_1D
    
    # Convert input data to NumPy arrays with explicit types
    pos_h_arr = np.asarray(pos_h, dtype=np.float64)
    vel_h_arr = np.asarray(vel_h, dtype=np.float64)
    M200c_arr = np.asarray(M200c, dtype=np.float64)
    radial_bins_inner_arr = np.asarray(radial_bins_inner, dtype=np.float64)
    radial_bins_outer_arr = np.asarray(radial_bins_outer, dtype=np.float64)

    # DM fields
    DM_coords_all = np.asarray(DM_fields["Coordinates"], dtype=np.float64)
    DM_masses_all = np.asarray(DM_fields["Masses"], dtype=np.float64)

    # Stellar fields
    stellar_coords_all = np.asarray(stellar_fields["Coordinates"], dtype=np.float64)
    stellar_masses_all = np.asarray(stellar_fields["Masses"], dtype=np.float64)

    # Gas fields
    coords_all = np.asarray(fields["Coordinates"], dtype=np.float64)
    velocities_all = np.asarray(fields["Velocities"], dtype=np.float64)
    masses_all = np.asarray(fields["Masses"], dtype=np.float64)
    potential_all = np.asarray(fields["Potential"], dtype=np.float64)
    internal_e_all = np.asarray(fields["InternalEnergy"], dtype=np.float64)
    electron_abundance_all = np.asarray(fields["ElectronAbundance"], dtype=np.float64)
    metallicity_all = np.asarray(fields["GFM_Metallicity"], dtype=np.float64)

    # Output arrays
    local_DM_density_arr = np.asarray(local_DM_density_array, dtype=np.float64)
    local_stellar_density_arr = np.asarray(local_stellar_density_array, dtype=np.float64)
    local_potential_arr = np.asarray(local_potential_array, dtype=np.float64)
    local_density_arr = np.asarray(local_density_array, dtype=np.float64)
    local_temperature_arr = np.asarray(local_temperature_array, dtype=np.float64)
    local_pressure_arr = np.asarray(local_pressure_array, dtype=np.float64)
    local_metallicity_arr = np.asarray(local_metallicity_array, dtype=np.float64)
    local_radial_gas_velocity_arr = np.asarray(local_radial_gas_velocity_array, dtype=np.float64)
    local_rotational_gas_velocity_arr = np.asarray(local_rotational_gas_velocity_array, dtype=np.float64)
    local_radial_gas_velocity_dispersion_arr = np.asarray(local_radial_gas_velocity_dispersion_array, dtype=np.float64)
    local_gas_velocity_dispersion_arr = np.asarray(local_gas_velocity_dispersion_array, dtype=np.float64)

    local_hot_density_arr = np.asarray(local_hot_density_array, dtype=np.float64)
    local_hot_temperature_arr = np.asarray(local_hot_temperature_array, dtype=np.float64)
    local_hot_pressure_arr = np.asarray(local_hot_pressure_array, dtype=np.float64)
    local_hot_metallicity_arr = np.asarray(local_hot_metallicity_array, dtype=np.float64)
    local_hot_radial_gas_velocity_arr = np.asarray(local_hot_radial_gas_velocity_array, dtype=np.float64)
    local_hot_rotational_gas_velocity_arr = np.asarray(local_hot_rotational_gas_velocity_array, dtype=np.float64)
    local_hot_radial_gas_velocity_dispersion_arr = np.asarray(local_hot_radial_gas_velocity_dispersion_array, dtype=np.float64)
    local_hot_gas_velocity_dispersion_arr = np.asarray(local_hot_gas_velocity_dispersion_array, dtype=np.float64)
    
    # Handle T_all which might be an astropy Quantity
    if hasattr(T_all, 'value'):
        T_all_val = np.asarray(T_all.value, dtype=np.float64)
    else:
        T_all_val = np.asarray(T_all, dtype=np.float64)

    # Precompute bin radii, volumes, and edges (physical units)
    r1_arr = np.empty(num_bins, dtype=np.float64)
    r2_arr = np.empty(num_bins, dtype=np.float64)
    volume_physical_arr = np.empty(num_bins, dtype=np.float64)
    volume_cm3_arr = np.empty(num_bins, dtype=np.float64)
    r_edges_arr = np.empty(num_bins + 1, dtype=np.float64)

    kpc_to_cm3 = kpc_to_cm * kpc_to_cm * kpc_to_cm

    for k in range(num_bins):
        r1_arr[k] = radial_bins_inner_arr[k] / scale_factor * hubble
        r2_arr[k] = radial_bins_outer_arr[k] / scale_factor * hubble
        volume_physical = volume_factor * (r2_arr[k]**3 - r1_arr[k]**3) * (scale_factor**3 / hubble**3)
        volume_physical_arr[k] = volume_physical
        volume_cm3_arr[k] = volume_physical * kpc_to_cm3
        r_edges_arr[k] = r1_arr[k]

    # Last edge is the outer radius of the last bin
    r_edges_arr[num_bins] = r2_arr[num_bins - 1]

    # Allocate per-bin mass accumulators (reused per halo)
    DM_mass_per_bin = np.empty(num_bins, dtype=np.float64)
    stellar_mass_per_bin = np.empty(num_bins, dtype=np.float64)

    # Set virial temperature once (currently fixed at 1e6 K)
    T_virial_keV = 1.0e6 * K_to_keV

    # Process each halo
    for ih in range(start, end):
        halo_index = ih - start

        # -------------------------
        # DM component (digitize)
        # -------------------------
        # Calculate particle positions relative to halo center (DM)
        DM_coords = io.periodic_bcs(DM_coords_all, pos_h_arr[ih], box=boxsize) - pos_h_arr[ih]
        # Calculate radial distance for each DM particle
        DM_r = np.sqrt(np.sum(DM_coords * DM_coords, axis=1))
        Ndm = DM_r.shape[0]

        # Map each DM particle to a radial bin
        bin_index_DM = np.digitize(DM_r, r_edges_arr).astype(np.int32)

        # Reset per-bin DM masses
        for k in range(num_bins):
            DM_mass_per_bin[k] = 0.0

        # Accumulate DM mass per bin
        for p in range(Ndm):
            b = bin_index_DM[p] - 1
            if b < 0 or b >= num_bins:
                continue
            DM_mass_per_bin[b] += DM_masses_all[p] * mass_factor

        # Convert to density per bin
        for k in range(num_bins):
            bin_DM_mass = DM_mass_per_bin[k]
            if bin_DM_mass > 0.0:
                # Density in mass / (kpc^3), using physical volume
                DM_density_profile = bin_DM_mass / volume_physical_arr[k]
                local_DM_density_arr[halo_index, k] = DM_density_profile
            else:
                local_DM_density_arr[halo_index, k] = 0.0

        # -------------------------
        # Stellar component (digitize)
        # -------------------------
        # Calculate particle positions relative to halo center (stars)
        stellar_coords = io.periodic_bcs(stellar_coords_all, pos_h_arr[ih], box=boxsize) - pos_h_arr[ih]
        # Calculate radial distance for each stellar particle
        stellar_r = np.sqrt(np.sum(stellar_coords * stellar_coords, axis=1))
        Nstar = stellar_r.shape[0]

        # Map each star to a radial bin
        bin_index_stellar = np.digitize(stellar_r, r_edges_arr).astype(np.int32)

        # Reset per-bin stellar masses
        for k in range(num_bins):
            stellar_mass_per_bin[k] = 0.0

        # Accumulate stellar mass per bin
        for p in range(Nstar):
            b = bin_index_stellar[p] - 1
            if b < 0 or b >= num_bins:
                continue
            stellar_mass_per_bin[b] += stellar_masses_all[p] * mass_factor

        # Convert to density per bin
        for k in range(num_bins):
            bin_stellar_mass = stellar_mass_per_bin[k]
            if bin_stellar_mass > 0.0:
                stellar_density_profile = bin_stellar_mass / volume_physical_arr[k]
                local_stellar_density_arr[halo_index, k] = stellar_density_profile
            else:
                local_stellar_density_arr[halo_index, k] = 0.0

        # -------------------------
        # Gas component
        # -------------------------
        # Calculate gas  particle positions relative to halo center
        coords = io.periodic_bcs(coords_all, pos_h_arr[ih], box=boxsize) - pos_h_arr[ih]
        
        # Calculate radial distance for each particle
        r = np.sqrt(np.sum(coords * coords, axis=1))
        
        # Map gas particles to radial bins
        Ngas = r.shape[0]
        bin_index_gas = np.digitize(r, r_edges_arr).astype(np.int32) - 1

        # Compute halo peculiar velocity - extract the row from the 2D array
        vhalo = vel_h_arr[ih,:] / scale_factor
        
        # Process each bin
        for k in range(num_bins):
            # Use precomputed radius bounds and volumes
            r1 = r1_arr[k]
            r2 = r2_arr[k]
            volume_physical = volume_physical_arr[k]
            volume_cm3 = volume_cm3_arr[k]
            
            # Create masks for normal and hot particles using bin indices
            bin_mask = (bin_index_gas == k)
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
                
                # Temperature profile (mass-weighted)
                mean_T = np.sum(bin_T * bin_mass * msun_to_g) / (bin_total_mass * msun_to_g)
                temperature_profile = mean_T
                
                # Pressure profile: P = rho * k_B * T / (mu m_p)
                pressure_profile = (density_profile * k_B.value * temperature_profile) / (mu * m_p.value)
                
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
   
                # Save normal gas results
                local_potential_arr[halo_index, k] = potential_profile
                local_density_arr[halo_index, k] = density_profile
                local_temperature_arr[halo_index, k] = temperature_profile
                local_pressure_arr[halo_index, k] = pressure_profile
                local_metallicity_arr[halo_index, k] = metallicity_profile
                local_radial_gas_velocity_arr[halo_index, k] = mean_radial_vel
                local_radial_gas_velocity_dispersion_arr[halo_index, k] = rad_vel_disp
                local_gas_velocity_dispersion_arr[halo_index, k] = vel_disp_1D
                local_rotational_gas_velocity_arr[halo_index, k] = mean_rot_vel
            else:
                # No particles in this bin
                local_potential_arr[halo_index, k] = 0
                local_density_arr[halo_index, k] = 0
                local_temperature_arr[halo_index, k] = 0
                local_pressure_arr[halo_index, k] = 0
                local_metallicity_arr[halo_index, k] = 0
                local_radial_gas_velocity_arr[halo_index, k] = 0
                local_radial_gas_velocity_dispersion_arr[halo_index, k] = 0
                local_gas_velocity_dispersion_arr[halo_index, k] = 0
                local_rotational_gas_velocity_arr[halo_index, k] = 0

            # --- Process hot gas bin ---
            n_particles = np.sum(superv_mask)
            if n_particles > 0:
                # Extract data for particles in this bin
                hot_bin_pos = coords[superv_mask]
                hot_bin_r = r[superv_mask]
                hot_bin_vel = velocities_all[superv_mask] * np.sqrt(scale_factor) - vhalo
                hot_bin_mass = masses_all[superv_mask] * mass_factor
                hot_bin_potential = potential_all[superv_mask] * scale_factor
                hot_bin_internal_e = internal_e_all[superv_mask]
                hot_bin_X_e = electron_abundance_all[superv_mask]
                hot_bin_metallicity = metallicity_all[superv_mask]
                
                # Calculate temperatures
                hot_bin_T = io.temperature(hot_bin_X_e, hot_bin_internal_e * (u.km / u.s) ** 2).value  # Extract the value
                hot_mu = 4.0 / (1.0 + 3.0 * XH + 4.0 * XH * np.mean(hot_bin_X_e))
                
                # Calculate total mass
                hot_bin_total_mass = np.sum(hot_bin_mass)
                
                # Potential in (km/s)^2
                potential_profile = np.sum(hot_bin_potential)/ n_particles

                # Density in g/cm^3
                hot_density_profile = hot_bin_total_mass * msun_to_g / volume_cm3
                
                # Temperature profile (mass-weighted)
                hot_mean_T = np.sum(hot_bin_T * hot_bin_mass * msun_to_g) / (hot_bin_total_mass * msun_to_g)
                hot_temperature_profile = hot_mean_T
                
                # Pressure profile: P = rho * k_B * T / (mu m_p)
                hot_pressure_profile = (hot_density_profile * k_B.value * hot_temperature_profile) / (hot_mu * m_p.value)
                
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
   
                # Save hot gas results
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

