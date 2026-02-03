import numpy as np
import h5py
import glob
import os
import re
from astropy import units as u
from astropy.constants import k_B, m_p,m_e, sigma_T, c
from scipy.spatial.transform import Rotation
import illustris_python as il


kpc = 1.0*u.kpc.to(u.cm)
Zsun = 0.0198
XH = 0.76

def return_header(base, snap):
    """Load snapshot header attrs (scale factor a, h, unit info)."""
    # Use chunk 0; header is identical across chunks
    f = h5py.File(il.snapshot.snapPath(base, snap, 0), "r")
    h = dict(f["Header"].attrs.items())
    f.close()
    return h


def periodic_bcs(posp, posh, box=50000.0):

    xp = posp[:,0]
    yp = posp[:,1]
    zp = posp[:,2]

    xh = posh[0]
    yh = posh[1]
    zh = posh[2]

    xdel = xp - xh
    ydel = yp - yh
    zdel = zp - zh

    xp[xdel >= box/2.] = xp[xdel >= box/2.] - box
    xp[xdel < -1.*box/2.] = xp[xdel < -1. *box/2.] + box
    yp[ydel >= box/2.] = yp[ydel >= box/2.] - box
    yp[ydel < -1.*box/2.] = yp[ydel < -1. *box/2.] + box
    zp[zdel >= box/2.] = zp[zdel >= box/2.] - box
    zp[zdel < -1.*box/2.] = zp[zdel < -1. *box/2.] + box

    posp=np.column_stack([xp,yp,zp])

    return posp

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )

def load_data(path, suite, sim_set, run, snapNum, data_type, field_list):
   
    basePath =  f'{path}/{suite}/{sim_set}/{run}/'
    header = return_header(basePath, snapNum)
    data = il.snapshot.loadSubset(basePath,snapNum,data_type,fields=field_list)

    if data_type == 'dm':
        mass_table = header[u'MassTable'] #1e10 Msun/h

        if suite == 'IllustrisTNG':
            part_mass = mass_table[1]
        else :
            part_mass = 1.0

        data['Masses']  = np.full (data['Coordinates'].shape[0], part_mass)

    return header, data


def extract_halo_properties(path, suite, sim_set, run, snapNum):
    
    basePath =  f'{path}/{suite}/{sim_set}/{run}/'

    halo_field_list = ['GroupPos','GroupVel',
              'Group_M_Crit200','Group_R_Crit200',
              'Group_M_Mean200','Group_R_Mean200',
              'Group_M_Crit500','Group_R_Crit500',
              'GroupSFR', 
              'GroupMass', 'GroupLen',
              'GroupMassType', 'GroupLenType',
              'GroupFirstSub'
              ]

    subhalo_field_list = ['SubhaloPos', 'SubhaloVel', 'SubhaloMassInRadType', 'SubhaloSFRinRad', 'SubhaloGrNr']

    #try:
        #halo_path = f'{path}/{suite}/{sim_set}/{run}/groups_{snap_str}'
        #halo_path = '%s/groups_%s.hdf5'%(halo_path,  str(snapNum).zfill(3))
        #f = h5py.File(halo_path, 'r')
    halos = il.groupcat.loadHalos(basePath,snapNum,fields=halo_field_list)
    subhalos = il.groupcat.loadSubhalos(basePath,snapNum,fields=subhalo_field_list)

    #except:
    #    print('FOF file not found!')
    #    print(f'basePath = {basePath}')
    #    exit(1)

    # Halo Quantities
    pos_h     = halos['GroupPos'] #kpc/h
    vel_h     = halos['GroupVel']       #km/s
    SFR_h     = halos['GroupSFR']       #Msun/yr
    mass_h    = halos['GroupMass']*1e10 #Msun/h
    len_h     = halos['GroupLen']      #the total number ohalos particles in the halo (gas+dm+stars+black_holes)
    lentype_h = halos['GroupLenType']   #the number ohalos particles in a halo by particle type
    M200c     = halos['Group_M_Crit200']*1e10 #Msun/h
    R200c     = halos['Group_R_Crit200'] #kpc/h
    M500c     = halos['Group_M_Crit500']*1e10   
    R500c     = halos['Group_R_Crit500']

    Sub_Pos = subhalos['SubhaloPos']
    Sub_Vel = subhalos['SubhaloVel']

    Sub_MStar = subhalos['SubhaloMassInRadType'][:,4]
    Sub_MGas = subhalos['SubhaloMassInRadType'][:,0]
    Sub_MBH = subhalos['SubhaloMassInRadType'][:,5]
    Sub_SFR = subhalos['SubhaloSFRinRad']
    Sub_GrpID = subhalos['SubhaloGrNr']

    Grp_FirstSub = halos['GroupFirstSub']
    non_empty = np.where(Grp_FirstSub>=0)[0]
    Grp_FirstSub = Grp_FirstSub[non_empty]
    Pos_FirstSub = Sub_Pos[Grp_FirstSub,:]
    Vel_FirstSub = Sub_Vel[Grp_FirstSub,:]

    Mstar = Sub_MStar[Grp_FirstSub]
    Mgas = Sub_MGas[Grp_FirstSub]
    SFR = Sub_SFR[Grp_FirstSub]
    Mbh = Sub_MBH[Grp_FirstSub]

    halo_n = len(len_h) # number of halos

    if suite == 'IllustrisTNG' :
        IDs = None
        start_stops = [ [np.sum(len_h[:idx]), np.sum(len_h[:idx+1])] for idx in np.arange(halo_n)]

    else :
        IDs = f['IDs']['ID'][:]
   
    #return M200c, R200c, M500c, R500c, Mstar, pos_h, vel_h, len_h, lentype_h, start_stops, IDs
    return M200c, R200c, M500c, R500c, Mstar, pos_h, vel_h, len_h, lentype_h, IDs

def temperature(Xe, internal_e):
    """
    https://www.tng-project.org/data/docs/faq/#gen6
    """
    XH = 0.76
    mu = 4./(1.+3.*XH+4.*XH*Xe) * m_p.to(u.g)
    Temp = 2./3. * internal_e * mu
    #return (Temp/k_B).to(u.K)
    return (Temp).to(u.keV)

def weighted_avg_and_std(values, weights, axis=0):
    """
    Return the weighted average and standard deviation.
    values, weights -- NumPy ndarrays with the same shape.
    """
    
    norm_weights = weights/np.sum(weights)
    average = np.average(values, weights=norm_weights, axis=axis)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=norm_weights, axis=axis)
    return (average, np.sqrt(variance))


def calculate_3D_velocity_dispersion(velocities, masses):

    # Convert inputs to numpy arrays if they aren't already
    velocities = np.asarray(velocities)
    masses = np.asarray(masses)
 
    mass_total = np.sum(masses)
    #v_mean = np.sum(velocities * masses[:, np.newaxis], axis=0) / mass_total
    v_dev_squared = (velocities)**2  # shape (N, 3)
    var_components = np.sum(v_dev_squared * masses[:, np.newaxis], axis=0) / mass_total

    sigma_3D = np.sqrt(np.sum(var_components))  # scalar

    return sigma_3D

def calculate_radial_velocity_mean_and_dispersion(positions, velocities, masses):

    # Convert inputs to numpy arrays if they aren't already
    positions = np.asarray(positions)
    velocities = np.asarray(velocities)
    masses = np.asarray(masses)
 
    # Calculate position vectors relative to center
    r_vectors = positions #- center
    
    # Calculate the squared magnitudes of position vectors
    r_magnitudes = np.linalg.norm(r_vectors, axis=1)
    # unit vector
    r_hat = r_vectors / r_magnitudes[:, np.newaxis]

    radial_velocities = np.einsum('ij,ij->i', velocities, r_hat)
    #radial_velocities = np.sum(velocities * r_hat, axis=1)

    average = np.average(radial_velocities, weights=masses)

    # Fast and numerically precise:
    variance = np.average((radial_velocities-average)**2, weights=masses)

    return (average, np.sqrt(variance))

  
def calculate_mean_rotational_velocity(positions, velocities, masses):
    """
    Calculate the mean rotational velocity for a system of particles.
    
    Parameters:
    -----------
    positions : numpy.ndarray
        Array of shape (n, 3) containing the [x, y, z] positions of n particles
    velocities : numpy.ndarray
        Array of shape (n, 3) containing the [vx, vy, vz] velocities of n particles
    masses : numpy.ndarray
        Array of shape (n,) containing the masses of n particles
    center : numpy.ndarray, optional
        Reference point [x, y, z] for rotation calculation. If None, the center of mass is used.
    
    Returns:
    --------
    dict
        Contains the mean angular momentum vector, mean angular velocity vector,
        and the scalar magnitude of the mean angular velocity
    """
    # Convert inputs to numpy arrays if they aren't already
    positions = np.asarray(positions)
    velocities = np.asarray(velocities)
    masses = np.asarray(masses)
    
    # Calculate center of mass if center is not provided
    #if center is None:
    #    center = calculate_center_of_mass(positions, masses)
    #    print(f"Calculated center of mass: {center}")
    
    # Calculate position vectors relative to center
    r_vectors = positions #- center
    
    # Calculate the squared magnitudes of position vectors
    r_magnitudes = np.linalg.norm(r_vectors)

    # Cross product of r × v for angular momentum per unit mass
    cross_product = np.cross(r_vectors, velocities)        

    # Weighted angular momentum
    angular_momenta = cross_product * masses[:, np.newaxis] 
   
    # Calculate total mass
    #total_mass = np.sum(masses)
    
    # Calculate total angular momentum
    total_angular_momentum = np.sum(angular_momenta, axis=0)
    total_angular_momentum_magnitude = np.linalg.norm(total_angular_momentum) 
    mass_moments = np.sum(r_magnitudes * masses)

    # Calculate magnitude of mean angular velocity
    mean_rotational_velocity_magnitude = total_angular_momentum_magnitude/mass_moments
    
    return mean_rotational_velocity_magnitude


def calculate_center_of_mass(positions, masses):
    """
    Calculate the center of mass for a system of particles.
    
    Parameters:
    -----------
    positions : numpy.ndarray
        Array of shape (n, 3) containing the [x, y, z] positions of n particles
    masses : numpy.ndarray
        Array of shape (n,) containing the masses of n particles
    
    Returns:
    --------
    numpy.ndarray
        The [x, y, z] coordinates of the center of mass
    """
    total_mass = np.sum(masses)
    weighted_positions = positions * masses[:, np.newaxis]
    center_of_mass = np.sum(weighted_positions, axis=0) / total_mass
    return center_of_mass


def virial_temperature(M_halo, z, mdef='200c', mu=0.6, Omega_M=0.3, Omega_L=0.7, hubble=0.7):
   # Constants
    kB = 1.380649e-16
    G = 6.67430e-8
    mp = 1.6726219e-24
    Msun = 1.989e33
    Mpc = 3.0856e24

    Omega_L = 1.0-Omega_M
    Ez = np.sqrt(Omega_M*(1+z)**3 + Omega_L)
    H0 = 100.0*hubble #km/s/Mpc

    # Calculate the critical density at redshift z (in Msun/Mpc^3)
    rho_crit_0 = 3.0 * (H0*1e5/Mpc)**2 / (8 * np.pi * G) / Msun * Mpc**3

    # Calculate overdensity factor for virialization
    # In a flat Omega_m + Omega_Lambda = 1 universe
    Om_z = Omega_M*(1+z)**3/Ez**2

    if mdef == 'vir':
        x = Om_z - 1
        delta = 18 * np.pi**2 + 82.0 * x - 39.0 * x**2
        rho = delta*rho_crit_0
    elif mdef[-1] == 'c':
        delta = int(mdef[:-1])
        rho = delta*rho_crit_0*Ez**2
    elif mdef[-1] == 'm':
        delta = int(mdef[:-1])
        rho = delta*rho_crit_0*Om_z

    if M_halo > 0 :
        # Calculate virial radius (in Mpc)
        r_halo = (3 * M_halo / (4 * np.pi * rho))**(1/3)

        # Calculate virial velocity (in cm/s)
        v_halo = np.sqrt(G * M_halo * Msun / (r_halo*Mpc ))

        # Calculate virial temperature (in K)
        T_halo = mu * mp * v_halo**2 / (2 * kB)

    else :
        T_halo = 0.0

    return T_halo
    

