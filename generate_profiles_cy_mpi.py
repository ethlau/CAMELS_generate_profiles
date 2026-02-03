import numpy as np
import sys
from astropy import units as u
from astropy.constants import k_B, m_p, m_e, sigma_T, c
import data_io as io

from pathlib import Path
from tqdm import tqdm

from mpi4py import MPI
# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Rank of this process
size = comm.Get_size()  # Total number of processes

mass_lower_lim = 3e12

load_all_data = True

# Try to import the optimized Cython function
try:
    from cython_src.process_halos_cy import process_halos
    import cython
    print(f"Process {rank} using optimized Cython function")
    USE_CYTHON = True
except ImportError:
    print(f"Process {rank} using Python function (Cython module not found)")
    USE_CYTHON = False
    exit()

kpc = 1.0*u.kpc.to(u.cm)
Zsun = 0.0198
XH = 0.76

rmin = 10.0
rmax = 5000.0
nbins = 30
dx = (np.log10(rmax/rmin))/nbins

log_radial_bins_inner = np.zeros(nbins)
log_radial_bins = np.zeros(nbins)
log_radial_bins_outer = np.zeros(nbins)

for i in np.arange(nbins):
    log_radial_bins_inner[i] = np.log10(rmin) + float(i)*dx
    log_radial_bins[i] = np.log10(rmin) + (float(i)+0.5)*dx
    log_radial_bins_outer[i] = np.log10(rmin) + float(i+1)*dx

radial_bins_inner = 10**log_radial_bins_inner
radial_bins = 10**log_radial_bins
radial_bins_outer = 10**log_radial_bins_outer

#path = '/home/etl28/CAMELS_Profiles/Sims'
path = '/home/etl28/palmer_scratch/CAMELS_L50'
suite = 'IllustrisTNG'
#suite = 'SIMBA'
sim_set = '1P'
run = sys.argv[1]
snapNum = 90

output_file_path = f'./Profiles_cy/{suite}_{run}_snap{str(snapNum).zfill(3)}.npz'

if Path(output_file_path).exists():
    print(f"File {output_file_path} exists. Skipping profile generation")
    exit()


# Extract halo data on root process
if rank == 0:
    data = io.extract_halo_properties(path, suite, sim_set, run, snapNum)
    M200c, R200c, M500c, R500c, Mstar, pos_h, vel_h, len_h, halo_ids = data

    mask = M200c >= mass_lower_lim

    M200c = M200c[mask]
    R200c = R200c[mask]
    M500c = M500c[mask]
    R500c = R500c[mask]
    Mstar = Mstar[mask]
    pos_h = pos_h[mask,:]
    vel_h = vel_h[mask,:]
    len_h = len_h[mask]
    #lentype_h = lentype_h[mask]

    nhalos = len(len_h)  # number of halos
    #nhalos = 10  # number of halos
    print(f"Number of halos is {nhalos}")
else:
    nhalos = None
    #M200c = R200c = M500c = R500c = Mstar = pos_h = vel_h = len_h = lentype_h = start_stops = IDs = None
    M200c = R200c = M500c = R500c = Mstar = pos_h = vel_h = len_h = None
    halo_ids = None

# Broadcast data to all processes
nhalos = comm.bcast(nhalos, root=0)
M200c = comm.bcast(M200c, root=0)
R200c = comm.bcast(R200c, root=0)
M500c = comm.bcast(M500c, root=0)
R500c = comm.bcast(R500c, root=0)
Mstar = comm.bcast(Mstar, root=0)
pos_h = comm.bcast(pos_h, root=0)
vel_h = comm.bcast(vel_h, root=0)
len_h = comm.bcast(len_h, root=0)
#lentype_h = comm.bcast(lentype_h, root=0)
halo_ids = comm.bcast(halo_ids, root=0)

input_fields = {}
input_fields['gas'] = ['InternalEnergy', 'ElectronAbundance', 'Coordinates', 'Masses',
                    'GFM_Metallicity', 'Velocities', 'Potential']

input_fields['dm'] = ['Coordinates', 'Velocities']

input_fields['stars'] = ['Coordinates', 'Masses', 'Velocities']

if suite != 'IllustrisTNG':
    # For non-TNG suites the DM halo selection in `load_data` uses ParticleIDs
    # internally, but we may still want to store them as a field as well.
    if 'ParticleIDs' not in input_fields['dm']:
        input_fields['dm'].append('ParticleIDs')

# ---------------------------------------------------------------------
#  - Rank 0 loads particles halo-by-halo using load_data(..., halo_index)
#  - Then concatenates them into global arrays that contain only
#    particles belonging to the selected halos.
# ---------------------------------------------------------------------
if True:
    basePath = f"{path}/{suite}/{sim_set}/{run}/"
    header   = io.return_header(basePath, snapNum)

    boxsize      = header[u'BoxSize']      # comoving kpc/h
    redshift     = header['Redshift']
    hubble       = header['HubbleParam']
    Omega_M      = header['Omega0']
    Omega_L      = header['OmegaLambda']
    scale_factor = 1.0 / (1.0 + redshift)

    DM_fields = {}
    fields = {}
    stellar_fields = {}


    if load_all_data :
        _, DM_fields = io.load_data(
                path, suite, sim_set, run, snapNum,
                'dm', input_fields['dm'],
                halo_index=None
                )
        # gas
        _, fields = io.load_data(
                path, suite, sim_set, run, snapNum,
                'gas', input_fields['gas'],
                halo_index=None
            )
        # stars
        _, stellar_fields = io.load_data(
                path, suite, sim_set, run, snapNum,
                'stars', input_fields['stars'],
                halo_index=None
            ) 

       
    else :
        # Accumulators for per-halo chunks
        DM_chunks      = {fld: [] for fld in input_fields['dm']}
        gas_chunks     = {fld: [] for fld in input_fields['gas']}
        stellar_chunks = {fld: [] for fld in input_fields['stars']}

        # Loop over selected halos and load only their particles
        for hid in halo_ids:
            # DM
            _, DM_halo = io.load_data(
                path, suite, sim_set, run, snapNum,
                'dm', input_fields['dm'],
                halo_index=int(hid)
                )
            # gas
            _, gas_halo = io.load_data(
                path, suite, sim_set, run, snapNum,
                'gas', input_fields['gas'],
                halo_index=int(hid)
            )
            # stars
            _, star_halo = io.load_data(
                path, suite, sim_set, run, snapNum,
                'stars', input_fields['stars'],
                halo_index=int(hid)
            ) 

            for fld in input_fields['dm']:
                DM_chunks[fld].append(DM_halo[fld])
            for fld in input_fields['gas']:
                gas_chunks[fld].append(gas_halo[fld])
            for fld in input_fields['stars']:
                stellar_chunks[fld].append(star_halo[fld])

        # Concatenate into global arrays (only halo particles)
        for fld, lst in DM_chunks.items():
            DM_fields[fld] = np.concatenate(lst, axis=0) if len(lst) > 0 else np.empty((0,), dtype=np.float64)

        for fld, lst in gas_chunks.items():
            fields[fld] = np.concatenate(lst, axis=0) if len(lst) > 0 else np.empty((0,), dtype=np.float64)

        for fld, lst in stellar_chunks.items():
            stellar_fields[fld] = np.concatenate(lst, axis=0) if len(lst) > 0 else np.empty((0,), dtype=np.float64)

    # Ensure float64 where needed (Cython likes float64)
    for d in (fields, DM_fields, stellar_fields):
        for key, arr in d.items():
            if isinstance(arr, np.ndarray) and arr.dtype != np.float64:
                d[key] = arr.astype(np.float64)

else:
    boxsize = redshift = hubble = Omega_M = Omega_L = scale_factor = None
    DM_fields = fields = stellar_fields = None
    #mass_table = None

# Broadcast header-derived cosmology and particle dictionaries
boxsize = comm.bcast(boxsize, root=0)
redshift = comm.bcast(redshift, root=0)
hubble = comm.bcast(hubble, root=0)
Omega_M = comm.bcast(Omega_M, root=0)
Omega_L = comm.bcast(Omega_L, root=0)
scale_factor = comm.bcast(scale_factor, root=0)
#mass_table = comm.bcast(mass_table, root=0)

#fields = comm.bcast(fields, root=0)
#DM_fields = comm.bcast(DM_fields, root=0)
#stellar_fields = comm.bcast(stellar_fields, root=0)


# Relevant fields for all particles and halos
internal_e_all = fields['InternalEnergy'] * (u.km / u.s)**2
X_e_all = fields['ElectronAbundance']
T_all = io.temperature(X_e_all, internal_e_all)  # Shape: (n_particles,)

# Mean molecular weight (mu)
#mu_all = 4. / (1. + 3. * XH + 4. * XH * np.mean(X_e_all))

# Determine the number of halos each process will handle
halos_per_proc = nhalos // size
start_halo = rank * halos_per_proc
if rank == size - 1:
    end_halo = nhalos  # Last process takes the remainder
else:
    end_halo = start_halo + halos_per_proc

chunk_sizes = np.zeros(size, dtype='i')
for i in range(size):
    chunk_sizes[i] = halos_per_proc
    if i == size-1:
        chunk_sizes[i] = nhalos - (size-1)*halos_per_proc

print(f"Process {rank} is handling halos {start_halo} to {end_halo - 1} with {chunk_sizes[rank]} halos")

# Allocate profile array for local halos
local_DM_density_array = np.zeros([end_halo - start_halo, nbins])
local_stellar_density_array = np.zeros([end_halo - start_halo, nbins])
local_potential_array = np.zeros([end_halo - start_halo, nbins])
local_density_array = np.zeros([end_halo - start_halo, nbins])
local_temperature_array = np.zeros([end_halo - start_halo, nbins])
local_pressure_array = np.zeros([end_halo - start_halo, nbins])
local_metallicity_array = np.zeros([end_halo - start_halo, nbins])
local_radial_gas_velocity_array = np.zeros([end_halo - start_halo, nbins])
local_gas_velocity_dispersion_array = np.zeros([end_halo - start_halo, nbins])
local_radial_gas_velocity_dispersion_array = np.zeros([end_halo - start_halo, nbins])
local_rotational_gas_velocity_array = np.zeros([end_halo - start_halo, nbins])

local_hot_density_array = np.zeros([end_halo - start_halo, nbins])
local_hot_temperature_array = np.zeros([end_halo - start_halo, nbins])
local_hot_pressure_array = np.zeros([end_halo - start_halo, nbins])
local_hot_metallicity_array = np.zeros([end_halo - start_halo, nbins])
local_hot_radial_gas_velocity_array = np.zeros([end_halo - start_halo, nbins])
local_hot_gas_velocity_dispersion_array = np.zeros([end_halo - start_halo, nbins])
local_hot_radial_gas_velocity_dispersion_array = np.zeros([end_halo - start_halo, nbins])
local_hot_rotational_gas_velocity_array = np.zeros([end_halo - start_halo, nbins])

# Process local halos
output_file = Path(output_file_path)


if not output_file.is_file():
    # Choose between Python or Cython implementation
    if USE_CYTHON:
        #with cython.nogil:
        process_halos(start_halo, end_halo, 
                 hubble, scale_factor, Omega_M, Omega_L, 
                 redshift, 
                 DM_fields, stellar_fields, fields, 
                 pos_h, vel_h, 
                 M200c, boxsize, 
                 radial_bins_inner, 
                 radial_bins_outer, 
                 T_all, nbins,
                 local_DM_density_array, 
                 local_stellar_density_array,
                 local_potential_array, 
                 local_density_array, 
                 local_temperature_array, 
                 local_pressure_array, 
                 local_metallicity_array,
                 local_radial_gas_velocity_array, 
                 local_rotational_gas_velocity_array,
                 local_radial_gas_velocity_dispersion_array, 
                 local_gas_velocity_dispersion_array,
                 local_hot_density_array, local_hot_temperature_array,
                 local_hot_pressure_array, local_hot_metallicity_array,
                 local_hot_radial_gas_velocity_array, 
                 local_hot_rotational_gas_velocity_array,
                 local_hot_radial_gas_velocity_dispersion_array, 
                 local_hot_gas_velocity_dispersion_array)
    else:
        print("Cython module not set up!")
        exit()
else:
    print(f"File already exists. Skipping file {output_file_path}")
    exit()

# Gather profile arrays from all processes to the root process
if rank == 0:
    DM_density_array = np.zeros(nhalos*nbins)
    stellar_density_array = np.zeros(nhalos*nbins)
    potential_array = np.zeros(nhalos*nbins)
    density_array = np.zeros(nhalos*nbins)
    temperature_array = np.zeros(nhalos*nbins)
    pressure_array = np.zeros(nhalos*nbins)
    metallicity_array = np.zeros(nhalos*nbins)
    radial_gas_velocity_array = np.zeros(nhalos*nbins)
    rotational_gas_velocity_array = np.zeros(nhalos*nbins)
    gas_velocity_dispersion_array = np.zeros(nhalos*nbins)
    radial_gas_velocity_dispersion_array = np.zeros(nhalos*nbins)

    hot_density_array = np.zeros(nhalos*nbins)
    hot_temperature_array = np.zeros(nhalos*nbins)
    hot_pressure_array = np.zeros(nhalos*nbins)
    hot_metallicity_array = np.zeros(nhalos*nbins)
    hot_radial_gas_velocity_array = np.zeros(nhalos*nbins)
    hot_rotational_gas_velocity_array = np.zeros(nhalos*nbins)
    hot_gas_velocity_dispersion_array = np.zeros(nhalos*nbins)
    hot_radial_gas_velocity_dispersion_array = np.zeros(nhalos*nbins)
else:
    DM_density_array = stellar_density_array = potential_array = None
    density_array = temperature_array = pressure_array = metallicity_array = None
    radial_gas_velocity_array = rotational_gas_velocity_array = gas_velocity_dispersion_array = None
    radial_gas_velocity_dispersion_array = None
    hot_density_array = hot_temperature_array = hot_pressure_array = None
    hot_metallicity_array = hot_radial_gas_velocity_array = None
    hot_rotational_gas_velocity_array = hot_gas_velocity_dispersion_array = None
    hot_radial_gas_velocity_dispersion_array = None

# Prepare sendcounts and displacements for `Gatherv`
sendcounts = chunk_sizes * nbins  # Total number of elements (rows * cols) for each process
disp = np.insert(np.cumsum(sendcounts[:-1]), 0, 0)  # Starting index for each process

# Use MPI.Gatherv to gather data of varying sizes from each process
comm.Gatherv(sendbuf=local_potential_array.flatten(), 
             recvbuf=(potential_array, sendcounts), 
             root=0)

comm.Gatherv(sendbuf=local_DM_density_array.flatten(), 
             recvbuf=(DM_density_array, sendcounts), 
             root=0)

comm.Gatherv(sendbuf=local_stellar_density_array.flatten(), 
             recvbuf=(stellar_density_array, sendcounts), 
             root=0)

comm.Gatherv(sendbuf=local_density_array.flatten(), 
             recvbuf=(density_array, sendcounts), 
             root=0)
comm.Gatherv(sendbuf=local_temperature_array.flatten(), 
             recvbuf=(temperature_array, sendcounts), 
             root=0)
comm.Gatherv(sendbuf=local_pressure_array.flatten(), 
             recvbuf=(pressure_array, sendcounts),
             root=0)
comm.Gatherv(sendbuf=local_metallicity_array.flatten(), 
             recvbuf=(metallicity_array, sendcounts), 
             root=0)
comm.Gatherv(sendbuf=local_radial_gas_velocity_array.flatten(), 
             recvbuf=(radial_gas_velocity_array, sendcounts), 
             root=0)
comm.Gatherv(sendbuf=local_radial_gas_velocity_dispersion_array.flatten(), 
             recvbuf=(radial_gas_velocity_dispersion_array, sendcounts), 
             root=0)
comm.Gatherv(sendbuf=local_rotational_gas_velocity_array.flatten(), 
             recvbuf=(rotational_gas_velocity_array, sendcounts), 
             root=0)
comm.Gatherv(sendbuf=local_gas_velocity_dispersion_array.flatten(), 
             recvbuf=(gas_velocity_dispersion_array, sendcounts), 
             root=0)

comm.Gatherv(sendbuf=local_hot_density_array.flatten(), 
             recvbuf=(hot_density_array, sendcounts), 
             root=0)
comm.Gatherv(sendbuf=local_hot_temperature_array.flatten(), 
             recvbuf=(hot_temperature_array, sendcounts), 
             root=0)
comm.Gatherv(sendbuf=local_hot_pressure_array.flatten(), 
             recvbuf=(hot_pressure_array, sendcounts),
             root=0)
comm.Gatherv(sendbuf=local_hot_metallicity_array.flatten(), 
             recvbuf=(hot_metallicity_array, sendcounts), 
             root=0)
comm.Gatherv(sendbuf=local_hot_radial_gas_velocity_array.flatten(), 
             recvbuf=(hot_radial_gas_velocity_array, sendcounts), 
             root=0)
comm.Gatherv(sendbuf=local_hot_radial_gas_velocity_dispersion_array.flatten(), 
             recvbuf=(hot_radial_gas_velocity_dispersion_array, sendcounts), 
             root=0)
comm.Gatherv(sendbuf=local_hot_rotational_gas_velocity_array.flatten(), 
             recvbuf=(hot_rotational_gas_velocity_array, sendcounts), 
             root=0)
comm.Gatherv(sendbuf=local_hot_gas_velocity_dispersion_array.flatten(), 
             recvbuf=(hot_gas_velocity_dispersion_array, sendcounts), 
             root=0)

# Save the final profile array on the root process
if rank == 0:
    np.savez(output_file_path,
             radial_bins_inner=radial_bins_inner,
             radial_bins=radial_bins,
             radial_bins_outer=radial_bins_outer,
             redshift=redshift,
             M200c=M200c,
             R200c=R200c,
             M500c=M500c,
             R500c=R500c,
             Mstar=Mstar,
             potential_array=potential_array.reshape(nhalos, nbins),
             DM_density_array=DM_density_array.reshape(nhalos, nbins),
             stellar_density_array=stellar_density_array.reshape(nhalos, nbins),
             gas_density_array=density_array.reshape(nhalos, nbins),
             temperature_array=temperature_array.reshape(nhalos, nbins),
             pressure_array=pressure_array.reshape(nhalos, nbins),
             metallicity_array=metallicity_array.reshape(nhalos, nbins),
             radial_gas_velocity_array=radial_gas_velocity_array.reshape(nhalos, nbins),
             radial_gas_velocity_dispersion_array=radial_gas_velocity_dispersion_array.reshape(nhalos, nbins),
             rotational_gas_velocity_array=rotational_gas_velocity_array.reshape(nhalos, nbins),

             gas_velocity_dispersion_array=gas_velocity_dispersion_array.reshape(nhalos, nbins),
             hot_gas_density_array=hot_density_array.reshape(nhalos, nbins),
             hot_temperature_array=hot_temperature_array.reshape(nhalos, nbins),
             hot_pressure_array=hot_pressure_array.reshape(nhalos, nbins),
             hot_metallicity_array=hot_metallicity_array.reshape(nhalos, nbins),
             hot_radial_gas_velocity_array=hot_radial_gas_velocity_array.reshape(nhalos, nbins),
             hot_radial_gas_velocity_dispersion_array=hot_radial_gas_velocity_dispersion_array.reshape(nhalos, nbins),
             hot_rotational_gas_velocity_array=hot_rotational_gas_velocity_array.reshape(nhalos, nbins),
             hot_gas_velocity_dispersion_array=hot_gas_velocity_dispersion_array.reshape(nhalos, nbins)
            )
