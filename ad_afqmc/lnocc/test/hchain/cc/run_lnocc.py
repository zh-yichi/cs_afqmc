from functools import partial
from jax import random, dtypes
#from jax import numpy as jnp
#import argparse
import pickle
#from mpi4py import MPI
import numpy as np
from jax import numpy as jnp
from ad_afqmc import config, lno_ccsd, wavefunctions, stat_utils
import time

print = partial(print, flush=True)

with open("options.bin", "rb") as f:
    options = pickle.load(f)

# with open("files.bin", "rb") as f:
#     files = pickle.load(f)

options["dt"] = options.get("dt", 0.01)
options["n_exp_terms"] = options.get("n_exp_terms",6)
options["n_walkers"] = 40
options["n_blocks"] = 50
rlx_steps = 10
#options["n_runs"] = options.get("n_runs", 100)
options["rlx_steps"] = options.get("rlx_steps", 5)
options["n_prop_steps"] = options.get("n_prop_steps", 10)
options["seed"] = options.get("seed", np.random.randint(1, int(1e6)))

nwalkers = options["n_walkers"]
#dt = options["dt"]
seed = options["seed"]
#prop_steps = options["n_prop_steps"]
#n_runs = options["n_runs"]
use_gpu = options.get("use_gpu", False)

# mo_file=files["mo"]
# chol_file=files["chol"]
# amp_file=files["amp"]

if use_gpu:
    config.afqmc_config["use_gpu"] = True

config.setup_jax()
MPI = config.setup_comm()

init_time = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Process rank
size = comm.Get_size()  # Total number of processes

ham_data, ham, prop, trial, wave_data, sampler, observable, options, _ = (
    lno_ccsd.prep_lnoccsd_afqmc(options,mo_file="mo_full.npz",
                                amp_file="amp_full.npz",chol_file="chol_full"))

### initialize propagation
seed = options["seed"]
propagator = prop
init_walkers = None
ham_data = wavefunctions.rhf(trial.norb, trial.nelec,n_batch=trial.n_batch
                                )._build_measurement_intermediates(ham_data, wave_data)
ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
ham_data = ham.build_propagation_intermediates(ham_data, propagator, trial, wave_data)

prop_data = propagator.init_prop_data(trial, wave_data, ham_data, init_walkers)
if jnp.abs(jnp.sum(prop_data["overlaps"])) < 1.0e-6:
    raise ValueError(
        "Initial overlaps are zero. Pass walkers with non-zero overlap."
    )
prop_data["key"] = random.PRNGKey(seed + rank)

prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
prop_data["n_killed_walkers"] = 0
prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]


### relaxation ###
init_time = time.time()
comm.Barrier()
if rank == 0:
    init_ccsd_cr = lno_ccsd.tot_ccsd_cr(prop_data["walkers"], ham_data, 
                           wave_data, trial,1e-5)[0]
    print(f'# afqmc propagation with ccsd trial using {nwalkers*size} walkers')
    print('# rlx_step \t ccsd_orb_cr \t rlx_time')
    print(f'{0:5d}'
          f'\t \t {init_ccsd_cr:.6f}'
          f'\t {0.00:.2f}')
comm.Barrier()

for n in range(rlx_steps):
    prop_data,(block_energy_n,_) \
                = lno_ccsd.propagate_phaseless_tot(ham_data,prop,prop_data,trial,wave_data,sampler)
    
    block_energy_n = np.array([block_energy_n], dtype="float32")
    block_weight_n = np.array([jnp.sum(prop_data["weights"])], dtype="float32")
    block_weighted_energy_n = np.array(
        [block_energy_n * block_weight_n], dtype="float32"
    )
    total_block_energy_n = np.zeros(1, dtype="float32")
    total_block_weight_n = np.zeros(1, dtype="float32")

    comm.Reduce(
            [block_weighted_energy_n, MPI.FLOAT],
            [total_block_energy_n, MPI.FLOAT],
            op=MPI.SUM,
            root=0,
        )
    comm.Reduce(
        [block_weight_n, MPI.FLOAT],
        [total_block_weight_n, MPI.FLOAT],
        op=MPI.SUM,
        root=0,
    )
    if rank == 0:
        block_weight_n = total_block_weight_n
        block_energy_n = total_block_energy_n / total_block_weight_n
    comm.Bcast(block_weight_n, root=0)
    comm.Bcast(block_energy_n, root=0)
    prop_data = propagator.orthonormalize_walkers(prop_data)
    prop_data = propagator.stochastic_reconfiguration_global(prop_data, comm)
    # prop_data["e_estimate"] = (
    #     0.9 * prop_data["e_estimate"] + 0.1 * block_energy_n[0]
    # )

    comm.Barrier()
    if rank == 0:
        print(
            f"{n+1:5d}"
            f"\t \t {block_energy_n[0]:.6f}"
            f"\t {time.time() - init_time:.2f} ",
            flush=True,
        )
    comm.Barrier()

# observable_op = jnp.array(ham_data["h1"])
# observable_constant = 0.0

global_block_weights = None
global_block_energies = None
# global_block_observables = None
# global_block_rdm1s = None
# global_block_rdm2s = None

if rank == 0:
    global_block_weights = np.zeros(size * sampler.n_blocks)
    global_block_energies = np.zeros(size * sampler.n_blocks)
    #global_block_observables = np.zeros(size * sampler.n_blocks)

prop_data_tangent = {}
for x in prop_data:
    if isinstance(prop_data[x], list):
        prop_data_tangent[x] = [np.zeros_like(y) for y in prop_data[x]]
    elif prop_data[x].dtype == "uint32":
        prop_data_tangent[x] = np.zeros(prop_data[x].shape, dtype=dtypes.float0)
    else:
        prop_data_tangent[x] = np.zeros_like(prop_data[x])
#block_rdm1_n = np.zeros_like(ham_data["h1"])
#block_rdm2_n = None
#block_observable_n = 0.0

local_large_deviations = np.array(0)

comm.Barrier()
init_time = time.time()
if rank == 0:
    print("# Sampling sweeps:")
    print("# Iter \t Mean energy \t error \t Walltime")
comm.Barrier()

for n in range(sampler.n_blocks):
    prop_data,(block_energy_n,_) \
        = lno_ccsd.propagate_phaseless_tot(ham_data,prop,prop_data,trial,wave_data,sampler)
    #block_observable_n = 0.0

    block_energy_n = np.array([block_energy_n], dtype="float32")
    # block_observable_n = np.array(
    #     [block_observable_n + observable_constant], dtype="float32"
    # )
    block_weight_n = np.array([jnp.sum(prop_data["weights"])], dtype="float32")
    #block_rdm1_n = np.array(block_rdm1_n, dtype="float32")

    gather_weights = None
    gather_energies = None
    #gather_observables = None
    gather_rdm1s = None
    gather_rdm2s = None

    if rank == 0:
        gather_weights = np.zeros(size, dtype="float32")
        gather_energies = np.zeros(size, dtype="float32")
        #gather_observables = np.zeros(size, dtype="float32")

    comm.Gather(block_weight_n, gather_weights, root=0)
    comm.Gather(block_energy_n, gather_energies, root=0)
    #comm.Gather(block_observable_n, gather_observables, root=0)

    block_energy_n = 0.0
    if rank == 0:
        global_block_weights[n * size : (n + 1) * size] = gather_weights
        global_block_energies[n * size : (n + 1) * size] = gather_energies
        #global_block_observables[n * size : (n + 1) * size] = gather_observables

        assert gather_weights is not None
        block_energy_n = np.sum(gather_weights * gather_energies) / np.sum(
            gather_weights
        )
    
    block_energy_n = comm.bcast(block_energy_n, root=0)
    prop_data = propagator.orthonormalize_walkers(prop_data)
    prop_data = propagator.stochastic_reconfiguration_global(prop_data, comm)
    # prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * block_energy_n

    if n % (max(sampler.n_blocks // 10, 1)) == 0:
        comm.Barrier()
        if rank == 0:
                e_afqmc, energy_error = stat_utils.blocking_analysis(
                    global_block_weights[: (n + 1) * size],
                    global_block_energies[: (n + 1) * size],
                    neql=0,
                )
                # obs_afqmc, _ = stat_utils.blocking_analysis(
                #     global_block_weights[: (n + 1) * size],
                #     global_block_observables[: (n + 1) * size],
                #     neql=0,
                # )
                if energy_error is not None:
                    print(f"{n:5d} \t {e_afqmc:.6f} \t {energy_error:.6f} \t {time.time() - init_time:.2f} ")
                else:
                    print(f"{n:5d} \t {e_afqmc:.6f} \t\t - \t {time.time() - init_time:.2f} ")
        comm.Barrier()

global_large_deviations = np.array(0)
comm.Reduce(
    [local_large_deviations, MPI.INT],
    [global_large_deviations, MPI.INT],
    op=MPI.SUM,
    root=0,
)
comm.Barrier()
if rank == 0:
    print(f"# Number of large deviations: {global_large_deviations}", flush=True)
comm.Barrier()
e_afqmc, e_err_afqmc = None, None
comm.Barrier()
if rank == 0:
    assert global_block_weights is not None
    assert global_block_energies is not None
    #assert global_block_observables is not None
    samples_clean, idx = stat_utils.reject_outliers(
    np.stack(
                (
                    global_block_weights,
                    global_block_energies
                )
            ).T,
            1,
        )

    print(
        f"# Number of outliers in post: {global_block_weights.size - samples_clean.shape[0]} "
        )
    global_block_weights = samples_clean[:, 0]
    global_block_energies = samples_clean[:, 1]

    e_afqmc, e_err_afqmc = stat_utils.blocking_analysis(
    global_block_weights, global_block_energies, neql=0, printQ=True
    )

    if e_err_afqmc is not None:
        sig_dec = int(abs(np.floor(np.log10(e_err_afqmc))))
        sig_err = np.around(
            np.round(e_err_afqmc * 10**sig_dec) * 10 ** (-sig_dec), sig_dec
        )
        sig_e = np.around(e_afqmc, sig_dec)
        print(f"ccsd energy correction: {sig_e:.{sig_dec}f} +/- {sig_err:.{sig_dec}f}\n")
    elif e_afqmc is not None:
        print(f"ccsd energy correction: {e_afqmc}\n", flush=True)
        e_err_afqmc = 0.0

    comm.Barrier()
    e_afqmc = comm.bcast(e_afqmc, root=0)
    e_err_afqmc = comm.bcast(e_err_afqmc, root=0)
    comm.Barrier()
comm.Barrier()
