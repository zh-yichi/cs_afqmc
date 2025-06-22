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

# with open("options.bin", "rb") as f:
#     options = pickle.load(f)

# # options['n_prop_steps'] = 50
# # options['n_sr_blocks'] = 10
# nwalkers = options["n_walkers"]
# seed = options["seed"]
# eql_steps = options["n_eql"]
# use_gpu = options.get("use_gpu", False)

ham_data, ham, prop, trial, wave_data, sampler, observable, options, _ = (
    lno_ccsd.prep_lnoccsd_afqmc())

# use_gpu = options.get("use_gpu", False)

if options["use_gpu"]:
    config.afqmc_config["use_gpu"] = True

config.setup_jax()
MPI = config.setup_comm()

init_time = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Process rank
size = comm.Get_size()  # Total number of processes


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
    init_ccsd_cr = lno_ccsd._frg_ccsd_orb_cr(
        prop_data["walkers"][0],ham_data,wave_data,trial,1e-5)
    print(f'# afqmc propagation with ccsd trial using {options["n_walkers"]*size} walkers')
    print(f'# system relaxing and appraching the equilibrium of the Markov Chain')
    #print('# eql_step \t hf_orb_cr \t ccsd_orb_cr \t  olp_rt \t wall_time')
    print('# eql_step \t sys_energy \t   orb_cr \t wall_time')
    print(f'{0:5d}'
          f'\t \t {prop_data["e_estimate"]:.6f}'
          f'\t {init_ccsd_cr:.6f}'
          f'\t    {0:.2f}')
comm.Barrier()

for n in range(options["n_eql"]):
    # prop_data,(block_energy_n,_) \
    prop_data, (blk_energy,blk_orb_cr,blk_wt)\
                = lno_ccsd.propagate_phaseless_orb(ham_data,prop,prop_data,trial,wave_data,sampler)
    
    # blk_hf_orb_cr_n = np.array([blk_hf_orb_cr_n], dtype="float32")
    # blk_ccsd_orb_cr_n = np.array([blk_ccsd_orb_cr_n], dtype="float32")
    # blk_olp_rt_n = np.array([blk_olp_rt_n], dtype="float32")
    # blk_wt_n = np.array([blk_wt_n], dtype="float32")
    # blk_wt_n = np.array([jnp.sum(prop_data["weights"])], dtype="float32")
    blk_energy = np.array([blk_energy], dtype="float32")
    blk_orb_cr = np.array([blk_orb_cr], dtype="float32")
    blk_wt = np.array([blk_wt], dtype="float32")

    blk_wt_energy = np.array(
        [blk_energy * blk_wt], dtype="float32"
    )
    blk_wt_orb_cr = np.array(
        [blk_orb_cr * blk_wt], dtype="float32"
    )
    # blk_wt_olp_rt_n = np.array(
    #     [blk_olp_rt_n * blk_wt_n], dtype="float32"
    # )

    tot_wt_energy = np.zeros(1, dtype="float32")
    tot_wt_orb_cr = np.zeros(1, dtype="float32")
    tot_wt = np.zeros(1, dtype="float32")

    comm.Reduce(
            [blk_wt_energy, MPI.FLOAT],
            [tot_wt_energy, MPI.FLOAT],
            op=MPI.SUM,
            root=0,
        )
    comm.Reduce(
            [blk_wt_orb_cr, MPI.FLOAT],
            [tot_wt_orb_cr, MPI.FLOAT],
            op=MPI.SUM,
            root=0,
        )
    comm.Reduce(
        [blk_wt, MPI.FLOAT],
        [tot_wt, MPI.FLOAT],
        op=MPI.SUM,
        root=0,
    )
    if rank == 0:
        blk_energy = tot_wt_energy / tot_wt
        blk_orb_cr = tot_wt_orb_cr / tot_wt
        blk_wt = tot_wt

    comm.Bcast(blk_energy, root=0)
    comm.Bcast(blk_orb_cr, root=0)
    comm.Bcast(blk_wt, root=0)
    prop_data = propagator.orthonormalize_walkers(prop_data)
    prop_data = propagator.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = (
         0.9 * prop_data["e_estimate"] + 0.1 * blk_energy[0]
         )

    comm.Barrier()
    if rank == 0:
        print(
            f"{n+1:5d}"
            f"\t \t {blk_energy[0]:.6f}"
            f"\t {blk_orb_cr[0]:.6f}"
            f"\t   {time.time() - init_time:.2f} "
        )
    comm.Barrier()


glb_blk_wt = None
glb_blk_energy = None
glb_blk_orb_cr = None

if rank == 0:
    glb_blk_energy = np.zeros(size * sampler.n_blocks)
    glb_blk_orb_cr = np.zeros(size * sampler.n_blocks)
    glb_blk_wt = np.zeros(size * sampler.n_blocks)
    #global_block_observables = np.zeros(size * sampler.n_blocks)

# prop_data_tangent = {}
# for x in prop_data:
#     if isinstance(prop_data[x], list):
#         prop_data_tangent[x] = [np.zeros_like(y) for y in prop_data[x]]
#     elif prop_data[x].dtype == "uint32":
#         prop_data_tangent[x] = np.zeros(prop_data[x].shape, dtype=dtypes.float0)
#     else:
#         prop_data_tangent[x] = np.zeros_like(prop_data[x])
#block_rdm1_n = np.zeros_like(ham_data["h1"])
#block_rdm2_n = None
#block_observable_n = 0.0

# local_large_deviations = np.array(0)

comm.Barrier()
init_time = time.time()
if rank == 0:
    print("# Sampling sweeps:")
    print("# Iter \t system energy \t error \t orbital correction \t error \t Walltime")
comm.Barrier()

for n in range(sampler.n_blocks):
    prop_data,(blk_energy,blk_orb_cr,blk_wt) \
        = lno_ccsd.propagate_phaseless_orb(ham_data,prop,prop_data,trial,wave_data,sampler)
    #block_observable_n = 0.0

    blk_energy = np.array([blk_energy], dtype="float32")
    blk_orb_cr = np.array([blk_orb_cr], dtype="float32")
    blk_wt = np.array([blk_wt], dtype="float32")

    gather_energy = None
    gather_orb_cr = None
    gather_wt = None

    if rank == 0:
        gather_energy = np.zeros(size, dtype="float32")
        gather_orb_cr = np.zeros(size, dtype="float32")
        gather_wt = np.zeros(size, dtype="float32")

    comm.Gather(blk_energy, gather_energy, root=0)
    comm.Gather(blk_orb_cr, gather_orb_cr, root=0)
    comm.Gather(blk_wt, gather_wt, root=0)
    #comm.Gather(block_energy_n, gather_energies, root=0)
    #comm.Gather(block_observable_n, gather_observables, root=0)

    #block_energy_n = 0.0
    if rank == 0:
        glb_blk_energy[n * size : (n + 1) * size] = gather_energy
        glb_blk_orb_cr[n * size : (n + 1) * size] = gather_orb_cr
        glb_blk_wt[n * size : (n + 1) * size] = gather_wt
        #global_block_energies[n * size : (n + 1) * size] = gather_energies
        #global_block_observables[n * size : (n + 1) * size] = gather_observables

        assert gather_wt is not None
        blk_energy = np.sum(gather_wt * gather_energy) / np.sum(gather_wt)
        blk_orb_cr= np.sum(gather_wt * gather_orb_cr) / np.sum(gather_wt)
    
    #block_energy_n = comm.bcast(block_energy_n, root=0)
    prop_data = propagator.orthonormalize_walkers(prop_data)
    prop_data = propagator.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * blk_energy

    if n % (max(sampler.n_blocks // 10, 1)) == 0:
        comm.Barrier()
        if rank == 0:
                energy, energy_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_energy[: (n + 1) * size],
                    neql=0,
                )
                orb_cr, orb_cr_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_orb_cr[: (n + 1) * size],
                    neql=0,
                )
                if energy_err is not None and orb_cr_err is not None:
                    print(f"{n:5d} \t {energy:.6f} \t {energy_err:.6f}"
                          f"\t {orb_cr:.6f} \t {orb_cr_err:.6f} " 
                          f"\t {time.time() - init_time:.2f} ")
                else:
                    if energy_err is not None:
                        print(f"{n:5d} \t {energy:.6f} \t {energy_err:.6f}"
                          f"\t {orb_cr:.6f} \t  - " 
                          f"\t {time.time() - init_time:.2f} ")
                    if orb_cr_err is not None:
                        print(f"{n:5d} \t {energy:.6f} \t  -"
                          f"\t {orb_cr:.6f} \t {orb_cr_err:.6f}" 
                          f"\t {time.time() - init_time:.2f} ")
                        
                #   print(f"{n:5d} \t {e_afqmc:.6f} \t\t - \t {time.time() - init_time:.2f} ")
        comm.Barrier()

# global_large_deviations = np.array(0)
# comm.Reduce(
#     [local_large_deviations, MPI.INT],
#     [global_large_deviations, MPI.INT],
#     op=MPI.SUM,
#     root=0,
# )
# comm.Barrier()
# if rank == 0:
#     print(f"# Number of large deviations: {global_large_deviations}", flush=True)
# comm.Barrier()
# e_afqmc, e_err_afqmc = None, None
comm.Barrier()
if rank == 0:
    assert glb_blk_energy is not None
    assert glb_blk_orb_cr is not None
    assert glb_blk_wt is not None
    # assert global_block_energies is not None
    #assert global_block_observables is not None
    samples_clean, idx = stat_utils.reject_outliers(
    np.stack(
                (
                    glb_blk_wt,
                    glb_blk_energy,
                    glb_blk_orb_cr,
                )
            ).T,
            1,
        )

    print(
        f"# Number of outliers in post: {glb_blk_wt.size - samples_clean.shape[0]} "
        )
    glb_blk_wt = samples_clean[:, 0]
    glb_blk_energy = samples_clean[:, 1]
    glb_blk_orb_cr = samples_clean[:, 2]

    # e_afqmc, e_err_afqmc = stat_utils.blocking_analysis(
    # global_block_weights, global_block_energies, neql=0, printQ=True
    # )

    energy, energy_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_energy[: (n + 1) * size],
                    neql=0,printQ=True
                )
    orb_cr, orb_cr_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_orb_cr[: (n + 1) * size],
                    neql=0,printQ=True
                )

    if energy_err is not None and orb_cr_err is not None:
        print(f"system energy: {energy:.6f} +/- {energy_err:.{6}f}")
        print(f"orbital correction: {orb_cr:.6f} +/- {orb_cr_err:.{6}f}")
        # sig_dec = int(abs(np.floor(np.log10(hf_orb_cr_err))))
        # sig_err = np.around(
        #     np.round(e_err_afqmc * 10**sig_dec) * 10 ** (-sig_dec), sig_dec
        # )
        # sig_e = np.around(e_afqmc, sig_dec)
        # print(f"ccsd energy correction: {sig_e:.{sig_dec}f} +/- {sig_err:.{sig_dec}f}\n")
    else:
        if energy_err is not None:
            print(f"system energy: {energy:.6f} +/- {energy_err:.{6}f}")
            print(f"orbital correction: {orb_cr:.6f} +/- None")
        if orb_cr_err is not None:
            print(f"system energy: {energy:.6f} +/- None")
            print(f"orbital correction: {orb_cr:.6f} +/- {orb_cr_err:.{6}f}")
        #e_err_afqmc = 0.0
comm.Barrier()
    #e_afqmc = comm.bcast(e_afqmc, root=0)
    #e_err_afqmc = comm.bcast(e_err_afqmc, root=0)