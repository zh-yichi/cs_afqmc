from functools import partial
from jax import random
#from mpi4py import MPI
import numpy as np
from jax import numpy as jnp
from ad_afqmc import config, wavefunctions, stat_utils
from ad_afqmc.lno_afqmc import afqmc_maker, lnoafqmc_runner
import time

print = partial(print, flush=True)

ham_data, ham, prop, trial, wave_data, sampler, observable, options, _ = (
    lnoafqmc_runner.prep_lnoafqmc_run())

if options["use_gpu"]:
    config.afqmc_config["use_gpu"] = True

config.setup_jax()
MPI = config.setup_comm()

# init_time = time.time()
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
    eorb = afqmc_maker._orb_energy(prop_data["walkers"][0],ham_data,wave_data,trial)
    e_init = prop_data["e_estimate"]

    print(f'# afqmc propagation with {options["n_walkers"]*size} walkers')
    print(f'# appraching equilibrium')
    print('# step  energy  orb_energy  time')
    print(f"  {0:3d}"
          f"  {e_init:.6f}"
          f"  {eorb:.6f}"
          f"  {time.time() - init_time:.2f} "
        )
comm.Barrier()

for n in range(options["n_eql"]):
    prop_data,(blk_wt,blk_en,blk_eorb) \
        = afqmc_maker.propagate_phaseless_orb(
            ham_data,prop,prop_data,trial,wave_data,sampler)

    blk_wt = np.array([blk_wt], dtype="float32") 
    blk_en = np.array([blk_en], dtype="float32")
    blk_eorb = np.array([blk_eorb], dtype="float32")    

    blk_wt_en = np.array(
        [blk_en * blk_wt], dtype="float32"
    )
    blk_wt_eorb = np.array(
        [blk_eorb * blk_wt], dtype="float32"
    )

    tot_wt = np.zeros(1, dtype="float32")
    tot_wt_en = np.zeros(1, dtype="float32")
    tot_wt_eorb = np.zeros(1, dtype="float32")

    comm.Reduce(
        [blk_wt, MPI.FLOAT],
        [tot_wt, MPI.FLOAT],
        op=MPI.SUM,
        root=0,
        )
    comm.Reduce(
            [blk_wt_en, MPI.FLOAT],
            [tot_wt_en, MPI.FLOAT],
            op=MPI.SUM,
            root=0,
        )
    comm.Reduce(
            [blk_wt_eorb, MPI.FLOAT],
            [tot_wt_eorb, MPI.FLOAT],
            op=MPI.SUM,
            root=0,
        )

    comm.Barrier()
    if rank == 0:
        blk_wt = tot_wt
        blk_en = tot_wt_en / tot_wt
        blk_eorb = tot_wt_eorb / tot_wt

    comm.Bcast(blk_wt, root=0)
    comm.Bcast(blk_en, root=0)
    comm.Bcast(blk_eorb, root=0)
    
    prop_data = propagator.orthonormalize_walkers(prop_data)
    prop_data = propagator.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = (
         0.9 * prop_data["e_estimate"] + 0.1 * blk_en[0]
         )
    comm.Barrier()

    comm.Barrier()
    if rank == 0:
        print(
            f"  {n+1:3d}"
            f"  {blk_en[0]:.6f}"
            f"  {blk_eorb[0]:.6f}"
            f"  {time.time() - init_time:.2f} "
        )
    comm.Barrier()

glb_blk_wt = None
glb_blk_en = None
glb_blk_eorb = None

if rank == 0:
    glb_blk_wt = np.zeros(size * sampler.n_blocks)
    glb_blk_en = np.zeros(size * sampler.n_blocks)
    glb_blk_eorb = np.zeros(size * sampler.n_blocks)
    
comm.Barrier()
if rank == 0:
    print("# Sampling sweeps:")
    print("# iter  energy    err    orb_energy    err    time")
comm.Barrier()

for n in range(sampler.n_blocks):
    prop_data,(blk_wt,blk_en,blk_eorb) \
        = afqmc_maker.propagate_phaseless_orb(
            ham_data,prop,prop_data,trial,wave_data,sampler)

    blk_wt = np.array([blk_wt], dtype="float32")
    blk_en = np.array([blk_en], dtype="float32")
    blk_eorb = np.array([blk_eorb], dtype="float32")

    gather_wt = None
    gather_en = None
    gather_eorb = None

    comm.Barrier()
    if rank == 0:
        gather_wt = np.zeros(size, dtype="float32")
        gather_en = np.zeros(size, dtype="float32")
        gather_eorb = np.zeros(size, dtype="float32")
    comm.Barrier()

    comm.Gather(blk_wt, gather_wt, root=0)
    comm.Gather(blk_en, gather_en, root=0)
    comm.Gather(blk_eorb, gather_eorb, root=0)

    comm.Barrier()
    if rank == 0:
        glb_blk_en[n * size : (n + 1) * size] = gather_en
        glb_blk_wt[n * size : (n + 1) * size] = gather_wt
        glb_blk_eorb[n * size : (n + 1) * size] = gather_eorb

        assert gather_wt is not None

        blk_wt= np.sum(gather_wt)
        blk_en = np.sum(gather_wt * gather_en) / blk_wt
    comm.Barrier()
    
    prop_data = propagator.orthonormalize_walkers(prop_data)
    prop_data = propagator.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * blk_en

    if n % (max(sampler.n_blocks // 10, 1)) == 0:
        comm.Barrier()
        if rank == 0:
                en, en_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_en[: (n + 1) * size],
                    neql=0,
                )
                eorb, eorb_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_eorb[: (n + 1) * size],
                    neql=0,
                )

                if en_err is not None:
                    en_err = f"{en_err:.6f}"
                else:
                    en_err = f"  {en_err}  "
                if eorb_err is not None:
                    eorb_err = f"{eorb_err:.6f}"
                else:
                    eorb_err = f"  {eorb_err}  "

                en = f"{en:.6f}"
                eorb = f"{eorb:.6f}"

                print(f"  {n:4d}  {en}  {en_err}  {eorb}  {eorb_err}"
                      f"  {time.time() - init_time:.2f}")
        comm.Barrier()


comm.Barrier()
if rank == 0:
    assert glb_blk_wt is not None
    assert glb_blk_en is not None
    assert glb_blk_eorb is not None


    samples_clean, idx = stat_utils.reject_outliers(
    np.stack(
                (
                    glb_blk_wt,
                    glb_blk_en,
                    glb_blk_eorb,
                )
            ).T,
            1,
        )

    print(
        f"# Number of outliers in post: {glb_blk_wt.size - samples_clean.shape[0]} "
        )
    
    glb_blk_wt = samples_clean[:, 0]
    glb_blk_en = samples_clean[:, 1]
    glb_blk_eorb = samples_clean[:, 2]

    en, en_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_en[: (n + 1) * size],
                    neql=0,printQ=True
                )
    eorb, eorb_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_eorb[: (n + 1) * size],
                    neql=0,printQ=True
                )
    
    if en_err is not None:
        en_err = f"{en_err:.6f}"
    else:
        en_err = f"  {en_err}  "
    if eorb_err is not None:
        eorb_err = f"{eorb_err:.6f}"
    else:
        eorb_err = f"  {eorb_err}  "

    en = f"{en:.6f}"
    eorb = f"{eorb:.6f}"

    print(f"Final Results:")
    print(f"lno-afqmc/cisd energy: {en} +/- {en_err}")
    print(f"lno-afqmc/cisd orbital energy: {eorb} +/- {eorb_err}")
    print(f"total run time: {time.time() - init_time:.2f}")

comm.Barrier()
