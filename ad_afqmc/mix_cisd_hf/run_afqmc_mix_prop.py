import time
import numpy as np
from jax import random
from jax import numpy as jnp
from functools import partial
from ad_afqmc import config, wavefunctions, stat_utils, mpi_jax
from ad_afqmc.mix_cisd_hf import propgate_mix

ham_data, ham, prop, trial, wave_data, sampler, observable, options, _ \
    = (mpi_jax._prep_afqmc())

nao = trial.norb
nocc = wave_data['ci1'].shape[0]
wave_data["mo_coeff"] = np.eye(nao)[:,:nocc]

if options["use_gpu"]:
    config.afqmc_config["use_gpu"] = True

config.setup_jax()
MPI = config.setup_comm()
init = time.time()
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
seed = options["seed"]
neql = options["n_eql"]

trial_rdm1 = trial.get_rdm1(wave_data)
if "rdm1" not in wave_data:
    wave_data["rdm1"] = trial_rdm1

ham_data = wavefunctions.rhf(
    trial.norb, trial.nelec,n_batch=trial.n_batch
    )._build_measurement_intermediates(ham_data, wave_data)
ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
ham_data = ham.build_propagation_intermediates(
    ham_data, prop, trial, wave_data
)
prop_data = prop.init_prop_data(trial, wave_data, ham_data, None)
if jnp.abs(jnp.sum(prop_data["overlaps"])) < 1.0e-6:
    raise ValueError(
        "Initial overlaps are zero. Pass walkers with non-zero overlap."
    )
prop_data["key"] = random.PRNGKey(seed + rank)

comm.Barrier()
init_time = time.time() - init
if rank == 0:
    print("# Equilibration sweeps:")
    print("#   Iter \t energy \t time")
    print(f"# {0:5d} \t {prop_data['e_estimate']:.6f} \t {init_time:.2f}")
comm.Barrier()

for n in range(1, neql + 1):
    prop_data, (blk_wt, blk_en) \
        = propgate_mix.propagate_phaseless(
            prop_data, ham_data, prop, trial, wave_data, sampler)
    
    blk_wt = np.array([blk_wt], dtype="float32")
    blk_en = np.array([blk_en], dtype="float32")

    blk_wt_en = np.array(
        [blk_en * blk_wt], dtype="float32"
    )

    tot_blk_wt = np.zeros(1, dtype="float32")
    tot_blk_en = np.zeros(1, dtype="float32")
    
    comm.Reduce(
        [blk_wt, MPI.FLOAT],
        [tot_blk_wt, MPI.FLOAT],
        op=MPI.SUM,
        root=0,
    )
    comm.Reduce(
        [blk_wt_en, MPI.FLOAT],
        [tot_blk_en, MPI.FLOAT],
        op=MPI.SUM,
        root=0,
    )

    if rank == 0:
        blk_wt = tot_blk_wt
        blk_en = tot_blk_en / tot_blk_wt

    comm.Bcast(blk_wt, root=0)
    comm.Bcast(blk_en, root=0)
    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = (
        0.9 * prop_data["e_estimate"] + 0.1 * blk_en[0]
    )

    comm.Barrier()
    if rank == 0:
        print(
            f"# {n:5d} \t {blk_en[0]:.6f} \t {time.time() - init:.2f} ",
            flush=True,
        )
    comm.Barrier()

glb_blk_wt = None
glb_blk_en = None

if rank == 0:
    glb_blk_wt = np.zeros(size * sampler.n_blocks)
    glb_blk_en = np.zeros(size * sampler.n_blocks)

comm.Barrier()
if rank == 0:
    print("#\n# Sampling sweeps:")
    print("#  Iter \t energy \t error \t \t time")
comm.Barrier()

for n in range(sampler.n_blocks):
    prop_data, (blk_wt, blk_en) = \
        propgate_mix.propagate_phaseless(
            prop_data, ham_data, prop, trial, wave_data, sampler)
    
    blk_wt = np.array([blk_wt], dtype="float32")
    blk_en = np.array([blk_en], dtype="float32")

    gather_wt = None
    gather_en = None

    comm.Barrier()
    if rank == 0:
        gather_wt = np.zeros(size, dtype="float32")
        gather_en = np.zeros(size, dtype="float32")
    comm.Barrier()

    comm.Gather(blk_wt, gather_wt, root=0)
    comm.Gather(blk_en, gather_en, root=0)

    comm.Barrier()
    if rank == 0:
        glb_blk_wt[n * size : (n + 1) * size] = gather_wt
        glb_blk_en[n * size : (n + 1) * size] = gather_en

        assert gather_wt is not None

        blk_wt= np.sum(gather_wt)
        blk_en = np.sum(gather_wt * gather_en) / blk_wt
    comm.Barrier()

    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * blk_en

    if n % (max(sampler.n_blocks // 10, 1)) == 0:
        comm.Barrier()
        if rank == 0:
                en, en_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_en[: (n + 1) * size],
                    neql=0,
                    )

                if en_err is not None:
                    en_err = f"{en_err:.6f}"
                else:
                    en_err = f"  {en_err}  "

                en = f"{en:.6f}"

                print(f"  {n:4d} \t \t {en} \t {en_err} \t"
                        f"  {time.time() - init:.2f}")
        comm.Barrier()

comm.Barrier()
if rank == 0:
    assert glb_blk_wt is not None
    assert glb_blk_en is not None

    samples_clean, idx = stat_utils.reject_outliers(
    np.stack(
                (
                    glb_blk_wt,
                    glb_blk_en,
                )
            ).T,
            1,
        )

    print(
        f"# Number of outliers in post: {glb_blk_wt.size - samples_clean.shape[0]} "
        )
    
    glb_blk_wt = samples_clean[:, 0]
    glb_blk_en = samples_clean[:, 1]

    en, en_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_en[: (n + 1) * size],
                    neql=0,printQ=True
                )
    
    if en_err is not None:
        en_err = f"{en_err:.6f}"
    else:
        en_err = f"  {en_err}  "

    en = f"{en:.6f}"

    print(f"# Final Results")
    print(f"# size-extensiveness experiment")
    print(f"# Trial - CISD: Guide - RHF")
    print(f"# AFQMC energy: {en} +/- {en_err}")
    print(f"# total run time: {time.time() - init:.2f}")

comm.Barrier()

