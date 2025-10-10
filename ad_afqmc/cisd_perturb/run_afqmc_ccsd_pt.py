import time
import numpy as np
from jax import random
from jax import numpy as jnp
from functools import partial
from ad_afqmc import config, wavefunctions, stat_utils, mpi_jax
from ad_afqmc.cisd_perturb import sample_pt2, cisd_pt2

ham_data, ham, prop, trial, wave_data, sampler, observable, options, _ \
    = (mpi_jax._prep_afqmc())
# trial = wavefunctions.cisd_pt(trial.norb, trial.nelec,n_batch=trial.n_batch)

norb = trial.norb
chol = ham_data["chol"].reshape(-1, norb, norb)
h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
# v1 the one-body energy from the reordering of the 
# two-body operators into non-normal ordered form
v0 = 0.5 * jnp.einsum("gik,gjk->ij",
                        chol.reshape(-1, norb, norb),
                        chol.reshape(-1, norb, norb),
                        optimize="optimal")

nocc = wave_data['ci1'].shape[0]
wave_data["mo_coeff"] = np.eye(norb)[:,:nocc]
ci1,ci2 = wave_data['ci1'],wave_data['ci2']
t2 = ci2 - jnp.einsum('ia,jb->iajb',ci1,ci1)
wave_data['t1'] = ci1
wave_data['t2'] = t2

h1_mod = h1 - v0 
ham_data['h1_mod'] = h1_mod

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

e0, e1, t = cisd_pt2._ccsd_walker_energy_pt(
    prop_data['walkers'][0],ham_data,wave_data,trial)
init_ept = ham_data['h0'] + e0 + e1 - t*e0

comm.Barrier()
init_time = time.time() - init
if rank == 0:
    print("# Equilibration sweeps:")
    print("#   Iter \t energy_pt \t Walltime")
    print(f"# {0:5d} \t {init_ept:.6f} \t {init_time:.2f}")
comm.Barrier()

for n in range(1, neql + 1):
    prop_data, (blk_wt, blk_ept) = sample_pt2.propagate_phaseless(
        prop_data, ham_data, prop, trial, wave_data, sampler
    )
    blk_wt = np.array([blk_wt], dtype="float32")
    blk_ept = np.array([blk_ept], dtype="float32")

    blk_wt_ept = np.array(
        [blk_ept * blk_wt], dtype="float32"
    )

    tot_blk_wt = np.zeros(1, dtype="float32")
    tot_blk_ept = np.zeros(1, dtype="float32")
    
    comm.Reduce(
        [blk_wt, MPI.FLOAT],
        [tot_blk_wt, MPI.FLOAT],
        op=MPI.SUM,
        root=0,
    )
    comm.Reduce(
        [blk_wt_ept, MPI.FLOAT],
        [tot_blk_ept, MPI.FLOAT],
        op=MPI.SUM,
        root=0,
    )

    if rank == 0:
        blk_wt = tot_blk_wt
        blk_ept = tot_blk_ept / tot_blk_wt

    comm.Bcast(blk_wt, root=0)
    comm.Bcast(blk_ept, root=0)
    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = (
        0.9 * prop_data["e_estimate"] + 0.1 * blk_ept[0]
    )

    comm.Barrier()
    if rank == 0:
        print(
            f"# {n:5d} \t {blk_ept[0]:.6f} \t {time.time() - init:.2f} ",
            flush=True,
        )
    comm.Barrier()

glb_blk_wt = None
glb_blk_ept = None

if rank == 0:
    glb_blk_wt = np.zeros(size * sampler.n_blocks)
    glb_blk_ept = np.zeros(size * sampler.n_blocks)

comm.Barrier()
if rank == 0:
    print("#\n# Sampling sweeps:")
    print("#  Iter \t energy_pt \t error \t \t Walltime")
comm.Barrier()

for n in range(sampler.n_blocks):
    prop_data, (blk_wt, blk_ept) = \
        sample_pt2.propagate_phaseless(
            prop_data, ham_data, prop, trial, wave_data, sampler
        )
    blk_wt = np.array([blk_wt], dtype="float32")
    blk_ept = np.array([blk_ept], dtype="float32")

    gather_wt = None
    gather_ept = None

    comm.Barrier()
    if rank == 0:
        gather_wt = np.zeros(size, dtype="float32")
        gather_ept = np.zeros(size, dtype="float32")
    comm.Barrier()

    comm.Gather(blk_wt, gather_wt, root=0)
    comm.Gather(blk_ept, gather_ept, root=0)

    comm.Barrier()
    if rank == 0:
        glb_blk_wt[n * size : (n + 1) * size] = gather_wt
        glb_blk_ept[n * size : (n + 1) * size] = gather_ept

        assert gather_wt is not None

        blk_wt= np.sum(gather_wt)
        blk_ept = np.sum(gather_wt * gather_ept) / blk_wt
    comm.Barrier()

    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * blk_ept

    if n % (max(sampler.n_blocks // 10, 1)) == 0:
        comm.Barrier()
        if rank == 0:
                ept, ept_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_ept[: (n + 1) * size],
                    neql=0,
                )

                if ept_err is not None:
                    ept_err = f"{ept_err:.6f}"
                else:
                    ept_err = f"  {ept_err}  "

                ept = f"{ept:.6f}"

                print(f"  {n:4d} \t \t {ept} \t {ept_err} \t"
                      f"  {time.time() - init:.2f}")
        comm.Barrier()

comm.Barrier()
if rank == 0:
    assert glb_blk_wt is not None
    assert glb_blk_ept is not None

    samples_clean, idx = stat_utils.reject_outliers(
    np.stack(
                (
                    glb_blk_wt,
                    glb_blk_ept,
                )
            ).T,
            1,
        )

    print(
        f"# Number of outliers in post: {glb_blk_wt.size - samples_clean.shape[0]} "
        )
    
    glb_blk_wt = samples_clean[:, 0]
    glb_blk_ept = samples_clean[:, 1]

    ept, ept_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_ept[: (n + 1) * size],
                    neql=0,printQ=True
                )
    
    if ept_err is not None:
        ept_err = f"{ept_err:.6f}"
    else:
        ept_err = f"  {ept_err}  "

    ept = f"{ept:.6f}"

    print(f"Final Results:")
    print(f"AFQMC/CCSD_PT energy: {ept} +/- {ept_err}")
    print(f"total run time: {time.time() - init:.2f}")

comm.Barrier()

