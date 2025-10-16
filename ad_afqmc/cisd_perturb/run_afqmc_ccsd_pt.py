import time
import numpy as np
from jax import random
from jax import numpy as jnp
from ad_afqmc import config, wavefunctions, stat_utils, mpi_jax
from ad_afqmc.cisd_perturb import sample_ccsd_pt, ccsd_pt

ham_data, ham, prop, trial, wave_data, sampler, observable, options, _ \
    = (mpi_jax._prep_afqmc())
trial = wavefunctions.rhf(trial.norb, trial.nelec,n_batch=trial.n_batch)

norb = trial.norb
chol = ham_data["chol"].reshape(-1, norb, norb)
h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
# v1 the one-body energy from the reordering of the 
# two-body operators into non-normal ordered form
v0 = 0.5 * jnp.einsum("gik,gjk->ij",
                        chol.reshape(-1, norb, norb),
                        chol.reshape(-1, norb, norb),
                        optimize="optimal")
h1_mod = h1 - v0 
ham_data['h1_mod'] = h1_mod

nocc = wave_data['ci1'].shape[0]
wave_data["mo_coeff"] = np.eye(norb)[:,:nocc]
t1, t2 = wave_data['ci1'],wave_data['ci2']
wave_data['t1'] = t1
wave_data['t2'] = t2

h0 = ham_data['h0']

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

t, e0, e1 = ccsd_pt._ccsd_walker_energy_pt(
    prop_data['walkers'][0],ham_data,wave_data,trial)
init_ept = e0 + e1 - t*(e0-h0)

comm.Barrier()
init_time = time.time() - init
if rank == 0:
    print("# Equilibration sweeps:")
    print("#   Iter \t energy \t Walltime")
    print(f"  {0:5d} \t {init_ept:.6f} \t {init_time:.2f}")
comm.Barrier()

for n in range(1, neql + 1):
    prop_data, (blk_wt, blk_t, blk_e0, blk_e1) =\
        sample_ccsd_pt.propagate_phaseless(
            prop_data, ham_data, prop, trial, wave_data, sampler)
    
    blk_wt = np.array([blk_wt], dtype="float32")
    blk_t = np.array([blk_t], dtype="float32") #"complex64")
    blk_e0 = np.array([blk_e0], dtype="float32") #"complex64")
    blk_e1 = np.array([blk_e1], dtype="float32") #"complex64")

    # blk_rho = np.array([blk_t * blk_wt], dtype="float32")

    blk_wt_t = np.array([blk_t * blk_wt], dtype="float32") #"complex64"
    blk_wt_e0 = np.array([blk_e0 * blk_wt], dtype="float32") #"complex64"
    blk_wt_e1 = np.array([blk_e1 * blk_wt], dtype="float32") #"complex64"

    tot_blk_wt = np.zeros(1, dtype="float32")
    tot_blk_t = np.zeros(1, dtype="float32") #"complex64")
    tot_blk_e0 = np.zeros(1, dtype="float32") #"complex64")
    tot_blk_e1 = np.zeros(1, dtype="float32") #"complex64")
    
    comm.Reduce(
        [blk_wt, MPI.FLOAT],
        [tot_blk_wt, MPI.FLOAT],
        op=MPI.SUM,
        root=0,
    )
    comm.Reduce(
        [blk_wt_t, MPI.FLOAT],
        [tot_blk_t, MPI.FLOAT],
        op=MPI.SUM,
        root=0,
    )
    comm.Reduce(
        [blk_wt_e0, MPI.FLOAT],
        [tot_blk_e0, MPI.FLOAT],
        op=MPI.SUM,
        root=0,
    )
    comm.Reduce(
        [blk_wt_e1, MPI.FLOAT],
        [tot_blk_e1, MPI.FLOAT],
        op=MPI.SUM,
        root=0,
    )

    comm.Barrier()
    if rank == 0:
        blk_wt = tot_blk_wt
        blk_t = tot_blk_t / tot_blk_wt
        blk_e0 = tot_blk_e0 / tot_blk_wt
        blk_e1 = tot_blk_e1 / tot_blk_wt
    comm.Barrier()

    comm.Bcast(blk_wt, root=0)
    comm.Bcast(blk_t, root=0)
    comm.Bcast(blk_e0, root=0)
    comm.Bcast(blk_e1, root=0)

    blk_ept = blk_e0 + blk_e1 - blk_t * (blk_e0-h0)
    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = (
        0.9 * prop_data["e_estimate"] + 0.1 * blk_ept[0]
    )

    comm.Barrier()
    if rank == 0:
        print(
            f"  {n:5d} \t {blk_ept[0]:.6f} \t {time.time() - init:.2f} ",
            flush=True,
        )
    comm.Barrier()

glb_blk_wt = None
glb_blk_t = None
glb_blk_e0 = None
glb_blk_e1 = None

if rank == 0:
    glb_blk_wt = np.zeros(size * sampler.n_blocks,dtype="float32")
    glb_blk_t = np.zeros(size * sampler.n_blocks,dtype="float32")#"complex64")
    glb_blk_e0 = np.zeros(size * sampler.n_blocks,dtype="float32")#"complex64")
    glb_blk_e1 = np.zeros(size * sampler.n_blocks,dtype="float32")#"complex64")
    ept_samples = np.zeros(sampler.n_blocks,dtype="float32")
    
comm.Barrier()
if rank == 0:
    print("#\n# Sampling sweeps:")
    print("#  Iter \t energy \t error \t \t Walltime")
comm.Barrier()

for n in range(sampler.n_blocks):
    prop_data, (blk_wt, blk_t, blk_e0, blk_e1) = \
        sample_ccsd_pt.propagate_phaseless(
            prop_data, ham_data, prop, trial, wave_data, sampler
        )
    blk_wt = np.array([blk_wt], dtype="float32")
    blk_t = np.array([blk_t], dtype="float32")#"complex64")
    blk_e0 = np.array([blk_e0], dtype="float32")#"complex64")
    blk_e1 = np.array([blk_e1], dtype="float32")#"complex64")

    gather_wt = None
    gather_t = None
    gather_e0 = None
    gather_e1 = None

    comm.Barrier()
    if rank == 0:
        gather_wt = np.zeros(size, dtype="float32")
        gather_t = np.zeros(size, dtype="float32")#"complex64")
        gather_e0 = np.zeros(size, dtype="float32")#"complex64")
        gather_e1 = np.zeros(size, dtype="float32")#"complex64")
    comm.Barrier()

    comm.Gather(blk_wt, gather_wt, root=0)
    comm.Gather(blk_t, gather_t, root=0)
    comm.Gather(blk_e0, gather_e0, root=0)
    comm.Gather(blk_e1, gather_e1, root=0)

    comm.Barrier()
    if rank == 0:
        glb_blk_wt[n * size : (n + 1) * size] = gather_wt
        glb_blk_t[n * size : (n + 1) * size] = gather_t
        glb_blk_e0[n * size : (n + 1) * size] = gather_e0
        glb_blk_e1[n * size : (n + 1) * size] = gather_e1

        assert gather_wt is not None

        blk_wt= np.sum(gather_wt)
        blk_t = np.sum(gather_wt * gather_t) / blk_wt
        blk_e0 = np.sum(gather_wt * gather_e0) / blk_wt
        blk_e1 = np.sum(gather_wt * gather_e1) / blk_wt
    comm.Barrier()

    comm.Bcast(blk_wt, root=0)
    comm.Bcast(blk_t, root=0)
    comm.Bcast(blk_e0, root=0)
    comm.Bcast(blk_e1, root=0)

    blk_ept = blk_e0 + blk_e1 - blk_t * (blk_e0 - h0)
    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * blk_ept

    comm.Barrier()
    if rank == 0:
        ept_samples[n] = blk_ept
    comm.Barrier()

    if n % (max(sampler.n_blocks // 10, 1)) == 0:
        comm.Barrier()
        if rank == 0:
                # t, t_err = stat_utils.blocking_analysis(
                #     glb_blk_wt[: (n + 1) * size],
                #     glb_blk_t[: (n + 1) * size],
                #     neql=0,
                # )
                # e0, e0_err = stat_utils.blocking_analysis(
                #     glb_blk_wt[: (n + 1) * size],
                #     glb_blk_e0[: (n + 1) * size],
                #     neql=0,
                # )
                # e1, e1_err = stat_utils.blocking_analysis(
                #     glb_blk_wt[: (n + 1) * size],
                #     glb_blk_e1[: (n + 1) * size],
                #     neql=0,
                # )

                t = np.sum(glb_blk_wt * glb_blk_t)/np.sum(glb_blk_wt)
                e0 = np.sum(glb_blk_wt * glb_blk_e0)/np.sum(glb_blk_wt)
                e1 = np.sum(glb_blk_wt * glb_blk_e1)/np.sum(glb_blk_wt)

                ept = e0 + e1 - t * (e0 - h0)
                # dE = [(dE/dt),(dE/de0),(dE/de1)]
                dE = np.array([-e0+h0,1-t,1])
                cov_te0e1 = np.cov([glb_blk_t[:(n+1)*size],
                                    glb_blk_e0[:(n+1)*size],
                                    glb_blk_e1[:(n+1)*size]])
                ept_err = np.sqrt(dE @ cov_te0e1 @ dE)/np.sqrt((n+1)*size)
                
                # if t_err is None:
                #     ept_err = "  None  "
                # if e0_err is None:
                #     ept_err = "  None  "
                # if e1_err is None:
                #     ept_err = "  None  "
                # else:
                # ept_err = f"{ept_err:.6f}"

                # ept = f"{ept:.6f}"
                # ept_err = f"{ept_err}"

                # if t_err is not None:
                #     t_err = f"{t_err:.6f}"
                # else:
                #     t_err = f"  {t_err}  "
                # if e0_err is not None:
                #     e0_err = f"{e0_err:.6f}"
                # else:
                #     e0_err = f"  {e0_err}  "
                # if e1_err is not None:
                #     e1_err = f"{e1_err:.6f}"
                # else:
                #     e1_err = f"  {e1_err}  "

                print(f"  {n:4d} \t \t {ept:.6f} \t {ept_err:.6f} \t"
                      f"  {time.time() - init:.2f}")
        comm.Barrier()
    # comm.bcast(ept, root=0)
    # prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * ept

comm.Barrier()
if rank == 0:
    assert glb_blk_wt is not None
    samples_clean, idx = stat_utils.reject_outliers(
    np.stack(
                (
                    glb_blk_wt,
                    glb_blk_t,
                    glb_blk_e0,
                    glb_blk_e1,
                )
            ).T,
            1,
        )

    print(
        f"# Number of outliers in post: {glb_blk_wt.size - samples_clean.shape[0]} "
        )
    
    glb_blk_wt = samples_clean[:, 0]
    glb_blk_t = samples_clean[:, 1]
    glb_blk_e0 = samples_clean[:, 2]
    glb_blk_e1 = samples_clean[:, 3]

    # t, t_err = stat_utils.blocking_analysis(
    #     glb_blk_wt[: (n + 1) * size],
    #     glb_blk_t[: (n + 1) * size],
    #     neql=0,
    # )
    # e0, e0_err = stat_utils.blocking_analysis(
    #     glb_blk_wt[: (n + 1) * size],
    #     glb_blk_e0[: (n + 1) * size],
    #     neql=0,
    # )
    # e1, e1_err = stat_utils.blocking_analysis(
    #     glb_blk_wt[: (n + 1) * size],
    #     glb_blk_e1[: (n + 1) * size],
    #     neql=0,
    # )

    t = np.sum(glb_blk_wt * glb_blk_t)/np.sum(glb_blk_wt)
    e0 = np.sum(glb_blk_wt * glb_blk_e0)/np.sum(glb_blk_wt)
    e1 = np.sum(glb_blk_wt * glb_blk_e1)/np.sum(glb_blk_wt)

    ept = e0 + e1 - t*(e0-h0)
    # dE = [(dE/dt),(dE/de0),(dE/de1)]
    dE = np.array([-e0+h0,1-t,1])
    cov_te0e1 = np.cov([glb_blk_t,glb_blk_e0,glb_blk_e1])
    ept_err = np.sqrt(dE @ cov_te0e1 @ dE)/np.sqrt(samples_clean.shape[0])
    
    # if t_err is None:
    #     ept_err = "  None  "
    # if e0_err is None:
    #     ept_err = "  None  "
    # if e1_err is None:
    #     ept_err = "  None  "
    # else:
    ept_err = f"{ept_err:.6f}"

    ept = f"{ept:.6f}"

    print(f"Final Results1:")
    print(f"AFQMC/CCSD_PT energy: {ept} +/- {ept_err}")

    d = np.abs(ept_samples-np.median(ept_samples))
    d_med = np.median(d) + 1e-7
    mask = d/d_med < 10
    ept_clean = ept_samples[mask]
    print('# remove outliers: ', len(ept_samples)-len(ept_clean))

    ept_mean = np.mean(ept_clean)
    ept_std = np.std(ept_clean)

    ept = f"{ept_mean:.6f}"
    ept_err = f"{ept_std/np.sqrt(n):.6f}"

    print(f"Final Results2:")
    print(f"AFQMC/CCSD_PT energy: {ept} +/- {ept_err}")
    print(f"total run time: {time.time() - init:.2f}")

comm.Barrier()

