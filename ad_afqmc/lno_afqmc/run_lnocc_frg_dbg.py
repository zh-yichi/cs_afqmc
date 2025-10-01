from functools import partial
from jax import random
#from mpi4py import MPI
import numpy as np
from jax import numpy as jnp
from ad_afqmc import config, wavefunctions, stat_utils
from ad_afqmc.lno_afqmc import lnoafqmc_runner, afqmc_maker
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
    olp_r,hf_orb_cr = afqmc_maker._frg_hf_cr_dbg(
        ham_data['rot_h1'], ham_data['rot_chol'],prop_data["walkers"][0],trial,wave_data)
    cc_orb_cr0,cc_orb_cr1,cc_orb_cr2,cc_orb_cr \
        = afqmc_maker._frg_cc_cr_dbg(prop_data["walkers"][0],ham_data,wave_data,trial)
    orb_en = hf_orb_cr + cc_orb_cr
    e_init = prop_data["e_estimate"]

    print(f'# afqmc propagation with {options["n_walkers"]*size} walkers')
    print(f'# appraching equilibrium')
    print('# step  energy  olp_ratio  hf_orb_cr'
          '  cc_orb_cr0  cc_orb_cr1  cc_orb_cr2  cc_orb_cr  orb_en  time')
    print(f"  {0:3d}"
          f"  {e_init:.6f}  {olp_r:.6f}  {hf_orb_cr:.6f}"
          f"  {cc_orb_cr0:.6f}  {cc_orb_cr1:.6f}  {cc_orb_cr2:.6f}"
          f"  {cc_orb_cr:.6f}  {orb_en:.6f}  {time.time() - init_time:.2f} "
        )
comm.Barrier()

for n in range(options["n_eql"]):
    prop_data, (blk_wt,blk_en,blk_olp_r,blk_hf_orb_cr,blk_cc_orb_cr0,
                blk_cc_orb_cr1,blk_cc_orb_cr2,blk_cc_orb_cr,blk_orb_en) \
                = afqmc_maker.propagate_phaseless_orb_dbg(
                    ham_data,prop,prop_data,trial,wave_data,sampler)
    
    blk_wt = np.array([blk_wt], dtype="float32")
    blk_en = np.array([blk_en], dtype="float32")
    blk_olp_r = np.array([blk_olp_r], dtype="float32")
    blk_hf_orb_cr = np.array([blk_hf_orb_cr], dtype="float32")
    blk_cc_orb_cr0 = np.array([blk_cc_orb_cr0], dtype="float32")
    blk_cc_orb_cr1 = np.array([blk_cc_orb_cr1], dtype="float32")
    blk_cc_orb_cr2 = np.array([blk_cc_orb_cr2], dtype="float32")
    blk_cc_orb_cr = np.array([blk_cc_orb_cr], dtype="float32")
    blk_orb_en = np.array([blk_orb_en], dtype="float32")   

    blk_wt_en = np.array(
        [blk_en * blk_wt], dtype="float32"
    )
    blk_wt_olp_r = np.array(
        [blk_olp_r * blk_wt], dtype="float32"
    )
    blk_wt_hf_orb_cr = np.array(
        [blk_hf_orb_cr * blk_wt], dtype="float32"
    )
    blk_wt_cc_orb_cr0 = np.array(
        [blk_cc_orb_cr0 * blk_wt], dtype="float32"
    )
    blk_wt_cc_orb_cr1 = np.array(
        [blk_cc_orb_cr1 * blk_wt], dtype="float32"
    )
    blk_wt_cc_orb_cr2 = np.array(
        [blk_cc_orb_cr2 * blk_wt], dtype="float32"
    )
    blk_wt_cc_orb_cr = np.array(
        [blk_cc_orb_cr * blk_wt], dtype="float32"
    )
    blk_wt_orb_en = np.array(
        [blk_orb_en * blk_wt], dtype="float32"
    )

    tot_wt = np.zeros(1, dtype="float32")
    tot_wt_en = np.zeros(1, dtype="float32")
    tot_wt_olp_r = np.zeros(1, dtype="float32")
    tot_wt_hf_orb_cr = np.zeros(1, dtype="float32")
    tot_wt_cc_orb_cr0 = np.zeros(1, dtype="float32")
    tot_wt_cc_orb_cr1 = np.zeros(1, dtype="float32")
    tot_wt_cc_orb_cr2 = np.zeros(1, dtype="float32")
    tot_wt_cc_orb_cr = np.zeros(1, dtype="float32")
    tot_wt_orb_en = np.zeros(1, dtype="float32")

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
            [blk_wt_olp_r, MPI.FLOAT],
            [tot_wt_olp_r, MPI.FLOAT],
            op=MPI.SUM,
            root=0,
        )
    comm.Reduce(
            [blk_wt_hf_orb_cr, MPI.FLOAT],
            [tot_wt_hf_orb_cr, MPI.FLOAT],
            op=MPI.SUM,
            root=0,
        )
    comm.Reduce(
            [blk_wt_cc_orb_cr0, MPI.FLOAT],
            [tot_wt_cc_orb_cr0, MPI.FLOAT],
            op=MPI.SUM,
            root=0,
        )
    comm.Reduce(
            [blk_wt_cc_orb_cr1, MPI.FLOAT],
            [tot_wt_cc_orb_cr1, MPI.FLOAT],
            op=MPI.SUM,
            root=0,
        )
    comm.Reduce(
            [blk_wt_cc_orb_cr2, MPI.FLOAT],
            [tot_wt_cc_orb_cr2, MPI.FLOAT],
            op=MPI.SUM,
            root=0,
        )
    comm.Reduce(
            [blk_wt_cc_orb_cr, MPI.FLOAT],
            [tot_wt_cc_orb_cr, MPI.FLOAT],
            op=MPI.SUM,
            root=0,
        )
    comm.Reduce(
            [blk_wt_orb_en, MPI.FLOAT],
            [tot_wt_orb_en, MPI.FLOAT],
            op=MPI.SUM,
            root=0,
        )

    comm.Barrier()
    if rank == 0:
        blk_wt = tot_wt
        blk_en = tot_wt_en / tot_wt
        blk_olp_r = tot_wt_olp_r / tot_wt
        blk_hf_orb_cr = tot_wt_hf_orb_cr / tot_wt
        blk_cc_orb_cr0 = tot_wt_cc_orb_cr0 / tot_wt
        blk_cc_orb_cr1 = tot_wt_cc_orb_cr1 / tot_wt
        blk_cc_orb_cr2 = tot_wt_cc_orb_cr2 / tot_wt
        blk_cc_orb_cr = tot_wt_cc_orb_cr / tot_wt
        blk_orb_en = tot_wt_orb_en / tot_wt

    comm.Bcast(blk_wt, root=0)
    comm.Bcast(blk_en, root=0)
    comm.Bcast(blk_olp_r, root=0)
    comm.Bcast(blk_hf_orb_cr, root=0)
    comm.Bcast(blk_cc_orb_cr0, root=0)
    comm.Bcast(blk_cc_orb_cr1, root=0)
    comm.Bcast(blk_cc_orb_cr2, root=0)
    comm.Bcast(blk_cc_orb_cr, root=0)
    comm.Bcast(blk_orb_en, root=0)
    
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
            f"  {blk_olp_r[0]:.6f}"
            f"  {blk_hf_orb_cr[0]:.6f}"
            f"  {blk_cc_orb_cr0[0]:.6f}"
            f"  {blk_cc_orb_cr1[0]:.6f}"
            f"  {blk_cc_orb_cr2[0]:.6f}"
            f"  {blk_cc_orb_cr[0]:.6f}"
            f"  {blk_orb_en[0]:.6f}"
            f"  {time.time() - init_time:.2f} "
        )
    comm.Barrier()

glb_blk_wt = None
glb_blk_en = None
glb_blk_olp_r = None
glb_blk_hf_orb_cr = None
glb_blk_cc_orb_cr0 = None
glb_blk_cc_orb_cr1 = None
glb_blk_cc_orb_cr2 = None
glb_blk_cc_orb_cr = None
glb_blk_orb_en = None

if rank == 0:
    glb_blk_wt = np.zeros(size * sampler.n_blocks)
    glb_blk_en = np.zeros(size * sampler.n_blocks)
    glb_blk_olp_r = np.zeros(size * sampler.n_blocks)
    glb_blk_hf_orb_cr = np.zeros(size * sampler.n_blocks)
    glb_blk_cc_orb_cr0 = np.zeros(size * sampler.n_blocks)
    glb_blk_cc_orb_cr1 = np.zeros(size * sampler.n_blocks)
    glb_blk_cc_orb_cr2 = np.zeros(size * sampler.n_blocks)
    glb_blk_cc_orb_cr = np.zeros(size * sampler.n_blocks)
    glb_blk_orb_en = np.zeros(size * sampler.n_blocks)
    
comm.Barrier()
if rank == 0:
    print("# Sampling sweeps:")
    print("# iter  energy  err  olp_ratio  err  hf_orb_cr  err"
          "  cc_orb_cr0  err  cc_orb_cr1  err  cc_orb_cr2"
          "  cc_orb_cr  err  orb_en  err  time")
comm.Barrier()

for n in range(sampler.n_blocks):
    prop_data, (blk_wt,blk_en,blk_olp_r,blk_hf_orb_cr,blk_cc_orb_cr0,
                blk_cc_orb_cr1,blk_cc_orb_cr2,blk_cc_orb_cr,blk_orb_en) \
        = afqmc_maker.propagate_phaseless_orb_dbg(
            ham_data,prop,prop_data,trial,wave_data,sampler)

    blk_wt = np.array([blk_wt], dtype="float32")
    blk_en = np.array([blk_en], dtype="float32")
    blk_olp_r = np.array([blk_olp_r], dtype="float32")
    blk_hf_orb_cr = np.array([blk_hf_orb_cr], dtype="float32")
    blk_cc_orb_cr0 = np.array([blk_cc_orb_cr0], dtype="float32")
    blk_cc_orb_cr1 = np.array([blk_cc_orb_cr1], dtype="float32")
    blk_cc_orb_cr2 = np.array([blk_cc_orb_cr2], dtype="float32")
    blk_cc_orb_cr = np.array([blk_cc_orb_cr], dtype="float32")
    blk_orb_en = np.array([blk_orb_en], dtype="float32")   

    gather_wt = None
    gather_en = None
    gather_olp_r = None
    gather_hf_orb_cr = None
    gather_cc_orb_cr0 = None
    gather_cc_orb_cr1 = None
    gather_cc_orb_cr2 = None
    gather_cc_orb_cr = None
    gather_orb_en = None  

    comm.Barrier()
    if rank == 0:
        gather_wt = np.zeros(size, dtype="float32")
        gather_en = np.zeros(size, dtype="float32")
        gather_olp_r = np.zeros(size, dtype="float32")
        gather_hf_orb_cr = np.zeros(size, dtype="float32")
        gather_cc_orb_cr0 = np.zeros(size, dtype="float32")
        gather_cc_orb_cr1 = np.zeros(size, dtype="float32")
        gather_cc_orb_cr2 = np.zeros(size, dtype="float32")
        gather_cc_orb_cr = np.zeros(size, dtype="float32")
        gather_orb_en = np.zeros(size, dtype="float32")
    comm.Barrier()

    comm.Gather(blk_wt, gather_wt, root=0)
    comm.Gather(blk_en, gather_en, root=0)
    comm.Gather(blk_olp_r, gather_olp_r, root=0)
    comm.Gather(blk_hf_orb_cr, gather_hf_orb_cr, root=0)
    comm.Gather(blk_cc_orb_cr0, gather_cc_orb_cr0, root=0)
    comm.Gather(blk_cc_orb_cr1, gather_cc_orb_cr1, root=0)
    comm.Gather(blk_cc_orb_cr2, gather_cc_orb_cr2, root=0)
    comm.Gather(blk_cc_orb_cr, gather_cc_orb_cr, root=0)
    comm.Gather(blk_orb_en, gather_orb_en, root=0)

    comm.Barrier()
    if rank == 0:
        glb_blk_wt[n * size : (n + 1) * size] = gather_wt
        glb_blk_en[n * size : (n + 1) * size] = gather_en
        glb_blk_olp_r[n * size : (n + 1) * size] = gather_olp_r
        glb_blk_hf_orb_cr[n * size : (n + 1) * size] = gather_hf_orb_cr
        glb_blk_cc_orb_cr0[n * size : (n + 1) * size] = gather_cc_orb_cr0
        glb_blk_cc_orb_cr1[n * size : (n + 1) * size] = gather_cc_orb_cr1
        glb_blk_cc_orb_cr2[n * size : (n + 1) * size] = gather_cc_orb_cr2
        glb_blk_cc_orb_cr[n * size : (n + 1) * size] = gather_cc_orb_cr
        glb_blk_orb_en[n * size : (n + 1) * size] = gather_orb_en

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
                olp_r, olp_r_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_olp_r[: (n + 1) * size],
                    neql=0,
                )
                hf_orb_cr, hf_orb_cr_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_hf_orb_cr[: (n + 1) * size],
                    neql=0,
                )
                cc_orb_cr0, cc_orb_cr0_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_cc_orb_cr0[: (n + 1) * size],
                    neql=0,
                )
                cc_orb_cr1, cc_orb_cr1_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_cc_orb_cr1[: (n + 1) * size],
                    neql=0,
                )
                cc_orb_cr2, cc_orb_cr2_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_cc_orb_cr2[: (n + 1) * size],
                    neql=0,
                )
                cc_orb_cr, cc_orb_cr_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_cc_orb_cr[: (n + 1) * size],
                    neql=0,
                )
                orb_en, orb_en_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_orb_en[: (n + 1) * size],
                    neql=0,
                )

                if en_err is not None:
                    en_err = f"{en_err:.6f}"
                else:
                    en_err = f"  {en_err}  "
                if olp_r_err is not None:
                    olp_r_err = f"{olp_r_err:.6f}"
                else:
                    olp_r_err = f"  {olp_r_err}  "
                if hf_orb_cr_err is not None:
                    hf_orb_cr_err = f"{hf_orb_cr_err:.6f}"
                else:
                    hf_orb_cr_err = f"  {hf_orb_cr_err}  "
                if cc_orb_cr0_err is not None:
                    cc_orb_cr0_err = f"{cc_orb_cr0_err:.6f}"
                else:
                    cc_orb_cr0_err = f"  {cc_orb_cr0_err}  "
                if cc_orb_cr1_err is not None:
                    cc_orb_cr1_err = f"{cc_orb_cr1_err:.6f}"
                else:
                    cc_orb_cr1_err = f"  {cc_orb_cr1_err}  "
                if cc_orb_cr2_err is not None:
                    cc_orb_cr2_err = f"{cc_orb_cr2_err:.6f}"
                else:
                    cc_orb_cr2_err = f"  {cc_orb_cr2_err}  "
                if cc_orb_cr_err is not None:
                    cc_orb_cr_err = f"{cc_orb_cr_err:.6f}"
                else:
                    cc_orb_cr_err = f"  {cc_orb_cr_err}  "
                if orb_en_err is not None:
                    orb_en_err = f"{orb_en_err:.6f}"
                else:
                    orb_en_err = f"  {orb_en_err}  "

                en = f"{en:.6f}"
                olp_r = f"{olp_r:.6f}"
                hf_orb_cr = f"{hf_orb_cr:.6f}"
                cc_orb_cr0 = f"{cc_orb_cr0:.6f}"
                cc_orb_cr1 = f"{cc_orb_cr1:.6f}"
                cc_orb_cr2 = f"{cc_orb_cr2:.6f}"
                cc_orb_cr = f"{cc_orb_cr:.6f}"
                orb_en = f"{orb_en:.6f}"

                print(f"  {n:4d}  {en}  {en_err}  {olp_r}  {olp_r_err}  {hf_orb_cr}  {hf_orb_cr_err}"
                      f"  {cc_orb_cr0}  {cc_orb_cr0_err}  {cc_orb_cr1}  {cc_orb_cr1_err}  {cc_orb_cr2}  {cc_orb_cr2_err}"
                      f"  {cc_orb_cr}  {cc_orb_cr_err}  {orb_en}  {orb_en_err}  {time.time() - init_time:.2f}")
        comm.Barrier()


comm.Barrier()
if rank == 0:
    assert glb_blk_wt is not None
    assert glb_blk_en is not None
    assert glb_blk_olp_r is not None
    assert glb_blk_hf_orb_cr is not None
    assert glb_blk_cc_orb_cr0 is not None
    assert glb_blk_cc_orb_cr1 is not None
    assert glb_blk_cc_orb_cr2 is not None
    assert glb_blk_cc_orb_cr is not None
    assert glb_blk_orb_en is not None

    samples_clean, idx = stat_utils.reject_outliers(
    np.stack(
                (
                    glb_blk_wt,
                    glb_blk_en,
                    glb_blk_olp_r,
                    glb_blk_hf_orb_cr,
                    glb_blk_cc_orb_cr0,
                    glb_blk_cc_orb_cr1,
                    glb_blk_cc_orb_cr2,
                    glb_blk_cc_orb_cr,
                    glb_blk_orb_en,
                )
            ).T,
            1,
        )

    print(
        f"# Number of outliers in post: {glb_blk_wt.size - samples_clean.shape[0]} "
        )
    
    glb_blk_wt = samples_clean[:, 0]
    glb_blk_en = samples_clean[:, 1]
    glb_blk_olp_r = samples_clean[:, 2]
    glb_blk_hf_orb_cr = samples_clean[:, 3]
    glb_blk_cc_orb_cr0 = samples_clean[:, 4]
    glb_blk_cc_orb_cr1 = samples_clean[:, 5]
    glb_blk_cc_orb_cr2 = samples_clean[:, 6]
    glb_blk_cc_orb_cr = samples_clean[:, 7]
    glb_blk_orb_en = samples_clean[:, 8]

    en, en_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_en[: (n + 1) * size],
                    neql=0,printQ=True
                )
    olp_r, olp_r_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_olp_r[: (n + 1) * size],
                    neql=0,printQ=True
                )
    hf_orb_cr, hf_orb_cr_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_hf_orb_cr[: (n + 1) * size],
                    neql=0,printQ=True
                )
    cc_orb_cr0, cc_orb_cr0_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_cc_orb_cr0[: (n + 1) * size],
                    neql=0,printQ=True
                )
    cc_orb_cr1, cc_orb_cr1_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_cc_orb_cr1[: (n + 1) * size],
                    neql=0,printQ=True
                )
    cc_orb_cr2, cc_orb_cr2_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_cc_orb_cr2[: (n + 1) * size],
                    neql=0,printQ=True
                )
    cc_orb_cr, cc_orb_cr_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_cc_orb_cr[: (n + 1) * size],
                    neql=0,printQ=True
                )
    orb_en, orb_en_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_orb_en[: (n + 1) * size],
                    neql=0,printQ=True
                )
    
    if en_err is not None:
        en_err = f"{en_err:.6f}"
    else:
        en_err = f"  {en_err}  "
    if olp_r_err is not None:
        olp_r_err = f"{olp_r_err:.6f}"
    else:
        olp_r_err = f"  {olp_r_err}  "
    if hf_orb_cr_err is not None:
        hf_orb_cr_err = f"{hf_orb_cr_err:.6f}"
    else:
        hf_orb_cr_err = f"  {hf_orb_cr_err}  "
    if cc_orb_cr0_err is not None:
        cc_orb_cr0_err = f"{cc_orb_cr0_err:.6f}"
    else:
        cc_orb_cr0_err = f"  {cc_orb_cr0_err}  "
    if cc_orb_cr1_err is not None:
        cc_orb_cr1_err = f"{cc_orb_cr1_err:.6f}"
    else:
        cc_orb_cr1_err = f"  {cc_orb_cr1_err}  "
    if cc_orb_cr2_err is not None:
        cc_orb_cr2_err = f"{cc_orb_cr2_err:.6f}"
    else:
        cc_orb_cr2_err = f"  {cc_orb_cr2_err}  "
    if cc_orb_cr_err is not None:
        cc_orb_cr_err = f"{cc_orb_cr_err:.6f}"
    else:
        cc_orb_cr_err = f"  {cc_orb_cr_err}  "
    if orb_en_err is not None:
        orb_en_err = f"{orb_en_err:.6f}"
    else:
        orb_en_err = f"  {orb_en_err}  "

    en = f"{en:.6f}"
    olp_r = f"{olp_r:.6f}"
    hf_orb_cr = f"{hf_orb_cr:.6f}"
    cc_orb_cr0 = f"{cc_orb_cr0:.6f}"
    cc_orb_cr1 = f"{cc_orb_cr1:.6f}"
    cc_orb_cr2 = f"{cc_orb_cr2:.6f}"
    cc_orb_cr = f"{cc_orb_cr:.6f}"
    orb_en = f"{orb_en:.6f}"

    print(f"Final Results:")
    print(f"lno-afqmc/cc energy: {en} +/- {en_err}")
    print(f"lno-afqmc/cc olp_r: {olp_r} +/- {olp_r_err}")
    print(f"lno-afqmc/cc hf_orb_cr: {hf_orb_cr} +/- {hf_orb_cr_err}")
    print(f"lno-afqmc/cc cc_orb_cr0: {cc_orb_cr0} +/- {cc_orb_cr0_err}")
    print(f"lno-afqmc/cc cc_orb_cr1: {cc_orb_cr1} +/- {cc_orb_cr1_err}")
    print(f"lno-afqmc/cc cc_orb_cr2: {cc_orb_cr2} +/- {cc_orb_cr2_err}")
    print(f"lno-afqmc/cc cc_orb_cr: {cc_orb_cr} +/- {cc_orb_cr_err}")
    print(f"lno-afqmc/cc orb_en: {orb_en} +/- {orb_en_err}")
    print(f"total run time: {time.time() - init_time:.2f}")

comm.Barrier()
