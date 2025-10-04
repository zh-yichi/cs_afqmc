from functools import partial
from jax import random
#from mpi4py import MPI
import numpy as np
from jax import numpy as jnp
from ad_afqmc import config, wavefunctions, stat_utils
from ad_afqmc.lno_afqmc import norhf_test, lnoafqmc_runner
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

## add fock_ia to ham_data
rot_h1,rot_chol = ham_data['rot_h1'],ham_data['rot_chol']
nocc = rot_h1.shape[0]
h_ia = rot_h1[:nocc,nocc:]
veff_ia = 2*jnp.einsum('gia,gjj->ia',rot_chol[:,:nocc,nocc:],rot_chol[:,:nocc,:nocc])\
          - jnp.einsum('gij,gja->ia',rot_chol[:,:nocc,:nocc],rot_chol[:,:nocc,nocc:])
f_ia = h_ia + veff_ia
ham_data['fock_ia'] = f_ia

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
which_rhf = options.get("which_rhf", 1)
comm.Barrier()
if rank == 0:
    orb_en,orb_fk,orb_cx = norhf_test._walker_norhf_orbenergy_test(
        prop_data["walkers"][0],ham_data,wave_data)

    e_init = prop_data["e_estimate"]

    print(f'# afqmc propagation with {options["n_walkers"]*size} walkers')
    print(f'# appraching equilibrium')
    print('# step  energy  orb_en  orb_fk  orb_cx  time')
    print(f"  {0:3d}"
          f"  {e_init:.6f}"
          f"  {orb_en:.6f}"
          f"  {orb_fk:.6f}"
          f"  {orb_cx:.6f}"
          f"  {time.time() - init_time:.2f} "
        )
comm.Barrier()

for n in range(options["n_eql"]):
    prop_data, (blk_wt,blk_en,blk_orb_en,blk_orb_fk,blk_orb_cx) \
                = norhf_test.propagate_phaseless_orb_norhf_test(
                    ham_data,prop,prop_data,trial,wave_data,sampler)
    
    blk_wt = np.array([blk_wt], dtype="float32")
    blk_en = np.array([blk_en], dtype="float32")
    blk_orb_en = np.array([blk_orb_en], dtype="float32")
    blk_orb_fk = np.array([blk_orb_fk], dtype="float32")
    blk_orb_cx = np.array([blk_orb_cx], dtype="float32")

    blk_wt_en = np.array(
        [blk_en * blk_wt], dtype="float32"
    )
    blk_wt_orb_en = np.array(
        [blk_orb_en * blk_wt], dtype="float32"
    )
    blk_wt_orb_fk = np.array(
        [blk_orb_fk * blk_wt], dtype="float32"
    )
    blk_wt_orb_cx = np.array(
        [blk_orb_cx * blk_wt], dtype="float32"
    )

    tot_wt = np.zeros(1, dtype="float32")
    tot_wt_en = np.zeros(1, dtype="float32")
    tot_wt_orb_en = np.zeros(1, dtype="float32")
    tot_wt_orb_fk = np.zeros(1, dtype="float32")
    tot_wt_orb_cx = np.zeros(1, dtype="float32")

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
            [blk_wt_orb_en, MPI.FLOAT],
            [tot_wt_orb_en, MPI.FLOAT],
            op=MPI.SUM,
            root=0,
        )
    comm.Reduce(
            [blk_wt_orb_fk, MPI.FLOAT],
            [tot_wt_orb_fk, MPI.FLOAT],
            op=MPI.SUM,
            root=0,
        )
    comm.Reduce(
            [blk_wt_orb_cx, MPI.FLOAT],
            [tot_wt_orb_cx, MPI.FLOAT],
            op=MPI.SUM,
            root=0,
        )

    comm.Barrier()
    if rank == 0:
        blk_wt = tot_wt
        blk_en = tot_wt_en / tot_wt
        blk_orb_en = tot_wt_orb_en / tot_wt
        blk_orb_fk = tot_wt_orb_fk / tot_wt
        blk_orb_cx = tot_wt_orb_cx / tot_wt

    comm.Bcast(blk_wt, root=0)
    comm.Bcast(blk_en, root=0)
    comm.Bcast(blk_orb_en, root=0)
    comm.Bcast(blk_orb_fk, root=0)
    comm.Bcast(blk_orb_cx, root=0)
    
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
            f"  {blk_orb_en[0]:.6f}"
            f"  {blk_orb_fk[0]:.6f}"
            f"  {blk_orb_cx[0]:.6f}"
            f"  {time.time() - init_time:.2f} "
        )
    comm.Barrier()

glb_blk_wt = None
glb_blk_en = None
glb_blk_orb_en = None
glb_blk_orb_fk = None
glb_blk_orb_cx = None

if rank == 0:
    glb_blk_wt = np.zeros(size * sampler.n_blocks)
    glb_blk_en = np.zeros(size * sampler.n_blocks)
    glb_blk_orb_en = np.zeros(size * sampler.n_blocks)
    glb_blk_orb_fk = np.zeros(size * sampler.n_blocks)
    glb_blk_orb_cx = np.zeros(size * sampler.n_blocks)
    
comm.Barrier()
if rank == 0:
    print("# Sampling sweeps:")
    print("# iter  energy  err  orb_energy  orb_fock  orb_colex  err  time")
comm.Barrier()

for n in range(sampler.n_blocks):
    prop_data,(blk_wt,blk_en,blk_orb_en,blk_orb_fk,blk_orb_cx) \
        = norhf_test.propagate_phaseless_orb_norhf_test(
            ham_data,prop,prop_data,trial,wave_data,sampler)

    blk_wt = np.array([blk_wt], dtype="float32")
    blk_en = np.array([blk_en], dtype="float32")
    blk_orb_en = np.array([blk_orb_en], dtype="float32")
    blk_orb_fk = np.array([blk_orb_fk], dtype="float32")
    blk_orb_cx = np.array([blk_orb_cx], dtype="float32")

    gather_wt = None
    gather_en = None
    gather_orb_en = None
    gather_orb_fk = None
    gather_orb_cx = None

    comm.Barrier()
    if rank == 0:
        gather_wt = np.zeros(size, dtype="float32")
        gather_en = np.zeros(size, dtype="float32")
        gather_orb_en = np.zeros(size, dtype="float32")
        gather_orb_fk = np.zeros(size, dtype="float32")
        gather_orb_cx = np.zeros(size, dtype="float32")
    comm.Barrier()

    comm.Gather(blk_wt, gather_wt, root=0)
    comm.Gather(blk_en, gather_en, root=0)
    comm.Gather(blk_orb_en, gather_orb_en, root=0)
    comm.Gather(blk_orb_fk, gather_orb_fk, root=0)
    comm.Gather(blk_orb_cx, gather_orb_cx, root=0)

    comm.Barrier()
    if rank == 0:
        glb_blk_wt[n * size : (n + 1) * size] = gather_wt
        glb_blk_en[n * size : (n + 1) * size] = gather_en
        glb_blk_orb_en[n * size : (n + 1) * size] = gather_orb_en
        glb_blk_orb_fk[n * size : (n + 1) * size] = gather_orb_fk
        glb_blk_orb_cx[n * size : (n + 1) * size] = gather_orb_cx
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
                orb_en, orb_en_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_orb_en[: (n + 1) * size],
                    neql=0,
                )
                orb_fk, orb_fk_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_orb_fk[: (n + 1) * size],
                    neql=0,
                )
                orb_cx, orb_cx_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_orb_cx[: (n + 1) * size],
                    neql=0,
                )

                if en_err is not None:
                    en_err = f"{en_err:.6f}"
                else:
                    en_err = f"  {en_err}  "
                if orb_en_err is not None:
                    orb_en_err = f"{orb_en_err:.6f}"
                else:
                    orb_en_err = f"  {orb_en_err}  "
                if orb_fk_err is not None:
                    orb_fk_err = f"{orb_fk_err:.6f}"
                else:
                    orb_fk_err = f"  {orb_fk_err}  "
                if orb_cx_err is not None:
                    orb_cx_err = f"{orb_cx_err:.6f}"
                else:
                    orb_cx_err = f"  {orb_cx_err}  "

                en = f"{en:.6f}"
                orb_en = f"{orb_en:.6f}"
                orb_fk = f"{orb_fk:.6f}"
                orb_cx = f"{orb_cx:.6f}"

                print(f"  {n:4d}  {en}  {en_err}  {orb_en}  {orb_en_err}"
                      f"  {orb_fk}  {orb_fk_err}  {orb_cx}  {orb_cx_err}"
                      f"  {time.time() - init_time:.2f}")
        comm.Barrier()


comm.Barrier()
if rank == 0:
    assert glb_blk_wt is not None
    assert glb_blk_en is not None
    assert glb_blk_orb_en is not None
    assert glb_blk_orb_fk is not None
    assert glb_blk_orb_cx is not None

    samples_clean, idx = stat_utils.reject_outliers(
    np.stack(
                (
                    glb_blk_wt,
                    glb_blk_en,
                    glb_blk_orb_en,
                    glb_blk_orb_fk,
                    glb_blk_orb_cx,
                )
            ).T,
            1,
        )

    print(
        f"# Number of outliers in post: {glb_blk_wt.size - samples_clean.shape[0]} "
        )
    
    glb_blk_wt = samples_clean[:, 0]
    glb_blk_en = samples_clean[:, 1]
    glb_blk_orb_en = samples_clean[:, 2]
    glb_blk_orb_fk = samples_clean[:, 3]
    glb_blk_orb_cx = samples_clean[:, 4]

    en, en_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_en[: (n + 1) * size],
                    neql=0,printQ=True
                )
    orb_en, orb_en_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_orb_en[: (n + 1) * size],
                    neql=0,printQ=True
                )
    orb_fk, orb_fk_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_orb_fk[: (n + 1) * size],
                    neql=0,printQ=True
                )
    orb_cx, orb_cx_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_orb_cx[: (n + 1) * size],
                    neql=0,printQ=True
                )
    
    if en_err is not None:
        en_err = f"{en_err:.6f}"
    else:
        en_err = f"  {en_err}  "
    if orb_en_err is not None:
        orb_en_err = f"{orb_en_err:.6f}"
    else:
        orb_en_err = f"  {orb_en_err}  "
    if orb_fk_err is not None:
        orb_fk_err = f"{orb_fk_err:.6f}"
    else:
        orb_fk_err = f"  {orb_fk_err}  "
    if orb_cx_err is not None:
        orb_cx_err = f"{orb_cx_err:.6f}"
    else:
        orb_cx_err = f"  {orb_cx_err}  "

    en = f"{en:.6f}"
    orb_en = f"{orb_en:.6f}"
    orb_fk = f"{orb_fk:.6f}"
    orb_cx = f"{orb_cx:.6f}"

    print(f"Final Results:")
    print('use norhf trial')
    print(f"lno-afqmc energy: {en} +/- {en_err}")
    print(f"lno-afqmc orbital energy: {orb_en} +/- {orb_en_err}")
    print(f"lno-afqmc orbital fock energy: {orb_fk} +/- {orb_fk_err}")
    print(f"lno-afqmc orbital colexc energy: {orb_cx} +/- {orb_cx_err}")
    print(f"total run time: {time.time() - init_time:.2f}")

comm.Barrier()
