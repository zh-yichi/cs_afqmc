from functools import partial
from jax import random
#from mpi4py import MPI
import numpy as np
from jax import numpy as jnp
from ad_afqmc import config, wavefunctions, stat_utils
from ad_afqmc.lno_ccsd import lno_ccsd
import time

print = partial(print, flush=True)

ham_data, ham, prop, trial, wave_data, sampler, observable, options, _ = (
    lno_ccsd.prep_lnoccsd_afqmc())

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
    hf_orb_cr_init,_ = lno_ccsd._frg_hf_cr(
        ham_data['rot_h1'], ham_data['rot_chol'],prop_data["walkers"][0],trial,wave_data)
    ccsd_orb_cr_init = lno_ccsd._frg_ccsd_cr(
        prop_data["walkers"][0],ham_data,wave_data,trial,1e-6)
    
    orb_cr_init = hf_orb_cr_init+ccsd_orb_cr_init
    e_init = prop_data["e_estimate"]

    print(f'# afqmc propagation with ccsd trial using {options["n_walkers"]*size} walkers')
    print(f'# appraching equilibrium of the Markov Chain')
    print('# step  energy  hf_orb_cr' \
          '  ccsd_orb_cr  orb_corr  time')
    print(f"  {0:3d}"
          f"  {e_init:.6f}"
          f"  {hf_orb_cr_init:.6f}"
          f"  {ccsd_orb_cr_init:.6f}"
          f"  {orb_cr_init:.6f}"
          f"  {time.time() - init_time:.2f} "
        )
comm.Barrier()

for n in range(options["n_eql"]):
    prop_data, (blk_energy,blk_wt,
               blk_hf_orb_cr,blk_ccsd_orb_cr,blk_orb_cr) \
                = lno_ccsd.propagate_phaseless_orb(
                    ham_data,prop,prop_data,trial,wave_data,sampler)
    
    blk_energy = np.array([blk_energy], dtype="float32")
    blk_wt = np.array([blk_wt], dtype="float32")
    blk_hf_orb_cr = np.array([blk_hf_orb_cr], dtype="float32")
    blk_ccsd_orb_cr = np.array([blk_ccsd_orb_cr], dtype="float32")
    blk_orb_cr = np.array([blk_orb_cr], dtype="float32")
    

    blk_wt_energy = np.array(
        [blk_energy * blk_wt], dtype="float32"
    )
    blk_wt_hf_orb_cr = np.array(
        [blk_hf_orb_cr * blk_wt], dtype="float32"
    )
    blk_wt_ccsd_orb_cr = np.array(
        [blk_ccsd_orb_cr * blk_wt], dtype="float32"
    )
    blk_wt_orb_cr = np.array(
        [blk_orb_cr * blk_wt], dtype="float32"
    )

    tot_wt_energy = np.zeros(1, dtype="float32")
    tot_wt = np.zeros(1, dtype="float32")
    tot_wt_hf_orb_cr = np.zeros(1, dtype="float32")
    tot_wt_ccsd_orb_cr = np.zeros(1, dtype="float32")
    tot_wt_orb_cr = np.zeros(1, dtype="float32")

    comm.Reduce(
            [blk_wt_energy, MPI.FLOAT],
            [tot_wt_energy, MPI.FLOAT],
            op=MPI.SUM,
            root=0,
        )
    comm.Reduce(
        [blk_wt, MPI.FLOAT],
        [tot_wt, MPI.FLOAT],
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
            [blk_wt_ccsd_orb_cr, MPI.FLOAT],
            [tot_wt_ccsd_orb_cr, MPI.FLOAT],
            op=MPI.SUM,
            root=0,
        )
    comm.Reduce(
            [blk_wt_orb_cr, MPI.FLOAT],
            [tot_wt_orb_cr, MPI.FLOAT],
            op=MPI.SUM,
            root=0,
        )

    comm.Barrier()
    if rank == 0:
        blk_energy = tot_wt_energy / tot_wt
        blk_wt = tot_wt
        blk_hf_orb_cr = tot_wt_hf_orb_cr / tot_wt
        blk_ccsd_orb_cr = tot_wt_ccsd_orb_cr / tot_wt
        blk_orb_cr = tot_wt_orb_cr / tot_wt

    comm.Bcast(blk_energy, root=0)
    comm.Bcast(blk_wt, root=0)
    comm.Bcast(blk_hf_orb_cr, root=0)
    comm.Bcast(blk_ccsd_orb_cr, root=0)
    comm.Bcast(blk_orb_cr, root=0)
    
    prop_data = propagator.orthonormalize_walkers(prop_data)
    prop_data = propagator.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = (
         0.9 * prop_data["e_estimate"] + 0.1 * blk_energy[0]
         )
    comm.Barrier()

    comm.Barrier()
    if rank == 0:
        print(
            f"  {n+1:3d}"
            f"  {blk_energy[0]:.6f}"
            f"  {blk_hf_orb_cr[0]:.6f}"
            f"  {blk_ccsd_orb_cr[0]:.6f}"
            f"  {blk_orb_cr[0]:.6f}"
            f"  {time.time() - init_time:.2f} "
        )
    comm.Barrier()


glb_blk_energy = None
glb_blk_wt = None
glb_blk_hf_orb_cr = None
glb_blk_ccsd_orb_cr = None
glb_blk_orb_cr = None

if rank == 0:
    glb_blk_energy = np.zeros(size * sampler.n_blocks)
    glb_blk_wt = np.zeros(size * sampler.n_blocks)
    glb_blk_hf_orb_cr = np.zeros(size * sampler.n_blocks)
    glb_blk_ccsd_orb_cr = np.zeros(size * sampler.n_blocks)
    glb_blk_orb_cr = np.zeros(size * sampler.n_blocks)


comm.Barrier()
# init_time = time.time()
if rank == 0:
    print("# Sampling sweeps:")
    print("# iter  energy  err  hf_orb_cr  err  "\
          "olp_ratio  ccsd_orb_cr  err  "\
            "tot_orb_cr  err  time")
comm.Barrier()

for n in range(sampler.n_blocks):
    prop_data,(blk_energy,blk_wt,
               blk_hf_orb_cr,blk_ccsd_orb_cr,blk_orb_cr) \
        = lno_ccsd.propagate_phaseless_orb(
            ham_data,prop,prop_data,trial,wave_data,sampler)

    blk_energy = np.array([blk_energy], dtype="float32")
    blk_wt = np.array([blk_wt], dtype="float32")
    blk_hf_orb_cr = np.array([blk_hf_orb_cr], dtype="float32")
    blk_ccsd_orb_cr = np.array([blk_ccsd_orb_cr], dtype="float32")
    blk_orb_cr = np.array([blk_orb_cr], dtype="float32")

    gather_energy = None
    gather_wt = None
    gather_hf_orb_cr = None
    gather_ccsd_orb_cr = None
    gather_orb_cr = None

    comm.Barrier()
    if rank == 0:
        gather_energy = np.zeros(size, dtype="float32")
        gather_wt = np.zeros(size, dtype="float32")
        gather_hf_orb_cr = np.zeros(size, dtype="float32")
        gather_ccsd_orb_cr = np.zeros(size, dtype="float32")
        gather_orb_cr = np.zeros(size, dtype="float32")
    comm.Barrier()

    comm.Gather(blk_energy, gather_energy, root=0)
    comm.Gather(blk_wt, gather_wt, root=0)
    comm.Gather(blk_hf_orb_cr, gather_hf_orb_cr, root=0)
    comm.Gather(blk_ccsd_orb_cr, gather_ccsd_orb_cr, root=0)
    comm.Gather(blk_orb_cr, gather_orb_cr, root=0)

    comm.Barrier()
    if rank == 0:
        glb_blk_energy[n * size : (n + 1) * size] = gather_energy
        glb_blk_wt[n * size : (n + 1) * size] = gather_wt
        glb_blk_hf_orb_cr[n * size : (n + 1) * size] = gather_hf_orb_cr
        glb_blk_ccsd_orb_cr[n * size : (n + 1) * size] = gather_ccsd_orb_cr
        glb_blk_orb_cr[n * size : (n + 1) * size] = gather_orb_cr

        assert gather_wt is not None

        blk_wt= np.sum(gather_wt)
        blk_energy = np.sum(gather_wt * gather_energy) / blk_wt
        # blk_hf_orb_cr = np.sum(gather_wt * gather_hf_orb_cr) / blk_wt
        # blk_ccsd_orb_cr = np.sum(gather_wt * gather_ccsd_orb_cr) / blk_wt
        # blk_orb_cr= np.sum(gather_wt * gather_orb_cr) / blk_wt

    comm.Barrier()
    
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
                hf_orb_cr, hf_orb_cr_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_hf_orb_cr[: (n + 1) * size],
                    neql=0,
                )
                ccsd_orb_cr, ccsd_orb_cr_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_ccsd_orb_cr[: (n + 1) * size],
                    neql=0,
                )
                orb_cr, orb_cr_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_orb_cr[: (n + 1) * size],
                    neql=0,
                )

                if energy_err is not None:
                    energy_err = f"{energy_err:.6f}"
                else:
                    energy_err = f"  {energy_err}  "
                if hf_orb_cr_err is not None:
                    hf_orb_cr_err = f"{hf_orb_cr_err:.6f}"
                else:
                    hf_orb_cr_err = f"  {hf_orb_cr_err}  "
                if ccsd_orb_cr_err is not None:
                    ccsd_orb_cr_err = f"{ccsd_orb_cr_err:.6f}"
                else:
                    ccsd_orb_cr_err = f"  {ccsd_orb_cr_err}  "
                if orb_cr_err is not None:
                    orb_cr_err = f"{orb_cr_err:.6f}"
                else:
                    orb_cr_err = f"  {orb_cr_err}  "

                energy = f"{energy:.6f}"
                hf_orb_cr = f"{hf_orb_cr:.6f}"
                ccsd_orb_cr = f"{ccsd_orb_cr:.6f}"
                orb_cr = f"{orb_cr:.6f}"

                print(f"  {n:4d}  {energy}  {energy_err}"
                      f"  {hf_orb_cr}  {hf_orb_cr_err}  {ccsd_orb_cr}  {ccsd_orb_cr_err}"
                      f"  {orb_cr}  {orb_cr_err}  {time.time() - init_time:.2f}")
        comm.Barrier()


comm.Barrier()
if rank == 0:
    assert glb_blk_energy is not None
    assert glb_blk_wt is not None
    assert glb_blk_hf_orb_cr is not None
    assert glb_blk_ccsd_orb_cr is not None
    assert glb_blk_orb_cr is not None


    samples_clean, idx = stat_utils.reject_outliers(
    np.stack(
                (
                    glb_blk_wt,
                    glb_blk_energy,
                    glb_blk_hf_orb_cr,
                    glb_blk_ccsd_orb_cr,
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
    glb_blk_hf_orb_cr = samples_clean[:, 2]
    glb_blk_ccsd_orb_cr = samples_clean[:, 3]
    glb_blk_orb_cr = samples_clean[:, 4]

    energy, energy_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_energy[: (n + 1) * size],
                    neql=0,printQ=True
                )
    hf_orb_cr, hf_orb_cr_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_hf_orb_cr[: (n + 1) * size],
                    neql=0,printQ=True
                )
    ccsd_orb_cr, ccsd_orb_cr_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_ccsd_orb_cr[: (n + 1) * size],
                    neql=0,printQ=True
                )
    orb_cr, orb_cr_err = stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_orb_cr[: (n + 1) * size],
                    neql=0,printQ=True
                )
    
    if energy_err is not None:
        energy_err = f"{energy_err:.6f}"
    else:
        energy_err = f"  {energy_err}  "
    if hf_orb_cr_err is not None:
        hf_orb_cr_err = f"{hf_orb_cr_err:.6f}"
    else:
        hf_orb_cr_err = f"  {hf_orb_cr_err}  "
    if ccsd_orb_cr_err is not None:
        ccsd_orb_cr_err = f"{ccsd_orb_cr_err:.6f}"
    else:
        ccsd_orb_cr_err = f"  {ccsd_orb_cr_err}  "
    if orb_cr_err is not None:
        orb_cr_err = f"{orb_cr_err:.6f}"
    else:
        orb_cr_err = f"  {orb_cr_err}  "

    energy = f"{energy:.6f}"
    hf_orb_cr = f"{hf_orb_cr:.6f}"
    ccsd_orb_cr = f"{ccsd_orb_cr:.6f}"
    orb_cr = f"{orb_cr:.6f}"

    print(f"Final Results:")
    print(f"lno-ccsd-afqmc energy: {energy} +/- {energy_err}")
    print(f"lno-ccsd-afqmc hf_orb_cr: {hf_orb_cr} +/- {hf_orb_cr_err}")
    print(f"lno-ccsd-afqmc ccsd_orb_cr: {ccsd_orb_cr} +/- {ccsd_orb_cr_err}")
    print(f"lno-ccsd-afqmc tot_orb_corr: {orb_cr} +/- {orb_cr_err}")
    print(f"total run time: {time.time() - init_time:.2f}")

comm.Barrier()
