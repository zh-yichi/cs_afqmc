from functools import partial
from jax import random
#from mpi4py import MPI
import numpy as np
from jax import numpy as jnp
from ad_afqmc import config, stat_utils
from ad_afqmc.lno_afqmc import sampling, ulno_afqmc
import time
import argparse

from ad_afqmc import config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    if args.use_gpu:
        config.afqmc_config["use_gpu"] = True

config.setup_jax()
MPI = config.setup_comm()
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print = partial(print, flush=True)

ham_data, prop, trial, wave_data, sampler, options, _ = (
    ulno_afqmc._prep_afqmc())

init_time = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

### initialize propagation
seed = options["seed"]
init_walkers = None
trial_rdm1 = trial.get_rdm1(wave_data)
if "rdm1" not in wave_data:
    wave_data["rdm1"] = trial_rdm1
ham_data = trial._build_measurement_intermediates(ham_data, wave_data)
ham_data = prop._build_propagation_intermediates(ham_data, trial, wave_data)

prop_data = prop.init_prop_data(trial, wave_data, ham_data, init_walkers)
if jnp.abs(jnp.sum(prop_data["overlaps"])) < 1.0e-6:
    raise ValueError(
        "Initial overlaps are zero. Pass walkers with non-zero overlap."
    )
prop_data["key"] = random.PRNGKey(seed + rank)
prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
prop_data["n_killed_walkers"] = 0
prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]

eorb0, eorb012, torb12, ecorr \
    = trial.calc_eorb_pt(prop_data['walkers'], ham_data, wave_data)
eorb0 = jnp.array(jnp.sum(eorb0)/ prop.n_walkers)
eorb012 = jnp.array(jnp.sum(eorb012)/ prop.n_walkers)
torb12 = jnp.array(jnp.sum(torb12)/ prop.n_walkers)
ecorr = jnp.array(jnp.sum(ecorr)/ prop.n_walkers)

comm.Barrier()
if rank == 0:
    print('# \n')
    e_init = prop_data["e_estimate"]
    print(f'# Propagating with {options["n_walkers"]*size} walkers')
    print(f"# Initial energy {e_init:.6f}")
    print("# Equilibration sweeps:")
    print("#   Iter \t <e0> \t \t <e0>_orb \t <e012>_orb \t <t12>_orb \t eorb \t time")
    print(f"  {0:5d} \t {ecorr:.6f} \t {eorb0:.6f} \t {eorb012:.6f} \t" 
          f"  {torb12:.6f} \t {time.time() - init_time:.2f}")
comm.Barrier()

sampler_eq = sampling.sampler_pt(
    n_prop_steps=50, n_ene_blocks=5, n_sr_blocks=10, n_chol = sampler.n_chol)

for n in range(1,options["n_eql"]+1):
    prop_data, (blk_wt, blk_ecorr, blk_eorb0, blk_eorb012, blk_torb12) = \
        sampler_eq.propagate_phaseless(ham_data, prop, prop_data, trial, wave_data)

    blk_wt = np.array([blk_wt], dtype="float64")
    blk_ecorr = np.array([blk_ecorr], dtype="float64")
    blk_eorb0 = np.array([blk_eorb0], dtype="float64")
    blk_eorb012 = np.array([blk_eorb012], dtype="float64")
    blk_torb12 = np.array([blk_torb12], dtype="float64")

    gather_wt = None
    gather_ecorr = None
    gather_eorb0 = None
    gather_eorb012 = None
    gather_torb12 = None

    comm.Barrier()
    if rank == 0:
        gather_wt = np.zeros(size, dtype="float64")
        gather_ecorr = np.zeros(size, dtype="float64")
        gather_eorb0 = np.zeros(size, dtype="float64")
        gather_eorb012 = np.zeros(size, dtype="float64")
        gather_torb12 = np.zeros(size, dtype="float64")
    comm.Barrier()

    comm.Gather(blk_wt, gather_wt, root=0)
    comm.Gather(blk_ecorr, gather_ecorr, root=0)
    comm.Gather(blk_eorb0, gather_eorb0, root=0)
    comm.Gather(blk_eorb012, gather_eorb012, root=0)
    comm.Gather(blk_torb12, gather_torb12, root=0)

    comm.Barrier()
    if rank == 0:
        assert gather_wt is not None
        blk_wt= np.sum(gather_wt)
        blk_ecorr = np.sum(gather_wt * gather_ecorr) / blk_wt
        blk_eorb0 = np.sum(gather_wt * gather_eorb0) / blk_wt
        blk_eorb012 = np.sum(gather_wt * gather_eorb012) / blk_wt
        blk_torb12 = np.sum(gather_wt * gather_torb12) / blk_wt
    comm.Barrier()

    comm.Bcast(blk_wt, root=0)
    comm.Bcast(blk_ecorr, root=0)
    comm.Bcast(blk_eorb0, root=0)
    comm.Bcast(blk_eorb012, root=0)
    comm.Bcast(blk_torb12, root=0)

    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = (0.9 * prop_data["e_estimate"] 
                               + 0.1 * (blk_ecorr + ham_data['E0']))
    eorb = blk_eorb012 - blk_torb12 * blk_ecorr

    comm.Barrier()
    if rank == 0:
        print(f"  {n:5d} \t {blk_ecorr:.6f} \t {blk_eorb0:.6f} \t"
              f"  {blk_eorb012:.6f} \t {blk_torb12:.6f} \t {eorb:.6f} \t"
              f"  {time.time() - init_time:.2f} "
        )
    comm.Barrier()

comm.Barrier()
if rank == 0:
    print("#\n# Sampling sweeps:")
    print("#  Iter   <e0>   error   <e0>_orb    error    <e012>_orb  error   <t12>_orb  error  <eorb>  error  time")
comm.Barrier()

glb_blk_wt = None
glb_blk_ecorr = None
glb_blk_eorb0 = None
glb_blk_eorb012 = None
glb_blk_torb12 = None

comm.Barrier()
if rank == 0:
    glb_blk_wt = np.zeros(size * sampler.n_blocks,dtype="float64")
    glb_blk_ecorr = np.zeros(size * sampler.n_blocks,dtype="float64")
    glb_blk_eorb0 = np.zeros(size * sampler.n_blocks,dtype="float64")
    glb_blk_eorb012 = np.zeros(size * sampler.n_blocks,dtype="float64")
    glb_blk_torb12 = np.zeros(size * sampler.n_blocks,dtype="float64")
comm.Barrier()

eorb_err = np.array([options["max_error"] + 1e-3])
for n in range(sampler.n_blocks):
    prop_data, (blk_wt, blk_ecorr, blk_eorb0, blk_eorb012, blk_torb12) = \
        sampler.propagate_phaseless(ham_data, prop, prop_data, trial, wave_data)
    
    blk_wt = np.array([blk_wt], dtype="float64")
    blk_ecorr = np.array([blk_ecorr], dtype="float64")
    blk_eorb0 = np.array([blk_eorb0], dtype="float64")
    blk_eorb012 = np.array([blk_eorb012], dtype="float64")
    blk_torb12 = np.array([blk_torb12], dtype="float64")

    gather_wt = None
    gather_ecorr = None
    gather_eorb0 = None
    gather_eorb012 = None
    gather_torb12 = None

    comm.Barrier()
    if rank == 0:
        gather_wt = np.zeros(size, dtype="float64")
        gather_ecorr = np.zeros(size, dtype="float64")
        gather_eorb0 = np.zeros(size, dtype="float64") 
        gather_eorb012 = np.zeros(size, dtype="float64") 
        gather_torb12 = np.zeros(size, dtype="float64")
    comm.Barrier()

    comm.Gather(blk_wt, gather_wt, root=0)
    comm.Gather(blk_ecorr, gather_ecorr, root=0)
    comm.Gather(blk_eorb0, gather_eorb0, root=0)
    comm.Gather(blk_eorb012, gather_eorb012, root=0)
    comm.Gather(blk_torb12, gather_torb12, root=0)

    comm.Barrier()
    if rank == 0:
        glb_blk_wt[n * size : (n + 1) * size] = gather_wt
        glb_blk_ecorr[n * size : (n + 1) * size] = gather_ecorr
        glb_blk_eorb0[n * size : (n + 1) * size] = gather_eorb0
        glb_blk_eorb012[n * size : (n + 1) * size] = gather_eorb012
        glb_blk_torb12[n * size : (n + 1) * size] = gather_torb12

        assert gather_wt is not None

        blk_wt= np.sum(gather_wt)
        blk_ecorr = np.sum(gather_wt * gather_ecorr) / blk_wt
        blk_eorb0 = np.sum(gather_wt * gather_eorb0) / blk_wt
        blk_eorb012 = np.sum(gather_wt * gather_eorb012) / blk_wt
        blk_torb12 = np.sum(gather_wt * gather_torb12) / blk_wt
    comm.Barrier()

    comm.Bcast(blk_wt, root=0)
    comm.Bcast(blk_ecorr, root=0)
    comm.Bcast(blk_eorb0, root=0)
    comm.Bcast(blk_eorb012, root=0)
    comm.Bcast(blk_torb12, root=0)

    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = (0.9 * prop_data["e_estimate"]
                               + 0.1 * (blk_ecorr + ham_data['E0'])) 
    
    if n % (max(sampler.n_blocks // 10, 1)) == 0 and n > 0:
        comm.Barrier()
        if rank == 0:
            # ecorr, ecorr_err = \
            #     stat_utils.blocking_analysis(
            #         glb_blk_wt[: (n + 1) * size],
            #         glb_blk_ecorr[: (n + 1) * size],
            #         neql=0,
            #     )
            # eorb0, eorb0_err = \
            #     stat_utils.blocking_analysis(
            #         glb_blk_wt[: (n + 1) * size],
            #         glb_blk_eorb0[: (n + 1) * size],
            #         neql=0,
            #     )
            # eorb012, eorb012_err = \
            #     stat_utils.blocking_analysis(
            #         glb_blk_wt[: (n + 1) * size],
            #         glb_blk_eorb012[: (n + 1) * size],
            #         neql=0,
            #     )
            # torb12, torb12_err = \
            #     stat_utils.blocking_analysis(
            #         glb_blk_wt[: (n + 1) * size],
            #         glb_blk_torb12[: (n + 1) * size],
            #         neql=0,
            #     )
            
            # if ecorr_err is not None:
            #     ecorr_err = f"{ecorr_err:.6f}"
            # else:
            #     ecorr_err = f"  {ecorr_err}  "
            
            # if eorb0_err is not None:
            #     eorb0_err = f"{eorb0_err:.6f}"
            # else:
            #     eorb0_err = f"  {eorb0_err}  "

            # if eorb012_err is not None:
            #     eorb012_err = f"{eorb012_err:.6f}"
            # else:
            #     eorb012_err = f"  {eorb012_err}  "
            
            # if torb12_err is not None:
            #     torb12_err = f"{torb12_err:.6f}"
            # else:
            #     torb12_err = f"  {torb12_err}  "

            glb_wt = np.sum(glb_blk_wt[:(n+1)*size])
            rho_ecorr = (glb_blk_wt[:(n+1)*size] * glb_blk_ecorr[:(n+1)*size])/glb_wt
            rho_eorb0 = (glb_blk_wt[:(n+1)*size] * glb_blk_eorb0[:(n+1)*size])/glb_wt
            rho_eorb012 = (glb_blk_wt[:(n+1)*size] * glb_blk_eorb012[:(n+1)*size])/glb_wt
            rho_torb12 = (glb_blk_wt[:(n+1)*size] * glb_blk_torb12[:(n+1)*size])/glb_wt

            ecorr = np.sum(rho_ecorr)
            eorb0 = np.sum(rho_eorb0)
            eorb012 = np.sum(rho_eorb012)
            torb12 = np.sum(rho_torb12)

            ecorr_err = np.std(rho_ecorr)
            eorb0_err = np.std(rho_eorb0)
            eorb012_err = np.std(rho_eorb012)
            torb12_err = np.std(rho_torb12)

            eorb = eorb012 - torb12 * ecorr
            dE = np.array([1,-ecorr,-torb12])
            cov_ete = np.cov([glb_blk_eorb012[:(n+1)*size],
                              glb_blk_torb12[:(n+1)*size],
                              glb_blk_ecorr[:(n+1)*size]])
            eorb_err[0] = np.sqrt(dE @ cov_ete @ dE)/np.sqrt((n+1)*size)
            
            # ecorr = f"{ecorr:.6f}"
            # eorb0 = f"{eorb0:.6f}"
            # eorb012 = f"{eorb012:.6f}"
            # torb12 = f"{torb12:.6f}"
            
            print(f"  {n:4d}  {ecorr:.6f}  {ecorr_err:.6f}"
                  f"  {eorb0:.6f}  {eorb0_err:.6f}"
                  f"  {eorb012:.6f}  {eorb012_err:.6f}"
                  f"  {torb12:.6f}  {torb12_err:.6f}"
                  f"  {eorb:.6f}  {eorb_err[0]:.6f}"
                  f"  {time.time() - init_time:.2f}")
        comm.Barrier()
        
        comm.Bcast(eorb_err, root=0)
        if eorb_err[0] < options["max_error"] and n > 2:
            break

comm.Barrier()
if rank == 0:
    assert glb_blk_wt is not None
    samples_clean, idx = stat_utils.reject_outliers(
        np.stack((
                glb_blk_wt[:(n+1)*size],
                glb_blk_ecorr[:(n+1)*size],
                glb_blk_eorb0[:(n+1)*size],
                glb_blk_eorb012[:(n+1)*size],
                glb_blk_torb12[:(n+1)*size],
                )).T,
                1,
            )

    print(
        f"# Number of outliers in post: {glb_blk_wt[:(n+1)*size].size - samples_clean.shape[0]} "
        )
    
    glb_blk_wt = samples_clean[:, 0]
    glb_blk_ecorr = samples_clean[:, 1]
    glb_blk_eorb0 = samples_clean[:, 2]
    glb_blk_eorb012 = samples_clean[:, 3]
    glb_blk_torb12 = samples_clean[:, 4]

    # ecorr, ecorr_err = stat_utils.blocking_analysis(glb_blk_wt,glb_blk_ecorr,neql=0,printQ=True)
    # eorb0, eorb0_err = stat_utils.blocking_analysis(glb_blk_wt,glb_blk_eorb0,neql=0,printQ=True)
    # eorb012, eorb012_err = stat_utils.blocking_analysis(glb_blk_wt,glb_blk_eorb012,neql=0,printQ=True)
    # torb12, torb12_err = stat_utils.blocking_analysis(glb_blk_wt,glb_blk_torb12,neql=0,printQ=True)

    # if ecorr_err is not None:
    #     ecorr_err = f"{ecorr_err:.6f}"
    # else:
    #     ecorr_err = f"  {ecorr_err}  "

    # if eorb0_err is not None:
    #     eorb0_err = f"{eorb0_err:.6f}"
    # else:
    #     eorb0_err = f"  {eorb0_err}  "

    # if eorb012_err is not None:
    #     eorb012_err = f"{eorb012_err:.6f}"
    # else:
    #     eorb012_err = f"  {eorb012_err}  "
    
    # if torb12_err is not None:
    #     torb12_err = f"{torb12_err:.6f}"
    # else:
    #     torb12_err = f"  {torb12_err}  "

    # eorb = eorb012 - torb12*ecorr
    
    # ehf = f"{ecorr + ham_data['E0']:.6f}"
    # ecorr = f"{ecorr:.6f}"
    # eorb012 = f"{eorb012:.6f}"
    # torb12 = f"{torb12:.6f}"

    glb_wt = np.sum(glb_blk_wt)
    rho_ecorr = (glb_blk_wt * glb_blk_ecorr)/glb_wt
    rho_eorb0 = (glb_blk_wt * glb_blk_eorb0)/glb_wt
    rho_eorb012 = (glb_blk_wt * glb_blk_eorb012)/glb_wt
    rho_torb12 = (glb_blk_wt * glb_blk_torb12)/glb_wt

    ecorr = np.sum(rho_ecorr)
    eorb0 = np.sum(rho_eorb0)
    eorb012 = np.sum(rho_eorb012)
    torb12 = np.sum(rho_torb12)

    ecorr_err = np.std(rho_ecorr)
    eorb0_err = np.std(rho_eorb0)
    eorb012_err = np.std(rho_eorb012)
    torb12_err = np.std(rho_torb12)

    eorb = eorb012 - torb12 * ecorr
    dE = np.array([1,-ecorr,-torb12])
    cov_ete = np.cov([glb_blk_eorb012,glb_blk_torb12,glb_blk_ecorr])
    eorb_err = np.sqrt(dE @ cov_ete @ dE)/np.sqrt(samples_clean.shape[0])
    
    print(f"# Final Results")
    print(f"# AFQMC/HF E_corr: {ecorr:.6f} +/- {ecorr_err:.6f}")
    print(f"# AFQMC/HF E_Orbital: {eorb0:.6f} +/- {eorb0_err:.6f}")
    print(f"# AFQMC/CCSD_PT E012_Orbital: {eorb012:.6f} +/- {eorb012_err:.6f}")
    print(f"# AFQMC/CCSD_PT T12_Orbital: {torb12:.6f} +/- {torb12_err:.6f}")
    print(f"# AFQMC/CCSD_PT E_Orbital: {eorb:.6f} +/- {eorb_err:.6f}")
    print(f"# total run time: {time.time() - init_time:.2f}")

comm.Barrier()
