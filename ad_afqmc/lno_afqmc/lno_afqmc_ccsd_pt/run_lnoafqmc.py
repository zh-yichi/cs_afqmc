import time
import argparse
import numpy as np
from jax import numpy as jnp
from jax import random
from functools import partial
from ad_afqmc import config, stat_utils
from ad_afqmc.lno_afqmc import sampling, ulno_afqmc

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

eorb, teorb, torb, ecorr \
    = trial.calc_eorb_pt(prop_data['walkers'], ham_data, wave_data)
eorb = jnp.array(jnp.sum(eorb)/ prop.n_walkers)
teorb = jnp.array(jnp.sum(teorb)/ prop.n_walkers)
torb = jnp.array(jnp.sum(torb)/ prop.n_walkers)
ecorr = jnp.array(jnp.sum(ecorr)/ prop.n_walkers)
eorb_pt = eorb + teorb - torb * ecorr

comm.Barrier()
if rank == 0:
    print('# \n')
    e_init = prop_data["e_estimate"]
    print(f'# Propagating with {options["n_walkers"]*size} walkers')
    print(f"# Initial energy {e_init:.6f}")
    print("# Equilibration sweeps:")
    print("#   Iter \t <H-E0> \t <H-E0>_orb \t <TH>_orb \t <T>_orb \t Ept_orb \t time")
    print(f"  {0:5d} \t {ecorr:.6f} \t {eorb:.6f} \t {teorb:.6f} \t" 
          f"  {torb:.6f} \t {eorb_pt:.6f} \t {time.time() - init_time:.2f}")
comm.Barrier()

sampler_eq = sampling.sampler_pt(
    n_prop_steps=50, n_ene_blocks=5, n_sr_blocks=10, n_chol = sampler.n_chol)

for n in range(1,options["n_eql"]+1):
    prop_data, (wt, eorb, teorb, torb, ecorr) = \
        sampler_eq.propagate_phaseless(ham_data, prop, prop_data, trial, wave_data)

    wt = np.array([wt], dtype="float64")
    eorb = np.array([eorb], dtype="float64")
    teorb = np.array([teorb], dtype="float64")
    torb = np.array([torb], dtype="float64")
    ecorr = np.array([ecorr], dtype="float64")

    gather_wt = None
    gather_eorb = None
    gather_teorb = None
    gather_torb = None
    gather_ecorr = None

    comm.Barrier()
    if rank == 0:
        gather_wt = np.zeros(size, dtype="float64")
        gather_eorb = np.zeros(size, dtype="float64")
        gather_teorb = np.zeros(size, dtype="float64")
        gather_torb = np.zeros(size, dtype="float64")
        gather_ecorr = np.zeros(size, dtype="float64")
    comm.Barrier()

    comm.Gather(wt, gather_wt, root=0)
    comm.Gather(eorb, gather_eorb, root=0)
    comm.Gather(teorb, gather_teorb, root=0)
    comm.Gather(torb, gather_torb, root=0)
    comm.Gather(ecorr, gather_ecorr, root=0)

    comm.Barrier()
    if rank == 0:
        assert gather_wt is not None
        wt= np.sum(gather_wt)
        eorb = np.sum(gather_wt * gather_eorb) / wt
        teorb = np.sum(gather_wt * gather_teorb) / wt
        torb = np.sum(gather_wt * gather_torb) / wt
        ecorr = np.sum(gather_wt * gather_ecorr) / wt
    comm.Barrier()

    comm.Bcast(wt, root=0)
    comm.Bcast(eorb, root=0)
    comm.Bcast(teorb, root=0)
    comm.Bcast(torb, root=0)
    comm.Bcast(ecorr, root=0)

    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = \
          0.9 * prop_data["e_estimate"] + 0.1 * (ham_data['E0']+ ecorr)
    eorb_pt = eorb + teorb - torb * ecorr

    comm.Barrier()
    if rank == 0:
        # print(type(ecorr), type(eorb), type(teorb), type(torb), type(eorb_pt))
        print(f"  {n:5d} \t {ecorr:.6f} \t {eorb:.6f} \t"
              f"  {teorb:.6f} \t {torb:.6f} \t {eorb_pt:.6f} \t"
              f"  {time.time() - init_time:.2f} ")
    comm.Barrier()

comm.Barrier()
if rank == 0:
    print("#\n Sampling sweeps:")
    print("#  Iter   <H-E0>   error   <H-E0>_orb    error "
          "   <T(H-E0)>_orb   error   <T>_orb   error "
          "   Ept_orb   error   time ")
comm.Barrier()

glb_wt = None
glb_eorb = None
glb_teorb = None
glb_torb = None
glb_ecorr = None

comm.Barrier()
if rank == 0:
    glb_wt = np.zeros(size * sampler.n_blocks,dtype="float64")
    glb_eorb = np.zeros(size * sampler.n_blocks,dtype="float64")
    glb_teorb = np.zeros(size * sampler.n_blocks,dtype="float64")
    glb_torb = np.zeros(size * sampler.n_blocks,dtype="float64")
    glb_ecorr = np.zeros(size * sampler.n_blocks,dtype="float64")
comm.Barrier()

eorb_pt_err = np.array([options["max_error"] + 1e-3])

for n in range(sampler.n_blocks):
    prop_data, (wt, eorb, teorb, torb, ecorr) = \
        sampler.propagate_phaseless(ham_data, prop, prop_data, trial, wave_data)
    
    wt = np.array([wt], dtype="float64")
    eorb = np.array([eorb], dtype="float64")
    teorb = np.array([teorb], dtype="float64")
    torb = np.array([torb], dtype="float64")
    ecorr = np.array([ecorr], dtype="float64")

    gather_wt = None
    gather_eorb = None
    gather_teorb = None
    gather_torb = None
    gather_ecorr = None

    comm.Barrier()
    if rank == 0:
        gather_wt = np.zeros(size, dtype="float64")
        gather_eorb = np.zeros(size, dtype="float64")
        gather_teorb = np.zeros(size, dtype="float64")
        gather_torb = np.zeros(size, dtype="float64")
        gather_ecorr = np.zeros(size, dtype="float64")
    comm.Barrier()

    comm.Gather(wt, gather_wt, root=0)
    comm.Gather(eorb, gather_eorb, root=0)
    comm.Gather(teorb, gather_teorb, root=0)
    comm.Gather(torb, gather_torb, root=0)
    comm.Gather(ecorr, gather_ecorr, root=0)

    comm.Barrier()
    if rank == 0:
        glb_wt[n * size : (n + 1) * size] = gather_wt
        glb_eorb[n * size : (n + 1) * size] = gather_eorb
        glb_teorb[n * size : (n + 1) * size] = gather_teorb
        glb_torb[n * size : (n + 1) * size] = gather_torb
        glb_ecorr[n * size : (n + 1) * size] = gather_ecorr

        assert gather_wt is not None

        wt = np.sum(gather_wt)
        eorb = np.sum(gather_wt * gather_eorb) / wt
        teorb = np.sum(gather_wt * gather_teorb) / wt
        torb = np.sum(gather_wt * gather_torb) / wt
        ecorr = np.sum(gather_wt * gather_ecorr) / wt
    comm.Barrier()

    comm.Bcast(wt, root=0)
    comm.Bcast(eorb, root=0)
    comm.Bcast(teorb, root=0)
    comm.Bcast(torb, root=0)
    comm.Bcast(ecorr, root=0)

    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = \
        0.9 * prop_data["e_estimate"] + 0.1 * (ham_data['E0'] + ecorr)
    
    if n % (max(sampler.n_blocks // 10, 1)) == 0 and n > 0:
        comm.Barrier()
        if rank == 0:

            wt = np.sum(glb_wt[:(n+1)*size])
            eorb = np.sum(glb_wt[:(n+1)*size] * glb_eorb[:(n+1)*size]) / wt
            teorb = np.sum(glb_wt[:(n+1)*size] * glb_teorb[:(n+1)*size]) / wt
            torb = np.sum(glb_wt[:(n+1)*size] * glb_torb[:(n+1)*size]) / wt
            ecorr = np.sum(glb_wt[:(n+1)*size] * glb_ecorr[:(n+1)*size]) / wt

            eorb_err = np.sqrt(np.sum(glb_wt[:(n+1)*size] * 
                        (glb_eorb[:(n+1)*size] - eorb)**2) / wt / (n+1)*size)
            teorb_err = np.sqrt(np.sum(glb_wt[:(n+1)*size] * 
                         (glb_teorb[:(n+1)*size] - teorb)**2) / wt / (n+1)*size)
            torb_err = np.sqrt(np.sum(glb_wt[:(n+1)*size] * 
                        (glb_torb[:(n+1)*size] - torb)**2) / wt / (n+1)*size)
            ecorr_err =np.sqrt( np.sum(glb_wt[:(n+1)*size] * 
                         (glb_ecorr[:(n+1)*size] - ecorr)**2) / wt / (n+1)*size)

            # eorb_err = np.std(rho_eorb)
            # teorb_err = np.std(rho_teorb)
            # torb_err = np.std(rho_torb)
            # ecorr_err = np.std(rho_ecorr)

            eorb_pt = eorb + teorb - torb * ecorr
            # (p_ept/p_eorb, p_ept/p_teorb, p_ept/p_torb, p_ept/p_ecorr)
            dE = np.array([1,1,-ecorr,-torb])
            cov = np.cov([glb_eorb[:(n+1)*size],
                          glb_teorb[:(n+1)*size],
                          glb_torb[:(n+1)*size],
                          glb_ecorr[:(n+1)*size]])
            eorb_pt_err[0] = np.sqrt(dE @ cov @ dE)/np.sqrt((n+1)*size)
            
            print(f"  {n:4d}  {ecorr:.6f}  {ecorr_err:.6f}"
                  f"  {eorb:.6f}  {eorb_err:.6f}"
                  f"  {teorb:.6f}  {teorb_err:.6f}"
                  f"  {torb:.6f}  {torb_err:.6f}"
                  f"  {eorb_pt:.6f}  {eorb_pt_err[0]:.6f}"
                  f"  {time.time() - init_time:.2f}")
        comm.Barrier()
        
        comm.Bcast(eorb_pt_err, root=0)
        if eorb_pt_err[0] < options["max_error"] and n > 5:
            break

comm.Barrier()
if rank == 0:
    assert glb_wt is not None
    samples_clean, idx = stat_utils.reject_outliers(
        np.stack((
                glb_wt[:(n+1)*size],
                glb_eorb[:(n+1)*size],
                glb_teorb[:(n+1)*size],
                glb_torb[:(n+1)*size],
                glb_ecorr[:(n+1)*size],
                )).T,
                4,
            )
    nsamples = samples_clean.shape[0]
    print(
        f"# Number of outliers in post: {glb_wt[:(n+1)*size].size - nsamples} "
        )
    
    glb_wt = samples_clean[:, 0]
    glb_eorb = samples_clean[:, 1]
    glb_teorb = samples_clean[:, 2]
    glb_torb = samples_clean[:, 3]
    glb_ecorr = samples_clean[:, 4]

    # ecorr, ecorr_err = stat_utils.blocking_analysis(glb_blk_wt,glb_blk_ecorr,neql=0,printQ=True)
    # eorb0, eorb0_err = stat_utils.blocking_analysis(glb_blk_wt,glb_blk_eorb0,neql=0,printQ=True)
    # eorb012, eorb012_err = stat_utils.blocking_analysis(glb_blk_wt,glb_blk_eorb012,neql=0,printQ=True)
    # torb12, torb12_err = stat_utils.blocking_analysis(glb_blk_wt,glb_blk_torb12,neql=0,printQ=True)

    wt = np.sum(glb_wt)
    eorb = np.sum(glb_wt * glb_eorb) / wt
    teorb = np.sum(glb_wt * glb_teorb) / wt
    torb = np.sum(glb_wt * glb_torb) / wt
    ecorr = np.sum(glb_wt * glb_ecorr) / wt

    eorb_err = np.sqrt(np.sum(glb_wt * (glb_eorb - eorb)**2) / wt / nsamples)
    teorb_err = np.sqrt(np.sum(glb_wt * (glb_teorb - teorb)**2) / wt / nsamples)
    torb_err = np.sqrt(np.sum(glb_wt * (glb_torb - torb)**2) / wt / nsamples)
    ecorr_err = np.sqrt(np.sum(glb_wt * (glb_ecorr - ecorr)**2) / wt / nsamples)

    eorb = eorb + teorb - torb * ecorr
    dE = np.array([1,1,-ecorr,-torb])
    cov = np.cov([glb_eorb,glb_teorb,glb_torb,glb_ecorr])
    eorb_pt_err = np.sqrt(dE @ cov @ dE) / np.sqrt(nsamples)
    
    print(f"# Final Results")
    print(f"# AFQMC/HF Correlation Energy: {ecorr:.6f} +/- {ecorr_err:.6f}")
    print(f"# AFQMC/HF Orbital <H-E0>: {eorb:.6f} +/- {eorb_err:.6f}")
    print(f"# AFQMC/CCSD_PT Orbital <T(H-E0)>: {teorb:.6f} +/- {teorb_err:.6f}")
    print(f"# AFQMC/CCSD_PT Orbital <T>: {torb:.6f} +/- {torb_err:.6f}")
    print(f"# AFQMC/CCSD_PT Orbital Ept: {eorb_pt:.6f} +/- {eorb_pt_err:.6f}")
    print(f"# total run time: {time.time() - init_time:.2f}")

comm.Barrier()
