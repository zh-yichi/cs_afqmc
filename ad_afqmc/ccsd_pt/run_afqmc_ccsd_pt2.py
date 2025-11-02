from functools import partial
from jax import random
#from mpi4py import MPI
import numpy as np
from jax import numpy as jnp
from ad_afqmc import config, sampling, stat_utils, mpi_jax
from ad_afqmc.ccsd_pt import sample_ccsd_pt2
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

ham_data, ham, prop, trial, wave_data, sampler, observable, options, _ = (
    mpi_jax._prep_afqmc())

# sampler.n_chol = ham_data["chol"].shape[0]

init_time = time.time()
### initialize propagation
seed = options["seed"]
init_walkers = None
trial_rdm1 = trial.get_rdm1(wave_data)
if "rdm1" not in wave_data:
    wave_data["rdm1"] = trial_rdm1
ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
ham_data = ham.build_propagation_intermediates(ham_data, prop, trial, wave_data)
h0 = ham_data['h0']

prop_data = prop.init_prop_data(trial, wave_data, ham_data, init_walkers)
if jnp.abs(jnp.sum(prop_data["overlaps"])) < 1.0e-6:
    raise ValueError(
        "Initial overlaps are zero. Pass walkers with non-zero overlap."
    )

prop_data["key"] = random.PRNGKey(seed + rank)
prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
prop_data["n_killed_walkers"] = 0
prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]

t1, t2, e0, e1 = trial.calc_energy_pt(prop_data["walkers"], ham_data, wave_data)
ept_sp = h0 + e0/t1 + e1/t1 - t2 * e0 / t1**2 
ept = jnp.array(jnp.sum(ept_sp) / prop.n_walkers)
prop_data["e_estimate"] = ept
# prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]

comm.Barrier()
if rank == 0:
    print('# \n')
    print(f'# Propagating with {options["n_walkers"]*size} walkers')
    print("# Equilibration sweeps:")
    print("#   Iter  <exp(t1)>  <t2>  <e0>  <e1>  energy  Walltime")
    print(f"  {0:5d}  {t1[0]:.6f}  {t2[0]:.6f}"
          f"  {e0[0]:.6f}   {e1[0]:.6f}   {ept:.6f}"
          f"  {time.time() - init_time:.2f}")
comm.Barrier()

sampler_eq = sampling.sampler(n_prop_steps=50, n_ene_blocks=5, n_sr_blocks=10)
for n in range(1,options["n_eql"]+1):
    prop_data, (blk_wt, blk_t1, blk_t2, blk_e0, blk_e1) =\
        sample_ccsd_pt2.propagate_phaseless(
            prop_data, ham_data, prop, trial, wave_data, sampler_eq)

    blk_wt = np.array([blk_wt], dtype="float32") 
    blk_t1 = np.array([blk_t1], dtype="float32")
    blk_t2 = np.array([blk_t2], dtype="float32")
    blk_e0 = np.array([blk_e0], dtype="float32")
    blk_e1 = np.array([blk_e1], dtype="float32")

    blk_wt_t1 = np.array([blk_t1 * blk_wt], dtype="float32")
    blk_wt_t2 = np.array([blk_t2 * blk_wt], dtype="float32")
    blk_wt_e0 = np.array([blk_e0 * blk_wt], dtype="float32")
    blk_wt_e1 = np.array([blk_e1 * blk_wt], dtype="float32")

    tot_blk_wt = np.zeros(1, dtype="float32")
    tot_blk_t1 = np.zeros(1, dtype="float32")
    tot_blk_t2 = np.zeros(1, dtype="float32")
    tot_blk_e0 = np.zeros(1, dtype="float32")
    tot_blk_e1 = np.zeros(1, dtype="float32")

    comm.Reduce(
        [blk_wt, MPI.FLOAT],
        [tot_blk_wt, MPI.FLOAT],
        op=MPI.SUM,
        root=0,
    )
    comm.Reduce(
        [blk_wt_t1, MPI.FLOAT],
        [tot_blk_t1, MPI.FLOAT],
        op=MPI.SUM,
        root=0,
    )
    comm.Reduce(
        [blk_wt_t2, MPI.FLOAT],
        [tot_blk_t2, MPI.FLOAT],
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
        blk_t1 = tot_blk_t1 / tot_blk_wt
        blk_t2 = tot_blk_t2 / tot_blk_wt
        blk_e0 = tot_blk_e0 / tot_blk_wt
        blk_e1 = tot_blk_e1 / tot_blk_wt
    comm.Barrier()

    comm.Bcast(blk_wt, root=0)
    comm.Bcast(blk_t1, root=0)
    comm.Bcast(blk_t2, root=0)
    comm.Bcast(blk_e0, root=0)
    comm.Bcast(blk_e1, root=0)
    
    blk_ept = (h0 + 1/blk_t1 * blk_e0 
               + 1/blk_t1 * blk_e1 - 1/blk_t1**2 * blk_t2 * blk_e0)
    
    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = (
         0.9 * prop_data["e_estimate"] + 0.1 * blk_ept[0]
         )

    comm.Barrier()
    if rank == 0:
        print(f"  {n:5d}  {blk_t1[0]:.6f}  {blk_t2[0]:.6f}"
             f"  {blk_e0[0]:.6f}   {blk_e1[0]:.6f}   {blk_ept[0]:.6f}"
             f"  {time.time() - init_time:.2f}")
    comm.Barrier()

comm.Barrier()
if rank == 0:
    print("#\n# Sampling sweeps:")
    print("#  Iter \t energy \t error \t \t Walltime")
comm.Barrier()

glb_blk_wt = None
glb_blk_t1 = None
glb_blk_t2 = None
glb_blk_e0 = None
glb_blk_e1 = None

comm.Barrier()
if rank == 0:
    glb_blk_wt = np.zeros(size * sampler.n_blocks,dtype="float32")
    glb_blk_t1 = np.zeros(size * sampler.n_blocks,dtype="float32")
    glb_blk_t2 = np.zeros(size * sampler.n_blocks,dtype="float32")
    glb_blk_e0 = np.zeros(size * sampler.n_blocks,dtype="float32")
    glb_blk_e1 = np.zeros(size * sampler.n_blocks,dtype="float32")
    ept_samples = np.zeros(sampler.n_blocks,dtype="float32")
comm.Barrier()
    
for n in range(sampler.n_blocks):
    prop_data, (blk_wt, blk_t1, blk_t2, blk_e0, blk_e1) =\
        sample_ccsd_pt2.propagate_phaseless(
            prop_data, ham_data, prop, trial, wave_data, sampler)
    
    blk_wt = np.array([blk_wt], dtype="float32")
    blk_t1 = np.array([blk_t1], dtype="float32")
    blk_t2 = np.array([blk_t2], dtype="float32")
    blk_e0 = np.array([blk_e0], dtype="float32")
    blk_e1 = np.array([blk_e1], dtype="float32")

    gather_wt = None
    gather_t1 = None
    gather_t2 = None
    gather_e0 = None
    gather_e1 = None

    comm.Barrier()
    if rank == 0:
        gather_wt = np.zeros(size, dtype="float32")
        gather_t1 = np.zeros(size, dtype="float32")
        gather_t2 = np.zeros(size, dtype="float32")
        gather_e0 = np.zeros(size, dtype="float32")
        gather_e1 = np.zeros(size, dtype="float32")
    comm.Barrier()

    comm.Gather(blk_wt, gather_wt, root=0)
    comm.Gather(blk_t1, gather_t1, root=0)
    comm.Gather(blk_t2, gather_t2, root=0)
    comm.Gather(blk_e0, gather_e0, root=0)
    comm.Gather(blk_e1, gather_e1, root=0)

    comm.Barrier()
    if rank == 0:
        glb_blk_wt[n * size : (n + 1) * size] = gather_wt
        glb_blk_t1[n * size : (n + 1) * size] = gather_t1
        glb_blk_t2[n * size : (n + 1) * size] = gather_t2
        glb_blk_e0[n * size : (n + 1) * size] = gather_e0
        glb_blk_e1[n * size : (n + 1) * size] = gather_e1

        assert gather_wt is not None

        blk_wt= np.sum(gather_wt)
        blk_t1 = np.sum(gather_wt * gather_t1) / blk_wt
        blk_t2 = np.sum(gather_wt * gather_t2) / blk_wt
        blk_e0 = np.sum(gather_wt * gather_e0) / blk_wt
        blk_e1 = np.sum(gather_wt * gather_e1) / blk_wt
    comm.Barrier()

    comm.Bcast(blk_wt, root=0)
    comm.Bcast(blk_t1, root=0)
    comm.Bcast(blk_t2, root=0)
    comm.Bcast(blk_e0, root=0)
    comm.Bcast(blk_e1, root=0)

    blk_ept = (h0 + 1/blk_t1 * blk_e0 
               + 1/blk_t1 * blk_e1 - 1/blk_t1**2 * blk_t2 * blk_e0)
    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * blk_ept

    comm.Barrier()
    if rank == 0:
        ept_samples[n] = blk_ept
    comm.Barrier()

    if n % (max(sampler.n_blocks // 10, 1)) == 0 and n > 0:
        comm.Barrier()
        if rank == 0:
            t1 = np.sum(glb_blk_wt * glb_blk_t1)/np.sum(glb_blk_wt)
            t2 = np.sum(glb_blk_wt * glb_blk_t2)/np.sum(glb_blk_wt)
            e0 = np.sum(glb_blk_wt * glb_blk_e0)/np.sum(glb_blk_wt)
            e1 = np.sum(glb_blk_wt * glb_blk_e1)/np.sum(glb_blk_wt)

            ept = h0 + 1/t1 * e0 + 1/t1 * e1 - 1/t1**2 * t2 * e0
            
            # (pE/pt1,pE/pt2,pE/pe0,pE/pe1)
            dE = np.array([-1/t1**2*(e0+e1)+2/t1**3*t2*e0,
                           -1/t1**2*e0,
                            1/t1 - 1/t1**2 * t2,
                            1/t1])
            cov_te0e1 = np.cov([glb_blk_t1[:(n+1)*size],
                                glb_blk_t2[:(n+1)*size],
                                glb_blk_e0[:(n+1)*size],
                                glb_blk_e1[:(n+1)*size]])
            ept_err = np.sqrt(dE @ cov_te0e1 @ dE)/np.sqrt((n+1)*size)
            
            print(f"  {n:4d} \t \t {ept:.6f} \t {ept_err:.6f} \t"
                  f"  {time.time() - init_time:.2f}")
        comm.Barrier()


comm.Barrier()
if rank == 0:
    assert glb_blk_wt is not None

    samples_clean, idx = stat_utils.reject_outliers(
    np.stack(
                (
                    glb_blk_wt,
                    glb_blk_t1,
                    glb_blk_t2,
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
    glb_blk_t1 = samples_clean[:, 1]
    glb_blk_t2 = samples_clean[:, 2]
    glb_blk_e0 = samples_clean[:, 3]
    glb_blk_e1 = samples_clean[:, 4]

    glb_wt = np.sum(glb_blk_wt)
    rho_t1 = (glb_blk_wt * glb_blk_t1)/glb_wt
    rho_t2 = (glb_blk_wt * glb_blk_t2)/glb_wt
    rho_e0 = (glb_blk_wt * glb_blk_e0)/glb_wt
    rho_e1 = (glb_blk_wt * glb_blk_e1)/glb_wt
    
    t1 = np.sum(rho_t1)
    t2 = np.sum(rho_t2)
    e0 = np.sum(rho_e0)
    e1 = np.sum(rho_e1)

    t1_err = np.std(rho_t1)
    t2_err = np.std(rho_t2)
    e0_err = np.std(rho_e0)
    e1_err = np.std(rho_e1)

    # t1 = np.sum(glb_blk_wt * glb_blk_t1)/np.sum(glb_blk_wt)
    # t2 = np.sum(glb_blk_wt * glb_blk_t2)/np.sum(glb_blk_wt)
    # e0 = np.sum(glb_blk_wt * glb_blk_e0)/np.sum(glb_blk_wt)
    # e1 = np.sum(glb_blk_wt * glb_blk_e1)/np.sum(glb_blk_wt)

    ept = h0 + 1/t1 * e0 + 1/t1 * e1 - 1/t1**2 * t2 * e0
    
    # dE = (pE/pt1,pE/pt2,pE/pe0,pE/pe1)
    dE = np.array([-1/t1**2*(e0+e1)+2/t1**3*t2*e0,
                   -1/t1**2*e0,
                    1/t1 - 1/t1**2 * t2,
                    1/t1])
    cov_te0e1 = np.cov([glb_blk_t1[:(n+1)*size],
                        glb_blk_t2[:(n+1)*size],
                        glb_blk_e0[:(n+1)*size],
                        glb_blk_e1[:(n+1)*size]])
    ept_err = np.sqrt(dE @ cov_te0e1 @ dE)/np.sqrt(samples_clean.shape[0])

    ept_err = f"{ept_err:.6f}"

    ept = f"{ept:.6f}"

    print(f"Final Results1:")
    print(f'# h0 = {h0:.8f}')
    print(f"# <exp(t1)> = {t1:.6f} +/- {t1_err:.6f}")
    print(f"# <t2> = {t2:.6f} +/- {t2_err:.6f}")
    print(f"# <e0> = {e0:.6f} +/- {e0_err:.6f}")
    print(f"# <e1> = {e1:.6f} +/- {e1_err:.6f}")
    print(f"# AFQMC/UCCSD_PT2 energy (covariance): {ept} +/- {ept_err}")

    d = np.abs(ept_samples-np.median(ept_samples))
    d_med = np.median(d) + 1e-7
    mask = d/d_med < 10
    ept_clean = ept_samples[mask]
    print('# remove outliers: ', len(ept_samples)-len(ept_clean))

    ept_mean = np.mean(ept_clean)
    ept_std = np.std(ept_clean)

    ept = f"{ept_mean:.6f}"
    ept_err = f"{ept_std/np.sqrt(n):.6f}"

    print(f"# AFQMC/UCCSD_PT2 energy (direct obs): {ept} +/- {ept_err}")
    print(f"# total run time: {time.time() - init_time:.2f}")

comm.Barrier()
