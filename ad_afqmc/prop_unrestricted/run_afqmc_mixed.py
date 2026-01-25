import numpy as np
from jax import random
from jax import numpy as jnp
from functools import partial
from ad_afqmc import config, stat_utils
from ad_afqmc.prop_unrestricted import prop_unrestricted, sampling
import time
import argparse

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
    prop_unrestricted._prep_afqmc())

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
ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
ham_data = ham.build_propagation_intermediates(ham_data, prop, trial, wave_data)

prop_data = prop.init_prop_data(trial, wave_data, ham_data, init_walkers)
if jnp.abs(jnp.sum(prop_data["overlaps"])) < 1.0e-6:
    raise ValueError(
        "Initial overlaps are zero. Pass walkers with non-zero overlap."
    )
prop_data["key"] = random.PRNGKey(seed + rank)

prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
prop_data["n_killed_walkers"] = 0
olp, e1, e2 = trial.calc_energy_mixed(
    prop_data['walkers'], ham_data, wave_data)
e1_init = jnp.array(jnp.sum(e1) / prop.n_walkers)
e2_init = jnp.array(jnp.sum(e2) / prop.n_walkers)
prop_data["e_estimate"] = e1
prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]

comm.Barrier()
if rank == 0:
    print(f'# Propagating with {options["n_walkers"]*size} walkers')
    print("# Equilibration sweeps:")
    print("#   Iter \t Energy_G \t Energy_T \t Walltime")
    print(f"  {0:5d} \t {e1_init:.6f} \t {e2_init:.6f} \t "
          f"  {time.time() - init_time:.2f}")
comm.Barrier()

sampler_eq = sampling.sampler_mixed(
    n_prop_steps=50, n_ene_blocks=5, n_sr_blocks=10, n_chol = sampler.n_chol)
for n in range(1,options["n_eql"]+1):
    prop_data, (blk_wt, blk_wp, blk_e1, blk_e2) =\
        sampler_eq.propagate_phaseless(prop_data, ham_data, prop, trial, wave_data)
    # e1 is by the guide and e2 is by the trial
    # wp = oT/oG
    blk_wt = np.array([blk_wt], dtype="float64") 
    blk_wp = np.array([blk_wp], dtype="float64")
    blk_e1 = np.array([blk_e1], dtype="float64")
    blk_e2 = np.array([blk_e2], dtype="float64")

    gather_wt = None
    gather_wp = None
    gather_e1 = None
    gather_e2 = None

    comm.Barrier()
    if rank == 0:
        gather_wt = np.zeros(size, dtype="float64")
        gather_wp = np.zeros(size, dtype="float64")
        gather_e1 = np.zeros(size, dtype="float64")
        gather_e2 = np.zeros(size, dtype="float64")
    comm.Barrier()

    comm.Gather(blk_wt, gather_wt, root=0)
    comm.Gather(blk_wp, gather_wp, root=0)
    comm.Gather(blk_e1, gather_e1, root=0)
    comm.Gather(blk_e2, gather_e2, root=0)

    comm.Barrier()
    if rank == 0:
        assert gather_wt is not None
        blk_wt = np.sum(gather_wt)
        blk_wp = np.sum(gather_wp)
        blk_e1 = np.sum(gather_wt * gather_e1) / blk_wt
        blk_e2 = np.sum(gather_wp * gather_e2) / blk_wp
    comm.Barrier()

    comm.Bcast(blk_wt, root=0)
    comm.Bcast(blk_wp, root=0)
    comm.Bcast(blk_e1, root=0)
    comm.Bcast(blk_e2, root=0)
    
    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * blk_e1

    comm.Barrier()
    if rank == 0:
        print(f"  {n:5d} \t {blk_e1:.6f} \t {blk_e2:.6f} \t"
              f"  {time.time() - init_time:.2f}")
    comm.Barrier()

comm.Barrier()
if rank == 0:
    print("#\n# Sampling sweeps:")
    print("#  Iter \t Energy_G \t Error \t Energy_T \t Error \t Walltime")
comm.Barrier()

glb_blk_wt = None
glb_blk_wp = None
glb_blk_e1 = None
glb_blk_e2 = None

comm.Barrier()
if rank == 0:
    glb_blk_wt = np.zeros(size * sampler.n_blocks,dtype="float64")
    glb_blk_wp = np.zeros(size * sampler.n_blocks,dtype="float64")
    glb_blk_e1 = np.zeros(size * sampler.n_blocks,dtype="float64")
    glb_blk_e2 = np.zeros(size * sampler.n_blocks,dtype="float64")
    e2_samples = np.zeros(sampler.n_blocks,dtype="float64")
comm.Barrier()
    
for n in range(sampler.n_blocks):
    prop_data, (blk_wt, blk_wp, blk_e1, blk_e2) =\
        sampler.propagate_phaseless(prop_data, ham_data, prop, trial, wave_data)
    
    blk_wt = np.array([blk_wt], dtype="float64")
    blk_wp = np.array([blk_wp], dtype="float64")
    blk_e1 = np.array([blk_e1], dtype="float64")
    blk_e2 = np.array([blk_e2], dtype="float64")

    gather_wt = None
    gather_wp = None
    gather_e1 = None
    gather_e2 = None

    comm.Barrier()
    if rank == 0:
        gather_wt = np.zeros(size, dtype="float64")
        gather_wp = np.zeros(size, dtype="float64")
        gather_e1 = np.zeros(size, dtype="float64")
        gather_e2 = np.zeros(size, dtype="float64")
    comm.Barrier()

    comm.Gather(blk_wt, gather_wt, root=0)
    comm.Gather(blk_wp, gather_wp, root=0)
    comm.Gather(blk_e1, gather_e1, root=0)
    comm.Gather(blk_e2, gather_e2, root=0)

    comm.Barrier()
    if rank == 0:
        glb_blk_wt[n * size : (n + 1) * size] = gather_wt
        glb_blk_wp[n * size : (n + 1) * size] = gather_wp
        glb_blk_e1[n * size : (n + 1) * size] = gather_e1
        glb_blk_e2[n * size : (n + 1) * size] = gather_e2

        assert gather_wt is not None

        blk_wt = np.sum(gather_wt)
        blk_wp = np.sum(gather_wp)
        blk_e1 = np.sum(gather_wt * gather_e1) / blk_wt
        blk_e2 = np.sum(gather_wp * gather_e2) / blk_wp
    comm.Barrier()

    comm.Bcast(blk_wt, root=0)
    comm.Bcast(blk_wp, root=0)
    comm.Bcast(blk_e1, root=0)
    comm.Bcast(blk_e2, root=0)

    # blk_eci = h0 + blk_e1/blk_o
    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * blk_e1

    comm.Barrier()
    if rank == 0:
        e2_samples[n] = blk_e2
    comm.Barrier()

    if n % (max(sampler.n_blocks // 10, 1)) == 0 and n > 0:
        comm.Barrier()
        if rank == 0:

            glb_wt = np.sum(glb_blk_wt[:(n+1)*size])
            glb_wp = np.sum(glb_blk_wp[:(n+1)*size])
            rho_e1 = (glb_blk_wt[:(n+1)*size] 
                      * glb_blk_e1[:(n+1)*size])/glb_wt
            rho_e2 = (glb_blk_wp[:(n+1)*size] 
                      * glb_blk_e2[:(n+1)*size])/glb_wp

            e1 = np.sum(rho_e1)
            e2 = np.sum(rho_e2)

            e1_err = np.sqrt(np.sum(glb_blk_wt[:(n+1)*size] * 
                         (glb_blk_e1[:(n+1)*size] - e1)**2) / glb_wt / (n+1)*size)
            e2_err = np.sqrt(np.sum(glb_blk_wp[:(n+1)*size] * 
                         (glb_blk_e2[:(n+1)*size] - e2)**2) / glb_wp / (n+1)*size)

            print(f"  {n:4d} \t {e1:.6f} \t {e2_err:.6f} \t"
                  f"  {e2:.6f} \t {e2_err:.6f} \t"
                  f"  {time.time() - init_time:.2f}")
        comm.Barrier()


comm.Barrier()
if rank == 0:
    assert glb_blk_wt is not None

    samples_clean, idx = stat_utils.reject_outliers(
    np.stack(
                (
                    glb_blk_wt,
                    glb_blk_wp,
                    glb_blk_e1,
                    glb_blk_e2,
                )
            ).T,
            2,
        )

    print(
        f"# Number of outliers in post: {glb_blk_wt.size - samples_clean.shape[0]} "
        )
    
    glb_blk_wt = samples_clean[:, 0]
    glb_blk_wp = samples_clean[:, 1]
    glb_blk_e1 = samples_clean[:, 2]
    glb_blk_e2 = samples_clean[:, 3]

    glb_wt = np.sum(glb_blk_wt)
    glb_wp = np.sum(glb_blk_wp)
    rho_e1 = (glb_blk_wt * glb_blk_e1)/glb_wt
    rho_e2 = (glb_blk_wp * glb_blk_e2)/glb_wp

    e1 = np.sum(rho_e1)
    e2 = np.sum(rho_e2)

    e1_err = np.sqrt(np.sum(glb_blk_wt * (glb_blk_e1 - e1)**2) / glb_wt / samples_clean.shape[0])
    e2_err = np.sqrt(np.sum(glb_blk_wp * (glb_blk_e2 - e2)**2) / glb_wp / samples_clean.shape[0])

    e1 = f"{e1:.6f}"
    e2 = f"{e2:.6f}"
    e2_err = f"{e2_err:.6f}"
    e2_err = f"{e2_err:.6f}"

    print(f"# Final Results:")
    print(f"# AFQMC Energy by Guide: {e1} +/- {e1_err}")
    print(f"# AFQMC Energy by Trial: {e2} +/- {e2_err}")

    d = np.abs(e2_samples-np.median(e2_samples))
    d_med = np.median(d) + 1e-7
    mask = d/d_med < 10
    e2_clean = e2_samples[mask]
    print('# remove outliers in direct sampling: ', len(e2_samples)-len(e2_clean))

    e2_mean = np.mean(e2_clean)
    e2_std = np.std(e2_clean)
    
    e2 = f"{e2_mean:.6f}"
    e2_err = f"{e2_std/np.sqrt(n):.6f}"

    print(f"# AFQMC/CISD_HF energy (direct obs): {e2} +/- {e2_err}")
    print(f"# total run time: {time.time() - init_time:.2f}")

comm.Barrier()
