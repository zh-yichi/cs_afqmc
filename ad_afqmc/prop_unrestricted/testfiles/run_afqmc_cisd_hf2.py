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
olp, ehf, eci = trial.calc_energy_cisd_hf(
    prop_data['walkers'], ham_data, wave_data)
ehf_init = jnp.array(jnp.sum(ehf) / prop.n_walkers)
eci_init = jnp.array(jnp.sum(eci) / prop.n_walkers)
prop_data["e_estimate"] = ehf
prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]

comm.Barrier()
if rank == 0:
    print(f'# Propagating with {options["n_walkers"]*size} walkers')
    print("# Equilibration sweeps:")
    print("#   Iter \t Energy_HF \t Energy_CISD \t Walltime")
    print(f"  {0:5d} \t {ehf_init:.6f} \t {eci_init:.6f} \t "
          f"  {time.time() - init_time:.2f}")
comm.Barrier()

sampler_eq = sampling.sampler_cisd_hf2(
    n_prop_steps=50, n_ene_blocks=5, n_sr_blocks=10, n_chol = sampler.n_chol)
for n in range(1,options["n_eql"]+1):
    prop_data, (blk_wt, blk_wp, blk_ehf, blk_eci) =\
        sampler_eq.propagate_phaseless(prop_data, ham_data, prop, trial, wave_data)

    blk_wt = np.array([blk_wt], dtype="float64") 
    blk_wp = np.array([blk_wp], dtype="float64")
    blk_ehf = np.array([blk_ehf], dtype="float64")
    blk_eci = np.array([blk_eci], dtype="float64")

    gather_wt = None
    gather_wp = None
    gather_ehf = None
    gather_eci = None

    comm.Barrier()
    if rank == 0:
        gather_wt = np.zeros(size, dtype="float64")
        gather_wp = np.zeros(size, dtype="float64")
        gather_ehf = np.zeros(size, dtype="float64")
        gather_eci = np.zeros(size, dtype="float64")
    comm.Barrier()

    comm.Gather(blk_wt, gather_wt, root=0)
    comm.Gather(blk_wp, gather_wp, root=0)
    comm.Gather(blk_ehf, gather_ehf, root=0)
    comm.Gather(blk_eci, gather_eci, root=0)

    comm.Barrier()
    if rank == 0:
        assert gather_wt is not None
        blk_wt = np.sum(gather_wt)
        blk_wp = np.sum(gather_wp)
        blk_ehf = np.sum(gather_wt * gather_ehf) / blk_wt
        blk_eci = np.sum(gather_wp * gather_eci) / blk_wp
    comm.Barrier()

    comm.Bcast(blk_wt, root=0)
    comm.Bcast(blk_wp, root=0)
    comm.Bcast(blk_ehf, root=0)
    comm.Bcast(blk_eci, root=0)
    
    # blk_eci = h0 + blk_e1/blk_o
    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * blk_ehf

    comm.Barrier()
    if rank == 0:
        print(f"  {n:5d} \t {blk_ehf:.6f} \t {blk_eci:.6f} \t"
              f"  {time.time() - init_time:.2f}")
    comm.Barrier()

comm.Barrier()
if rank == 0:
    print("#\n# Sampling sweeps:")
    print("#  Iter \t Energy_HF \t Error \t Energy_CI \t Error \t Walltime")
comm.Barrier()

glb_blk_wt = None
glb_blk_wp = None
glb_blk_ehf = None
glb_blk_eci = None

comm.Barrier()
if rank == 0:
    glb_blk_wt = np.zeros(size * sampler.n_blocks,dtype="float64")
    glb_blk_wp = np.zeros(size * sampler.n_blocks,dtype="float64")
    glb_blk_ehf = np.zeros(size * sampler.n_blocks,dtype="float64")
    glb_blk_eci = np.zeros(size * sampler.n_blocks,dtype="float64")
    eci_samples = np.zeros(sampler.n_blocks,dtype="float64")
comm.Barrier()
    
for n in range(sampler.n_blocks):
    prop_data, (blk_wt, blk_wp, blk_ehf, blk_eci) =\
        sampler.propagate_phaseless(prop_data, ham_data, prop, trial, wave_data)
    
    blk_wt = np.array([blk_wt], dtype="float64")
    blk_wp = np.array([blk_wp], dtype="float64")
    blk_ehf = np.array([blk_ehf], dtype="float64")
    blk_eci = np.array([blk_eci], dtype="float64")

    gather_wt = None
    gather_wp = None
    gather_ehf = None
    gather_eci = None

    comm.Barrier()
    if rank == 0:
        gather_wt = np.zeros(size, dtype="float64")
        gather_wp = np.zeros(size, dtype="float64")
        gather_ehf = np.zeros(size, dtype="float64")
        gather_eci = np.zeros(size, dtype="float64")
    comm.Barrier()

    comm.Gather(blk_wt, gather_wt, root=0)
    comm.Gather(blk_wp, gather_wp, root=0)
    comm.Gather(blk_ehf, gather_ehf, root=0)
    comm.Gather(blk_eci, gather_eci, root=0)

    comm.Barrier()
    if rank == 0:
        glb_blk_wt[n * size : (n + 1) * size] = gather_wt
        glb_blk_wp[n * size : (n + 1) * size] = gather_wp
        glb_blk_ehf[n * size : (n + 1) * size] = gather_ehf
        glb_blk_eci[n * size : (n + 1) * size] = gather_eci

        assert gather_wt is not None

        blk_wt = np.sum(gather_wt)
        blk_wp = np.sum(gather_wp)
        blk_ehf = np.sum(gather_wt * gather_ehf) / blk_wt
        blk_eci = np.sum(gather_wp * gather_eci) / blk_wp
    comm.Barrier()

    comm.Bcast(blk_wt, root=0)
    comm.Bcast(blk_wp, root=0)
    comm.Bcast(blk_ehf, root=0)
    comm.Bcast(blk_eci, root=0)

    # blk_eci = h0 + blk_e1/blk_o
    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * blk_ehf

    comm.Barrier()
    if rank == 0:
        eci_samples[n] = blk_eci
    comm.Barrier()

    if n % (max(sampler.n_blocks // 10, 1)) == 0 and n > 0:
        comm.Barrier()
        if rank == 0:

            glb_wt = np.sum(glb_blk_wt[:(n+1)*size])
            glb_wp = np.sum(glb_blk_wp[:(n+1)*size])
            rho_ehf = (glb_blk_wt[:(n+1)*size] 
                      * glb_blk_ehf[:(n+1)*size])/glb_wt
            rho_eci = (glb_blk_wp[:(n+1)*size] 
                      * glb_blk_eci[:(n+1)*size])/glb_wp

            ehf = np.sum(rho_ehf)
            eci = np.sum(rho_eci)

            ehf_err = np.sqrt(np.sum(glb_blk_wt[:(n+1)*size] * 
                         (glb_blk_ehf[:(n+1)*size] - ehf)**2) / glb_wt / (n+1)*size)
            eci_err = np.sqrt(np.sum(glb_blk_wp[:(n+1)*size] * 
                         (glb_blk_eci[:(n+1)*size] - eci)**2) / glb_wp / (n+1)*size)

            print(f"  {n:4d} \t {ehf:.6f} \t {ehf_err:.6f} \t"
                  f"  {eci:.6f} \t {eci_err:.6f} \t"
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
                    glb_blk_ehf,
                    glb_blk_eci,
                )
            ).T,
            2,
        )

    print(
        f"# Number of outliers in post: {glb_blk_wt.size - samples_clean.shape[0]} "
        )
    
    glb_blk_wt = samples_clean[:, 0]
    glb_blk_wp = samples_clean[:, 1]
    glb_blk_ehf = samples_clean[:, 2]
    glb_blk_eci = samples_clean[:, 3]

    glb_wt = np.sum(glb_blk_wt)
    glb_wp = np.sum(glb_blk_wp)
    rho_ehf = (glb_blk_wt * glb_blk_ehf)/glb_wt
    rho_eci = (glb_blk_wp * glb_blk_eci)/glb_wp

    ehf = np.sum(rho_ehf)
    eci = np.sum(rho_eci)

    ehf_err = np.sqrt(np.sum(glb_blk_wt * (glb_blk_ehf - ehf)**2) / glb_wt / samples_clean.shape[0])
    eci_err = np.sqrt(np.sum(glb_blk_wp * (glb_blk_eci - eci)**2) / glb_wp / samples_clean.shape[0])

    ehf = f"{ehf:.6f}"
    eci = f"{eci:.6f}"
    ehf_err = f"{ehf_err:.6f}"
    eci_err = f"{eci_err:.6f}"

    print(f"# Final Results:")
    print(f"# AFQMC/HF Energy: {ehf} +/- {ehf_err}")
    print(f"# AFQMC/CISD_HF Energy: {eci} +/- {eci_err}")

    d = np.abs(eci_samples-np.median(eci_samples))
    d_med = np.median(d) + 1e-7
    mask = d/d_med < 10
    eci_clean = eci_samples[mask]
    print('# remove outliers in direct sampling: ', len(eci_samples)-len(eci_clean))

    eci_mean = np.mean(eci_clean)
    eci_std = np.std(eci_clean)
    
    eci = f"{eci_mean:.6f}"
    eci_err = f"{eci_std/np.sqrt(n):.6f}"

    print(f"# AFQMC/CISD_HF energy (direct obs): {eci} +/- {eci_err}")
    print(f"# total run time: {time.time() - init_time:.2f}")

comm.Barrier()
