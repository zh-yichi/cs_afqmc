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
h0 = ham_data['h0']

prop_data = prop.init_prop_data(trial, wave_data, ham_data, init_walkers)
if jnp.abs(jnp.sum(prop_data["overlaps"])) < 1.0e-6:
    raise ValueError(
        "Initial overlaps are zero. Pass walkers with non-zero overlap."
    )
prop_data["key"] = random.PRNGKey(seed + rank)

prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
prop_data["n_killed_walkers"] = 0
o, e0, e1 = trial.calc_energy_cisd_hf(
    prop_data['walkers'], ham_data, wave_data)
eci = h0 + e1/o
o_init = jnp.array(jnp.sum(o) / prop.n_walkers)
e0_init = jnp.array(jnp.sum(e0) / prop.n_walkers)
eci_init = jnp.array(jnp.sum(eci) / prop.n_walkers)
prop_data["e_estimate"] = e0
prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]

comm.Barrier()
if rank == 0:
    print(f'# Propagating with {options["n_walkers"]*size} walkers')
    print("# Equilibration sweeps:")
    print("#   Iter \t olp_T/G \t Energy_HF \t Energy_CI \t Walltime")
    print(f"  {0:5d} \t {o_init:.6f} \t {e0_init:.6f} \t {eci_init:.6f} \t "
          f"  {time.time() - init_time:.2f}")
comm.Barrier()

sampler_eq = sampling.sampler_cisd_hf1(
    n_prop_steps=50, n_ene_blocks=5, n_sr_blocks=10, n_chol = sampler.n_chol)
for n in range(1,options["n_eql"]+1):
    prop_data, (blk_wt, blk_o, blk_e0, blk_e1) =\
        sampler_eq.propagate_phaseless(prop_data, ham_data, prop, trial, wave_data)

    blk_wt = np.array([blk_wt], dtype="float64") 
    blk_o = np.array([blk_o], dtype="float64")
    blk_e0 = np.array([blk_e0], dtype="float64")
    blk_e1 = np.array([blk_e1], dtype="float64")

    gather_wt = None
    gather_o = None
    gather_e0 = None
    gather_e1 = None

    comm.Barrier()
    if rank == 0:
        gather_wt = np.zeros(size, dtype="float64")
        gather_o = np.zeros(size, dtype="float64")
        gather_e0 = np.zeros(size, dtype="float64")
        gather_e1 = np.zeros(size, dtype="float64")
    comm.Barrier()

    comm.Gather(blk_wt, gather_wt, root=0)
    comm.Gather(blk_o, gather_o, root=0)
    comm.Gather(blk_e0, gather_e0, root=0)
    comm.Gather(blk_e1, gather_e1, root=0)

    comm.Barrier()
    if rank == 0:
        assert gather_wt is not None
        blk_wt= np.sum(gather_wt)
        blk_o = np.sum(gather_wt * gather_o) / blk_wt
        blk_e0 = np.sum(gather_wt * gather_e0) / blk_wt
        blk_e1 = np.sum(gather_wt * gather_e1) / blk_wt
    comm.Barrier()

    comm.Bcast(blk_wt, root=0)
    comm.Bcast(blk_o, root=0)
    comm.Bcast(blk_e0, root=0)
    comm.Bcast(blk_e1, root=0)
    
    blk_eci = h0 + blk_e1/blk_o
    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * blk_e0

    comm.Barrier()
    if rank == 0:
        print(f"  {n:5d} \t {blk_o:.6f} \t"
              f"  {blk_e0:.6f} \t {blk_eci:.6f} \t"
              f"  {time.time() - init_time:.2f}")
    comm.Barrier()

comm.Barrier()
if rank == 0:
    print("#\n# Sampling sweeps:")
    print("#  Iter \t Energy_HF \t Energy_CI \t error \t \t Walltime")
comm.Barrier()

glb_blk_wt = None
glb_blk_o = None
glb_blk_e0 = None
glb_blk_e1 = None

comm.Barrier()
if rank == 0:
    glb_blk_wt = np.zeros(size * sampler.n_blocks,dtype="float64")
    glb_blk_o = np.zeros(size * sampler.n_blocks,dtype="float64")
    glb_blk_e0 = np.zeros(size * sampler.n_blocks,dtype="float64")
    glb_blk_e1 = np.zeros(size * sampler.n_blocks,dtype="float64")
    eci_samples = np.zeros(sampler.n_blocks,dtype="float64")
comm.Barrier()
    
for n in range(sampler.n_blocks):
    prop_data, (blk_wt, blk_o, blk_e0, blk_e1) =\
        sampler.propagate_phaseless(prop_data, ham_data, prop, trial, wave_data)
    
    blk_wt = np.array([blk_wt], dtype="float64")
    blk_o = np.array([blk_o], dtype="float64")
    blk_e0 = np.array([blk_e0], dtype="float64")
    blk_e1 = np.array([blk_e1], dtype="float64")

    gather_wt = None
    gather_o = None
    gather_e0 = None
    gather_e1 = None

    comm.Barrier()
    if rank == 0:
        gather_wt = np.zeros(size, dtype="float64")
        gather_o = np.zeros(size, dtype="float64")
        gather_e0 = np.zeros(size, dtype="float64")
        gather_e1 = np.zeros(size, dtype="float64")
    comm.Barrier()

    comm.Gather(blk_wt, gather_wt, root=0)
    comm.Gather(blk_o, gather_o, root=0)
    comm.Gather(blk_e0, gather_e0, root=0)
    comm.Gather(blk_e1, gather_e1, root=0)

    comm.Barrier()
    if rank == 0:
        glb_blk_wt[n * size : (n + 1) * size] = gather_wt
        glb_blk_o[n * size : (n + 1) * size] = gather_o
        glb_blk_e0[n * size : (n + 1) * size] = gather_e0
        glb_blk_e1[n * size : (n + 1) * size] = gather_e1

        assert gather_wt is not None

        blk_wt= np.sum(gather_wt)
        blk_o = np.sum(gather_wt * gather_o) / blk_wt
        blk_e0 = np.sum(gather_wt * gather_e0) / blk_wt
        blk_e1 = np.sum(gather_wt * gather_e1) / blk_wt
    comm.Barrier()

    comm.Bcast(blk_wt, root=0)
    comm.Bcast(blk_o, root=0)
    comm.Bcast(blk_e0, root=0)
    comm.Bcast(blk_e1, root=0)

    blk_eci = h0 + blk_e1/blk_o
    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * blk_e0

    comm.Barrier()
    if rank == 0:
        eci_samples[n] = blk_eci
    comm.Barrier()

    if n % (max(sampler.n_blocks // 10, 1)) == 0 and n > 0:
        comm.Barrier()
        if rank == 0:

            glb_wt = np.sum(glb_blk_wt[:(n+1)*size])
            rho_o = (glb_blk_wt[:(n+1)*size] 
                     * glb_blk_o[:(n+1)*size])/glb_wt
            rho_e0 = (glb_blk_wt[:(n+1)*size] 
                      * glb_blk_e0[:(n+1)*size])/glb_wt
            rho_e1 = (glb_blk_wt[:(n+1)*size] 
                      * glb_blk_e1[:(n+1)*size])/glb_wt

            o = np.sum(rho_o)
            e0 = np.sum(rho_e0)
            e1 = np.sum(rho_e1)

            eci = h0 + e1/o
            dE = np.array([-e1/o**2,1/o])
            cov_oe1 = np.cov([glb_blk_o[:(n+1)*size],
                                glb_blk_e1[:(n+1)*size]])
            eci_err = np.sqrt(dE@cov_oe1@dE + 1e-10)/np.sqrt((n+1)*size)

            print(f"  {n:4d} \t \t {e0:.6f} \t"
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
                    glb_blk_o,
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
    glb_blk_o = samples_clean[:, 1]
    glb_blk_e0 = samples_clean[:, 2]
    glb_blk_e1 = samples_clean[:, 3]

    glb_wt = np.sum(glb_blk_wt)
    rho_o = (glb_blk_wt * glb_blk_o)/glb_wt
    rho_e0 = (glb_blk_wt * glb_blk_e0)/glb_wt
    rho_e1 = (glb_blk_wt * glb_blk_e1)/glb_wt

    o = np.sum(rho_o)
    e0 = np.sum(rho_e0)
    e1 = np.sum(rho_e1)

    o_err = np.std(rho_o)
    e0_err = np.std(rho_e0)
    e1_err = np.std(rho_e1)

    eci = h0 + e1/o
    dE = np.array([-e1/o**2,1/o])
    cov_oe1 = np.cov([glb_blk_o,glb_blk_e1])
    eci_err = np.sqrt(dE@cov_oe1@dE + 1e-10)/np.sqrt(samples_clean.shape[0])

    eci = f"{eci:.6f}"
    eci_err = f"{eci_err:.6f}"

    print(f"# Final Results:")
    print(f"# olp_T/G: {o:.6f} +/- {o_err:.6f}")
    print(f"# AFQMC/HF Energy: {e0:.6f} +/- {e0_err:.6f}")
    print(f"# AFQMC/CISD_HF energy (covariance): {eci} +/- {eci_err}")

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
