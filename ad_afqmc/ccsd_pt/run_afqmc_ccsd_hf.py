from functools import partial
from jax import random
#from mpi4py import MPI
import numpy as np
from jax import numpy as jnp
from ad_afqmc import config, sampling, stat_utils, mpi_jax
from ad_afqmc.ccsd_pt import sample_ccsd_hf
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

init_time = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

### initialize propagation
seed = options["seed"]
init_walkers = None
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

comm.Barrier()
if rank == 0:
    o, e = trial._calc_energy_hf_restricted(
        prop_data['walkers'][0], ham_data, wave_data)
    ehf = h0 + e/o
    print('# \n')
    print(f'# Propagating with {options["n_walkers"]*size} walkers')
    print("# Equilibration sweeps:")
    print("#   Iter \t energy \t Walltime")
    print(f"  {0:5d} \t {ehf:.6f} \t {time.time() - init_time:.2f}")
comm.Barrier()

sampler_eq = sampling.sampler(n_prop_steps=50, n_ene_blocks=5, n_sr_blocks=10)
for n in range(1,options["n_eql"]+1):
    prop_data, (blk_wt, blk_o, blk_e) =\
        sample_ccsd_hf.propagate_phaseless(
            prop_data, ham_data, prop, trial, wave_data, sampler_eq)

    blk_wt = np.array([blk_wt], dtype="float32") 
    blk_o = np.array([blk_o], dtype="float32")
    blk_e = np.array([blk_e], dtype="float32")
    # blk_e1 = np.array([blk_e1], dtype="float32")

    blk_wt_o = np.array([blk_o * blk_wt], dtype="float32")
    blk_wt_e = np.array([blk_e * blk_wt], dtype="float32")
    # blk_wt_e1 = np.array([blk_e1 * blk_wt], dtype="float32")

    tot_blk_wt = np.zeros(1, dtype="float32")
    tot_blk_o = np.zeros(1, dtype="float32")
    tot_blk_e = np.zeros(1, dtype="float32")
    # tot_blk_e1 = np.zeros(1, dtype="float32")

    comm.Reduce(
        [blk_wt, MPI.FLOAT],
        [tot_blk_wt, MPI.FLOAT],
        op=MPI.SUM,
        root=0,
    )
    comm.Reduce(
        [blk_wt_o, MPI.FLOAT],
        [tot_blk_o, MPI.FLOAT],
        op=MPI.SUM,
        root=0,
    )
    comm.Reduce(
        [blk_wt_e, MPI.FLOAT],
        [tot_blk_e, MPI.FLOAT],
        op=MPI.SUM,
        root=0,
    )
    # comm.Reduce(
    #     [blk_wt_e1, MPI.FLOAT],
    #     [tot_blk_e1, MPI.FLOAT],
    #     op=MPI.SUM,
    #     root=0,
    # )

    comm.Barrier()
    if rank == 0:
        blk_wt = tot_blk_wt
        blk_o = tot_blk_o / tot_blk_wt
        blk_e = tot_blk_e / tot_blk_wt
        # blk_e1 = tot_blk_e1 / tot_blk_wt
    comm.Barrier()

    comm.Bcast(blk_wt, root=0)
    comm.Bcast(blk_o, root=0)
    comm.Bcast(blk_e, root=0)
    # comm.Bcast(blk_e1, root=0)
    
    # blk_ept = blk_e0 + blk_e1 - blk_t * (blk_e0-h0)
    blk_ehf = h0 + blk_e/blk_o
    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = (
         0.9 * prop_data["e_estimate"] + 0.1 * blk_ehf[0]
         )
    # comm.Barrier()

    comm.Barrier()
    if rank == 0:
        print(
            f"  {n:5d} \t {blk_ehf[0]:.6f} \t {time.time() - init_time:.2f} "
        )
    comm.Barrier()

comm.Barrier()
if rank == 0:
    print("#\n# Sampling sweeps:")
    print("#  Iter \t energy \t error \t \t Walltime")
comm.Barrier()

glb_blk_wt = None
glb_blk_o = None
glb_blk_e = None
# glb_blk_e1 = None

comm.Barrier()
if rank == 0:
    glb_blk_wt = np.zeros(size * sampler.n_blocks,dtype="float32")
    glb_blk_o = np.zeros(size * sampler.n_blocks,dtype="float32")
    glb_blk_e = np.zeros(size * sampler.n_blocks,dtype="float32")
    # glb_blk_e1 = np.zeros(size * sampler.n_blocks,dtype="float32")
    ehf_samples = np.zeros(sampler.n_blocks,dtype="float32")
comm.Barrier()
    
for n in range(sampler.n_blocks):
    prop_data, (blk_wt, blk_o, blk_e) =\
        sample_ccsd_hf.propagate_phaseless(
            prop_data, ham_data, prop, trial, wave_data, sampler)
    
    blk_wt = np.array([blk_wt], dtype="float32")
    blk_o = np.array([blk_o], dtype="float32")
    blk_e = np.array([blk_e], dtype="float32")
    # blk_e1 = np.array([blk_e1], dtype="float32")

    gather_wt = None
    gather_o = None
    gather_e = None
    # gather_e1 = None

    comm.Barrier()
    if rank == 0:
        gather_wt = np.zeros(size, dtype="float32")
        gather_o = np.zeros(size, dtype="float32")
        gather_e = np.zeros(size, dtype="float32")
        # gather_e1 = np.zeros(size, dtype="float32")
    comm.Barrier()

    comm.Gather(blk_wt, gather_wt, root=0)
    comm.Gather(blk_o, gather_o, root=0)
    comm.Gather(blk_e, gather_e, root=0)
    # comm.Gather(blk_e1, gather_e1, root=0)

    comm.Barrier()
    if rank == 0:
        glb_blk_wt[n * size : (n + 1) * size] = gather_wt
        glb_blk_o[n * size : (n + 1) * size] = gather_o
        glb_blk_e[n * size : (n + 1) * size] = gather_e
        # glb_blk_e1[n * size : (n + 1) * size] = gather_e1

        assert gather_wt is not None

        blk_wt= np.sum(gather_wt)
        blk_o = np.sum(gather_wt * gather_o) / blk_wt
        blk_e = np.sum(gather_wt * gather_e) / blk_wt
        # blk_e1 = np.sum(gather_wt * gather_e1) / blk_wt
    comm.Barrier()

    comm.Bcast(blk_wt, root=0)
    comm.Bcast(blk_o, root=0)
    comm.Bcast(blk_e, root=0)
    # comm.Bcast(blk_e1, root=0)

    blk_ehf = h0 + blk_e/blk_o
    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * blk_ehf

    comm.Barrier()
    if rank == 0:
        ehf_samples[n] = blk_ehf
    comm.Barrier()

    if n % (max(sampler.n_blocks // 10, 1)) == 0 and n > 0:
        comm.Barrier()
        if rank == 0:
            o = np.sum(glb_blk_wt * glb_blk_o)/np.sum(glb_blk_wt)
            e = np.sum(glb_blk_wt * glb_blk_e)/np.sum(glb_blk_wt)
            # e1 = np.sum(glb_blk_wt * glb_blk_e1)/np.sum(glb_blk_wt)

            ehf = h0 + e/o
            # (pE/po,pE/pe)
            dE = np.array([-e/o**2,1/o])
            cov_oe = np.cov([glb_blk_o[:(n+1)*size],
                             glb_blk_e[:(n+1)*size]])
            ehf_err = np.sqrt(dE @ cov_oe @ dE)/np.sqrt((n+1)*size)
            print(f"  {n:4d} \t \t {ehf:.6f} \t {ehf_err:.6f} \t"
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
                    glb_blk_e
                )
            ).T,
            1,
        )

    print(
        f"# Number of outliers in post: {glb_blk_wt.size - samples_clean.shape[0]} "
        )
    
    glb_blk_wt = samples_clean[:, 0]
    glb_blk_o = samples_clean[:, 1]
    glb_blk_e = samples_clean[:, 2]
    # glb_blk_e1 = samples_clean[:, 3]

    o = np.sum(glb_blk_wt * glb_blk_o)/np.sum(glb_blk_wt)
    e = np.sum(glb_blk_wt * glb_blk_e)/np.sum(glb_blk_wt)
    # e1 = np.sum(glb_blk_wt * glb_blk_e1)/np.sum(glb_blk_wt)

    ehf = h0 + e/o
    # (pE/po,pE/pe)
    dE = np.array([-e/o**2,1/o])
    cov_oe = np.cov([glb_blk_o,glb_blk_e])
    ehf_err = np.sqrt(dE @ cov_oe @ dE)/np.sqrt(samples_clean.shape[0])

    ehf_err = f"{ehf_err:.6f}"

    ehf = f"{ehf:.6f}"

    print(f"Final Results1:")
    print(f"AFQMC/CCSD_PT energy: {ehf} +/- {ehf_err}")

    d = np.abs(ehf_samples-np.median(ehf_samples))
    d_med = np.median(d) + 1e-7
    mask = d/d_med < 10
    ehf_clean = ehf_samples[mask]
    print('# remove outliers: ', len(ehf_samples)-len(ehf_clean))

    ehf_mean = np.mean(ehf_clean)
    ehf_std = np.std(ehf_clean)

    ehf = f"{ehf_mean:.6f}"
    ehf_err = f"{ehf_std/np.sqrt(n):.6f}"

    print(f"Final Results2:")
    print(f"AFQMC/CCSD_PT energy: {ehf} +/- {ehf_err}")
    print(f"total run time: {time.time() - init_time:.2f}")

comm.Barrier()
