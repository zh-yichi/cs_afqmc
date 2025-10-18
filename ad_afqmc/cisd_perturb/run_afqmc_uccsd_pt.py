import time
import argparse
import numpy as np
from jax import random
from jax import numpy as jnp
from functools import partial
from ad_afqmc import config, wavefunctions, stat_utils, mpi_jax
from ad_afqmc.cisd_perturb import sample_uccsd_pt, ccsd_pt
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
init_time = time.time()

print = partial(print, flush=True)

ham_data, ham, prop, trial, wave_data, sampler, observable, options, _ \
    = (mpi_jax._prep_afqmc())
# trial = wavefunctions.cisd_pt(trial.norb, trial.nelec,n_batch=trial.n_batch)

h0 = ham_data['h0']
seed = options["seed"]
neql = options["n_eql"]

trial_rdm1 = trial.get_rdm1(wave_data)
if "rdm1" not in wave_data:
    wave_data["rdm1"] = trial_rdm1

mo_coeff = wave_data["mo_coeff"]
guide_wave_data = wave_data.copy()
guide_wave_data["mo_coeff"] = [
    mo_coeff[0][:, : trial.nelec[0]],
    mo_coeff[1][:, : trial.nelec[1]],
]

guide = wavefunctions.uhf(
    trial.norb, trial.nelec,n_batch=trial.n_batch
    )
guide_ham_data = ham_data.copy()
guide_ham_data = guide._build_measurement_intermediates(
    guide_ham_data, guide_wave_data)
guide_ham_data = prop._build_propagation_intermediates(
    guide_ham_data, guide, wave_data)

ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
ham_data = ham.build_propagation_intermediates(
    ham_data, prop, trial, wave_data
)

prop_data = prop.init_prop_data(trial, wave_data, ham_data, None)
if jnp.abs(jnp.sum(prop_data["overlaps"])) < 1.0e-6:
    raise ValueError(
        "Initial overlaps are zero. Pass walkers with non-zero overlap."
    )
prop_data["key"] = random.PRNGKey(seed + rank)

t, e0, e1 = ccsd_pt._uccsd_walker_energy_pt(
    prop_data['walkers'][0][0],prop_data['walkers'][1][0],ham_data,wave_data,trial)
init_ept = e0 + e1 - t* (e0-h0)

comm.Barrier()
if rank == 0:
    print("# Equilibration sweeps:")
    print("#   Iter \t energy \t Walltime")
    print(f"  {0:5d} \t {init_ept:.6f} \t {time.time() - init_time:.2f}")
comm.Barrier()

for n in range(1, neql + 1):
    prop_data, (blk_wt, blk_t, blk_e0, blk_e1) =\
        sample_uccsd_pt.propagate_phaseless(
            prop_data, ham_data, prop, trial, wave_data, sampler,
            guide,guide_ham_data,guide_wave_data)
    
    blk_wt = np.array([blk_wt], dtype="float32")
    blk_t = np.array([blk_t], dtype="float32") #"complex64")
    blk_e0 = np.array([blk_e0], dtype="float32") #"complex64")
    blk_e1 = np.array([blk_e1], dtype="float32") #"complex64")

    blk_wt_t = np.array([blk_t * blk_wt], dtype="float32") #"complex64"
    blk_wt_e0 = np.array([blk_e0 * blk_wt], dtype="float32") #"complex64"
    blk_wt_e1 = np.array([blk_e1 * blk_wt], dtype="float32") #"complex64"

    tot_blk_wt = np.zeros(1, dtype="float32")
    tot_blk_t = np.zeros(1, dtype="float32") #"complex64")
    tot_blk_e0 = np.zeros(1, dtype="float32") #"complex64")
    tot_blk_e1 = np.zeros(1, dtype="float32") #"complex64")
    
    comm.Reduce(
        [blk_wt, MPI.FLOAT],
        [tot_blk_wt, MPI.FLOAT],
        op=MPI.SUM,
        root=0,
    )
    comm.Reduce(
        [blk_wt_t, MPI.FLOAT],
        [tot_blk_t, MPI.FLOAT],
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
        blk_t = tot_blk_t / tot_blk_wt
        blk_e0 = tot_blk_e0 / tot_blk_wt
        blk_e1 = tot_blk_e1 / tot_blk_wt
    comm.Barrier()

    comm.Bcast(blk_wt, root=0)
    comm.Bcast(blk_t, root=0)
    comm.Bcast(blk_e0, root=0)
    comm.Bcast(blk_e1, root=0)

    blk_ept = blk_e0 + blk_e1 - blk_t * (blk_e0 - h0)
    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = (
        0.9 * prop_data["e_estimate"] + 0.1 * blk_ept[0]
    )

    comm.Barrier()
    if rank == 0:
        print(
            f"  {n:5d} \t {blk_ept[0]:.6f} \t {time.time() - init_time:.2f} ",
            flush=True,
        )
    comm.Barrier()

glb_blk_wt = None
glb_blk_t = None
glb_blk_e0 = None
glb_blk_e1 = None

comm.Barrier()
if rank == 0:
    glb_blk_wt = np.zeros(size * sampler.n_blocks,dtype="float32")
    glb_blk_t = np.zeros(size * sampler.n_blocks,dtype="float32")#"complex64")
    glb_blk_e0 = np.zeros(size * sampler.n_blocks,dtype="float32")#"complex64")
    glb_blk_e1 = np.zeros(size * sampler.n_blocks,dtype="float32")#"complex64")
    ept_samples = np.zeros(sampler.n_blocks,dtype="float32")
comm.Barrier()
    
comm.Barrier()
if rank == 0:
    print("#\n# Sampling sweeps:")
    print("#  Iter \t energy \t error \t \t Walltime")
comm.Barrier()

for n in range(sampler.n_blocks):
    prop_data, (blk_wt, blk_t, blk_e0, blk_e1) = \
        sample_uccsd_pt.propagate_phaseless(
            prop_data, ham_data, prop, trial, wave_data, sampler,
            guide,guide_ham_data,guide_wave_data
        )
    blk_wt = np.array([blk_wt], dtype="float32")
    blk_t = np.array([blk_t], dtype="float32")#"complex64")
    blk_e0 = np.array([blk_e0], dtype="float32")#"complex64")
    blk_e1 = np.array([blk_e1], dtype="float32")#"complex64")

    gather_wt = None
    gather_t = None
    gather_e0 = None
    gather_e1 = None

    comm.Barrier()
    if rank == 0:
        gather_wt = np.zeros(size, dtype="float32")
        gather_t = np.zeros(size, dtype="float32")#"complex64")
        gather_e0 = np.zeros(size, dtype="float32")#"complex64")
        gather_e1 = np.zeros(size, dtype="float32")#"complex64")
    comm.Barrier()

    comm.Gather(blk_wt, gather_wt, root=0)
    comm.Gather(blk_t, gather_t, root=0)
    comm.Gather(blk_e0, gather_e0, root=0)
    comm.Gather(blk_e1, gather_e1, root=0)

    comm.Barrier()
    if rank == 0:
        glb_blk_wt[n * size : (n + 1) * size] = gather_wt
        glb_blk_t[n * size : (n + 1) * size] = gather_t
        glb_blk_e0[n * size : (n + 1) * size] = gather_e0
        glb_blk_e1[n * size : (n + 1) * size] = gather_e1

        assert gather_wt is not None

        blk_wt= np.sum(gather_wt)
        blk_t = np.sum(gather_wt * gather_t) / blk_wt
        blk_e0 = np.sum(gather_wt * gather_e0) / blk_wt
        blk_e1 = np.sum(gather_wt * gather_e1) / blk_wt
    comm.Barrier()

    comm.Bcast(blk_wt, root=0)
    comm.Bcast(blk_t, root=0)
    comm.Bcast(blk_e0, root=0)
    comm.Bcast(blk_e1, root=0)

    blk_ept = blk_e0 + blk_e1 - blk_t * (blk_e0 - h0)
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
            t = np.sum(glb_blk_wt * glb_blk_t)/np.sum(glb_blk_wt)
            e0 = np.sum(glb_blk_wt * glb_blk_e0)/np.sum(glb_blk_wt)
            e1 = np.sum(glb_blk_wt * glb_blk_e1)/np.sum(glb_blk_wt)

            ept = e0 + e1 - t * (e0 - h0)
            dE = np.array([-e0+h0,1-t,1])
            cov_te0e1 = np.cov([glb_blk_t[:(n+1)*size],
                                glb_blk_e0[:(n+1)*size],
                                glb_blk_e1[:(n+1)*size]])
            ept_err = np.sqrt(dE @ cov_te0e1 @ dE)/np.sqrt((n+1)*size)
                

            print(f"  {n:4d} \t \t {blk_ept:.6f} \t {ept_err:.6f} \t"
                  f"  {time.time() - init_time:.2f}")
        comm.Barrier()

comm.Barrier()
if rank == 0:
    assert glb_blk_wt is not None

    samples_clean, idx = stat_utils.reject_outliers(
    np.stack(
                (
                    glb_blk_wt,
                    glb_blk_t,
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
    glb_blk_t = samples_clean[:, 1]
    glb_blk_e0 = samples_clean[:, 2]
    glb_blk_e1 = samples_clean[:, 3]

    t = np.sum(glb_blk_wt * glb_blk_t)/np.sum(glb_blk_wt)
    e0 = np.sum(glb_blk_wt * glb_blk_e0)/np.sum(glb_blk_wt)
    e1 = np.sum(glb_blk_wt * glb_blk_e1)/np.sum(glb_blk_wt)

    ept = e0 + e1 - t * (e0 - h0)
    dE = np.array([-e0+h0,1-t,1])
    cov_te0e1 = np.cov([glb_blk_t,glb_blk_e0,glb_blk_e1])
    ept_err = np.sqrt(dE @ cov_te0e1 @ dE)/np.sqrt(samples_clean.shape[0])

    print(f"Final Results1:")
    print(f"AFQMC/CCSD_PT energy: {ept:.6f} +/- {ept_err:.6f}")
    print(f"total run time: {time.time() - init_time:.2f}")

    ept_mean = np.mean(ept_samples)
    ept_std = np.std(ept_samples)

    d = np.abs(ept_samples-np.median(ept_samples))
    d_med = np.median(d) + 1e-7
    mask = d/d_med < 10
    ept_clean = ept_samples[mask]
    print('# remove outliers: ', len(ept_samples)-len(ept_clean))

    ept_mean = np.mean(ept_clean)
    ept_std = np.std(ept_clean)

    ept = f"{ept_mean:.6f}"
    ept_err = f"{ept_std/np.sqrt(n):.6f}"

    print(f"Final Results2:")
    print(f"AFQMC/CCSD_PT energy: {ept} +/- {ept_err}")
    print(f"total run time: {time.time() - init_time:.2f}")

comm.Barrier()

