from functools import partial
from jax import random
#from mpi4py import MPI
import numpy as np
from jax import numpy as jnp
from ad_afqmc import config, stat_utils
from ad_afqmc.prop_unrestricted import sampling, prep
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

ham_data, ham, prop, trial, wave_data, sampler, observable, options = (prep._prep_afqmc())

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
# h0 = ham_data['h0']

prop_data_init = prop.init_prop_data(trial, wave_data, ham_data, init_walkers)
# prop_data["weights"] = jnp.ones(prop.n_walkers,dtype=jnp.complex128)

if jnp.abs(jnp.sum(prop_data_init["overlaps"])) < 1.0e-6:
    raise ValueError(
        "Initial overlaps are zero. Pass walkers with non-zero overlap."
    )
prop_data_init["key"] = random.PRNGKey(seed + rank)

prop_data_init["overlaps"] = trial.calc_overlap(prop_data_init["walkers"], wave_data)
# prop_data["n_killed_walkers"] = 0
# prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]

blk_time = prop.dt * sampler.n_prop_steps

# comm.Barrier()
# if rank == 0:
#     e_init = prop_data_init["e_estimate"]
#     print('# \n')
#     print(f'# Propagating with {options["n_walkers"]*size} walkers')
#     print("# Free Projection AFQMC Equilibration:")
#     print("# atom_time \t energy \t d_energy \t Walltime")
#     print(f"  {0:5d} \t {e_init:.6f} \t {time.time() - init_time:.2f}")
# comm.Barrier()

glb_blk_w = None
glb_blk_e = None

comm.Barrier()
if rank == 0:
    glb_blk_w = np.zeros((sampler.n_eql_blocks, sampler.n_trj), dtype="complex128")
    glb_blk_e = np.zeros((sampler.n_eql_blocks, sampler.n_trj), dtype="complex128")
comm.Barrier()

for n in range(sampler.n_trj):
    prop_data_init["key"] += n

    comm.Barrier()
    if rank == 0:
        e_init = prop_data_init["e_estimate"]
        print(f'# \n')
        print(f'# Propagating with {options["n_walkers"]*size} walkers')
        print(f"# Free Projection AFQMC Equilibration trajector {n+1}/{sampler.n_trj}")
        print("# atom_time \t energy \t d_energy \t Walltime")
        print(f"    {0:.2f} \t {e_init:.6f} \t ------ \t {time.time() - init_time:.2f}")
    comm.Barrier()
    
    _, (blk_w, blk_e) \
        = sampler.scan_eql_blocks(prop_data_init, ham_data, prop, trial, wave_data)

    blk_w = np.array([blk_w], dtype="complex128")
    blk_e = np.array([blk_e], dtype="complex128")

    # gather_wt = None
    # gather_e = None

    # comm.Barrier()
    # if rank == 0:
    #     gather_wt = np.zeros(size, dtype="float64")
    #     gather_e = np.zeros(size, dtype="float64")
    # comm.Barrier()

    # comm.Gather(blk_wt, gather_wt, root=0)
    # comm.Gather(blk_e, gather_e, root=0)

    # comm.Barrier()
    # if rank == 0:
    #     assert gather_wt is not None
    #     blk_wt= np.sum(gather_wt)
    #     blk_e = np.sum(gather_wt * gather_e) / blk_wt
    # comm.Barrier()

    # comm.Bcast(blk_wt, root=0)
    # comm.Bcast(blk_e, root=0)

    # prop_data = prop.orthonormalize_walkers(prop_data)
    # prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    # prop_data["e_estimate"] = (0.9 * prop_data["e_estimate"] + 0.1 * blk_e)

    comm.Barrier()
    if rank == 0:
        glb_blk_w[:,n] = blk_w
        glb_blk_e[:,n] = blk_e
        # print(blk_w)
        # print(blk_e)
        
        e_mean = np.real(
            np.sum(glb_blk_w[:,:n+1] * glb_blk_e[:,:n+1], axis=1) / np.sum(glb_blk_w[:,:n+1], axis=1)
            )
        if n > 0:
            de_mean = np.real(
                np.sqrt(
                np.sum(glb_blk_w[:,:n+1] * (glb_blk_e[:,:n+1]-e_mean[:, None])**2, axis=1
                       ) / np.sum(glb_blk_w[:,:n+1], axis=1)) / np.sqrt(n))

            for nb in range(sampler.n_eql_blocks):
                print(f"    {(nb+1)*blk_time:.2f} \t {e_mean[nb]:.6f} \t {de_mean[nb]:.6f} \t {time.time() - init_time:.2f} ")
        elif n == 0:
            for nb in range(sampler.n_eql_blocks):
                print(f"    {(nb+1)*blk_time:.2f} \t {e_mean[nb]:.6f} \t ------ \t {time.time() - init_time:.2f} ")

        # print(glb_blk_w)
        # print(glb_blk_e)

    comm.Barrier()

comm.Barrier()
