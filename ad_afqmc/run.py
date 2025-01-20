import os
import pickle
from functools import partial
from ad_afqmc import config
#from jax import random
#from jax import numpy as jnp
#import argparse
#from mpi4py import MPI
#import numpy as np
#from ad_afqmc.corr_sample_test import corr_sample
#import time

# print = partial(print, flush=True)
# nwalkers = 10
# n_runs = 100
# rlx_steps = 0
# prop_steps = 10
# dt = 0.01
# n_exp_terms = 6
# seed = 23
# cs = True
# use_gpu = False

# options = {
#     "dt": dt,
#     "n_exp_terms": n_exp_terms,
#     "n_walkers": nwalkers,
#     "seed": seed,
#     "walker_type": "rhf",
#     "trial": "rhf",
#     "corr_samp": cs,
#     "free_proj": False,
#     "use_gpu": False
# }

files = {
    "mo1":"mo1.npz",
    "chol1":"chol1",
    "amp1":"amp1.npz",
    "mo2":"mo2.npz",
    "chol2":"chol2",
    "amp2":"amp2.npz",
}

def run_cs_afqmc(options=None,files=None,script=None,mpi_prefix=None, nproc=None):
    if options is None:
        options = {}
    if files is None:
        raise ValueError("files for correlated sampling not found!")
    with open("options.bin", "wb") as f:
        pickle.dump(options, f)
    with open("files.pkl", "wb") as f:
        pickle.dump(files, f)
    if script is None:
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        script = f"{dir_path}/run_cs.py"
    use_gpu = options["use_gpu"]
    if use_gpu:
        config.afqmc_config["use_gpu"] = True
        gpu_flag = "--use_gpu"
    else: gpu_flag = ""
    if mpi_prefix is None:
        if use_gpu:
            mpi_prefix = ""
        else:
            mpi_prefix = "mpirun "
    if nproc is not None:
        mpi_prefix += f"-np {nproc} "
    os.system(
        f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; {mpi_prefix} python {script} {gpu_flag} |tee cs_afqmc.out"
    )
    # try:
    #     ene_err = np.loadtxt("ene_err.txt")
    # except:
    #     print("AFQMC did not execute correctly.")
    #     ene_err = 0.0, 0.0
    return None
