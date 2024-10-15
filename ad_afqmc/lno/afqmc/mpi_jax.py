import os
import platform
import socket
# os.environ["XLA_FLAGS"] = (
#     "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
# )
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
from jax import config
# config.update("jax_platform_name", "cpu")
import pickle
import time
import h5py
import numpy as np
from jax import numpy as jnp
# from mpi4py import MPI
from ad_afqmc import driver, hamiltonian, propagation, sampling, wavefunctions

use_gpu = True

if use_gpu:
    os.environ["JAX_ENABLE_X64"] = "True"
    config.update("jax_enable_x64", True)
    config.update("jax_platform_name", "gpu")
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    hostname = socket.gethostname()
    system_type = platform.system()
    machine_type = platform.machine()
    processor = platform.processor()
    print(f"# Hostname: {hostname}")
    print(f"# System Type: {system_type}")
    print(f"# Machine Type: {machine_type}")
    print(f"# Processor: {processor}")
    uname_info = platform.uname()
    print("# Using GPU.")
    print(f"# System: {uname_info.system}")
    print(f"# Node Name: {uname_info.node}")
    print(f"# Release: {uname_info.release}")
    print(f"# Version: {uname_info.version}")
    print(f"# Machine: {uname_info.machine}")
    print(f"# Processor: {uname_info.processor}")
    rank = 0
    size = 1

else:
    from mpi4py import MPI
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["XLA_FLAGS"] = ("--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    print("# Using CPU.")
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    hostname = socket.gethostname()
    system_type = platform.system()
    machine_type = platform.machine()
    processor = platform.processor()
    print(f"# Hostname: {hostname}")
    print(f"# System Type: {system_type}")
    print(f"# Machine Type: {machine_type}")
    print(f"# Processor: {processor}")

def _prep_afqmc(options=None):
    if rank == 0:
        print(f"# Number of MPI ranks: {size}\n#")

    with h5py.File("FCIDUMP_chol", "r") as fh5:
        [nelec, nmo, ms, nchol] = fh5["header"]
        h0 = jnp.array(fh5.get("energy_core"))
        h1 = jnp.array(fh5.get("hcore")).reshape(nmo, nmo)
        chol = jnp.array(fh5.get("chol")).reshape(-1, nmo, nmo)

    assert type(ms) is np.int64
    assert type(nelec) is np.int64
    assert type(nmo) is np.int64
    assert type(nchol) is np.int64
    ms, nelec, nmo, nchol = int(ms), int(nelec), int(nmo), int(nchol)

    nelec_sp = ((nelec + abs(ms)) // 2, (nelec - abs(ms)) // 2)

    norb = nmo

    if options is None:
        try:
            with open("options.bin", "rb") as f:
                options = pickle.load(f)
        except:
            options = {}

    options["dt"] = options.get("dt", 0.01)
    options["n_walkers"] = options.get("n_walkers", 50)
    options["n_prop_steps"] = options.get("n_prop_steps", 50)
    options["n_ene_blocks"] = options.get("n_ene_blocks", 50)
    options["n_sr_blocks"] = options.get("n_sr_blocks", 1)
    options["n_blocks"] = options.get("n_blocks", 50)
    options["seed"] = options.get("seed", np.random.randint(1, int(1e6)))
    options["n_eql"] = options.get("n_eql", 1)
    options["ad_mode"] = options.get("ad_mode", None)
    assert options["ad_mode"] in [None, "forward", "reverse", "2rdm"]
    options["orbital_rotation"] = options.get("orbital_rotation", True)
    options["do_sr"] = options.get("do_sr", True)
    options["walker_type"] = options.get("walker_type", "rhf")
    options["symmetry"] = options.get("symmetry", False)
    options["save_walkers"] = options.get("save_walkers", False)
    options["trial"] = options.get("trial", None)
    options['prjlo'] = options.get('prjlo',None)
    options["orbE"] = options.get("orbE",0)
    options['maxError'] = options.get('maxError',1e-3)
    if options["trial"] is None:
        if rank == 0:
            print(f"# No trial specified in options.")
    options["ene0"] = options.get("ene0", 0.0)
    options["free_projection"] = options.get("free_projection", False)
    options["n_batch"] = options.get("n_batch", 1)
    options["LNO"] = options.get("LNO",True)
    options['prjlo'] = options.get('prjlo',None)
    options["orbE"] = options.get("orbE",0)
    options['maxError'] = options.get('maxError',1e-3)

    try:
        with h5py.File("observable.h5", "r") as fh5:
            [observable_constant] = fh5["constant"]
            observable_op = np.array(fh5.get("op")).reshape(nmo, nmo)
            if options["walker_type"] == "uhf":
                observable_op = jnp.array([observable_op, observable_op])
            observable = [observable_op, observable_constant]
    except:
        observable = None

    ham = hamiltonian.hamiltonian(nmo)
    ham_data = {}
    ham_data["h0"] = h0
    ham_data["h1"] = jnp.array([h1, h1])
    ham_data["chol"] = chol.reshape(nchol, -1)
    ham_data["ene0"] = options["ene0"]

    wave_data = {}
    wave_data["prjlo"] = options["prjlo"]
    mo_coeff = jnp.array(np.load("mo_coeff.npz")["mo_coeff"])
    wave_data["rdm1"] = jnp.array(
        [
            mo_coeff[0][:, : nelec_sp[0]] @ mo_coeff[0][:, : nelec_sp[0]].T,
            mo_coeff[1][:, : nelec_sp[1]] @ mo_coeff[1][:, : nelec_sp[1]].T,
        ]
    )

    if options["trial"] == "rhf":
        trial = wavefunctions.rhf(norb, nelec_sp)
        wave_data["mo_coeff"] = mo_coeff[0][:, : nelec_sp[0]]
    elif options["trial"] == "uhf":
        trial = wavefunctions.uhf(norb, nelec_sp)
        wave_data["mo_coeff"] = [
            mo_coeff[0][:, : nelec_sp[0]],
            mo_coeff[1][:, : nelec_sp[1]],
        ]
    elif options["trial"] == "noci":
        with open("dets.pkl", "rb") as f:
            ci_coeffs_dets = pickle.load(f)
        ci_coeffs_dets = [
            jnp.array(ci_coeffs_dets[0]),
            [jnp.array(ci_coeffs_dets[1][0]), jnp.array(ci_coeffs_dets[1][1])],
        ]
        wave_data["ci_coeffs_dets"] = ci_coeffs_dets
        trial = wavefunctions.noci(norb, nelec_sp, ci_coeffs_dets[0].size)
    else:
        try:
            with open("trial.pkl", "rb") as f:
                [trial, trial_wave_data] = pickle.load(f)
            wave_data.update(trial_wave_data)
            if rank == 0:
                print(f"# Read trial of type {type(trial).__name__} from trial.pkl.")
        except:
            if rank == 0:
                print(
                    "# trial.pkl not found, make sure to construct the trial separately."
                )
            trial = None

    if abs(ms) != 0:
        assert (
            options["walker_type"] != "rhf" or type(trial).__name__ == "UCISD"
        ), "Open shell systems have to use UHF walkers and non-RHF trials."


    if options["walker_type"] == "rhf":
        if options["symmetry"]:
            ham_data["mask"] = jnp.where(jnp.abs(ham_data["h1"]) > 1.0e-10, 1.0, 0.0)
        else:
            ham_data["mask"] = jnp.ones(ham_data["h1"].shape)

        prop = propagation.propagator_restricted(options["dt"], options["n_walkers"])

    elif options["walker_type"] == "uhf":
        if options["symmetry"]:
            ham_data["mask"] = jnp.where(jnp.abs(ham_data["h1"]) > 1.0e-10, 1.0, 0.0)
        else:
            ham_data["mask"] = jnp.ones(ham_data["h1"].shape)
        if options["free_projection"]:
            prop = propagation.propagator_unrestricted(
                options["dt"],
                options["n_walkers"],
                10,
            )
        else:
            prop = propagation.propagator_unrestricted(
                options["dt"],
                options["n_walkers"],
            )

    sampler = sampling.sampler(
        options["n_prop_steps"],
        options["n_ene_blocks"],
        options["n_sr_blocks"],
        options["n_blocks"],
    )

    if rank == 0:
        print(f"# norb: {norb}")
        print(f"# nelec: {nelec_sp}")
        print("#")
        for op in options:
            if options[op] is not None:
                print(f"# {op}: {options[op]}")
        print("#")
    #import pdb;pdb.set_trace()
    return ham_data, ham, prop, trial, wave_data, sampler, observable, options


if __name__ == "__main__":
    ham_data, ham, prop, trial, wave_data, sampler, observable, options = _prep_afqmc()
    init = time.time()
    # comm.Barrier()
    e_afqmc, err_afqmc = 0.0, 0.0
    if options["free_projection"]:
        driver.fp_afqmc(
            ham_data, ham, prop, trial, wave_data, sampler, observable, options
        )
    else:
        e_afqmc, err_afqmc = driver.LNOafqmc(
            ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI
        )
    # comm.Barrier()
    end = time.time()
    if rank == 0:
        print(f"ph_afqmc walltime: {end - init}", flush=True)
        np.savetxt("ene_err.txt", np.array([e_afqmc, err_afqmc]))
    # comm.Barrier()
