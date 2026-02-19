import pickle
import h5py
import numpy as np
import jax.numpy as jnp

# from ad_afqmc import config
from ad_afqmc import hamiltonian
# from ad_afqmc import pyscf_interface
from ad_afqmc.prop_unrestricted import propagation
from ad_afqmc.prop_unrestricted.mixed_wave import wavefunctions_restricted
from ad_afqmc.prop_unrestricted.mixed_wave import sampling
# from ad_afqmc.prop_unrestricted.mixed_wave import wavefunctions_unrestricted
from ad_afqmc.prop_unrestricted.prep import prep_afqmc
from functools import partial
print = partial(print, flush=True)

prep_afqmc = prep_afqmc

def _prep_afqmc(options=None,
                option_file="options.bin",
                amp_file="amplitudes.npz",
                chol_file="FCIDUMP_chol"
                ):
    
    if options is None:
        try:
            with open(option_file, "rb") as f:
                options = pickle.load(f)
        except:
            options = {}

    options["dt"] = options.get("dt", 0.01)
    options["n_exp_terms"] = options.get("n_exp_terms",6)
    options["n_walkers"] = options.get("n_walkers", 50)
    options["n_prop_steps"] = options.get("n_prop_steps", 50)
    options["n_ene_blocks"] = options.get("n_ene_blocks", 50)
    options["n_sr_blocks"] = options.get("n_sr_blocks", 1)
    options["n_blocks"] = options.get("n_blocks", 50)
    options["seed"] = options.get("seed", np.random.randint(1, int(1e6)))
    options["n_eql"] = options.get("n_eql", 1)
    options["walker_type"] = options.get("walker_type", "rhf")
    # options["symmetry"] = options.get("symmetry", False)
    # options["save_walkers"] = options.get("save_walkers", False)
    options["trial"] = options.get("trial", None)
    # options["free_projection"] = options.get("free_projection", False)
    # options["fp_abs"] = options.get("fp_abs", False)
    options["n_batch"] = options.get("n_batch", 1)
    options["max_error"] = options.get("max_error", 1e-3)

    if 'u' not in options['trial']:
        spin_type = 'restricted'
    elif 'u' in options['trial']:
        spin_type = 'unrestricted'

    if spin_type == 'restricted':
        with h5py.File(chol_file, "r") as fh5:
            [nelec, nmo, ms, nchol] = fh5["header"]
            h0 = jnp.array(fh5.get("energy_core"))
            h1 = jnp.array(fh5.get("hcore")).reshape(nmo, nmo)
            chol = jnp.array(fh5.get("chol")).reshape(-1, nmo, nmo)
            h1_mod = jnp.array(fh5.get("hcore_mod")).reshape(nmo, nmo)
    elif spin_type == 'unrestricted':
        with h5py.File(chol_file, "r") as fh5:
            [nelec, nmo, ms, nchol] = fh5["header"]
            h0 = jnp.array(fh5.get("energy_core"))
            h1 = jnp.array(fh5.get("hcore")).reshape(2, nmo, nmo)
            chol = jnp.array(fh5.get("chol")).reshape(2, -1, nmo, nmo)
            h1_mod = jnp.array(fh5.get("hcore_mod")).reshape(2, nmo, nmo)

    assert type(ms) is np.int64
    assert type(nelec) is np.int64
    assert type(nmo) is np.int64
    assert type(nchol) is np.int64
    ms, nelec, nmo, nchol = int(ms), int(nelec), int(nmo), int(nchol)
    nelec_sp = ((nelec + abs(ms)) // 2, (nelec - abs(ms)) // 2)
    norb = nmo

    ham = hamiltonian.hamiltonian(nmo)
    ham_data = {}
    ham_data["h0"] = h0

    if spin_type == 'restricted':
        ham_data["h1"] = jnp.array([h1, h1])
        ham_data["h1_mod"] = jnp.array(h1_mod)
        nchol = chol.shape[0]
        ham_data["chol"] = jnp.array(chol.reshape(chol.shape[0], -1))
    elif spin_type == 'unrestricted':
        ham_data["h1"] = jnp.array(h1)
        ham_data["h1_mod"] = jnp.array(h1_mod)
        nchol = chol[0].shape[0]
        ham_data["chol"] = jnp.array([chol[0].reshape(chol[0].shape[0], -1),
                                      chol[1].reshape(chol[1].shape[0], -1)])

    wave_data = {}
    mo_coeff = jnp.array([np.eye(norb),np.eye(norb)])

    # if options["trial"] == "rhf":
    #     trial = wavefunctions_restricted.rhf(norb, nelec_sp, n_batch=options["n_batch"])
    #     wave_data["mo_coeff"] = mo_coeff[0][:, : nelec_sp[0]]

    if options["trial"] == "stoccsd2":
        trial = wavefunctions_restricted.stoccsd2(
            norb,
            nelec_sp,
            n_batch = options["n_batch"],
            nslater = options['nslater']
            )
        sampler = sampling.sampler_stoccsd2(
            options["n_prop_steps"],
            options["n_ene_blocks"],
            options["n_sr_blocks"],
            options["n_blocks"],
            nchol,
            )
        nocc = nelec_sp[0]
        amplitudes = np.load(amp_file)
        t1 = jnp.array(amplitudes["t1"])
        t2 = jnp.array(amplitudes["t2"])
        trial_wave_data = {"t1": t1, "t2": t2}
        wave_data.update(trial_wave_data)
        init_sd = jnp.eye(norb)[:,:nocc]
        mo_t = trial._thouless(init_sd, t1)
        wave_data['mo_t'] = mo_t
        wave_data["mo_coeff"] = mo_coeff[0][:,:nelec_sp[0]]

    if options["walker_type"] == "rhf":
        # if options["symmetry"]:
        #     ham_data["mask"] = jnp.where(jnp.abs(ham_data["h1"]) > 1.0e-10, 1.0, 0.0)
        # else:
        #     ham_data["mask"] = jnp.ones(ham_data["h1"].shape)
        prop = propagation.propagator_restricted(
            options["dt"], 
            options["n_walkers"], 
            options["n_exp_terms"],
            options["n_batch"]
        )

    elif options["walker_type"] == "uhf":
        # if options["symmetry"]:
        #     ham_data["mask"] = jnp.where(jnp.abs(ham_data["h1"]) > 1.0e-10, 1.0, 0.0)
        # else:
        #     ham_data["mask"] = jnp.ones(ham_data["h1"].shape)

        # if options["free_projection"]:
        #     prop = propagation.propagator_unrestricted(
        #         options["dt"],
        #         options["n_walkers"],
        #         10,
        #         n_batch=options["n_batch"],
        #     )
        # else:
        prop = propagation.propagator_unrestricted(
            options["dt"],
            options["n_walkers"],
            n_batch=options["n_batch"],
        )

    print(f"# norb: {norb}")
    print(f"# nelec: {nelec_sp}")
    for op in options:
        if options[op] is not None:
            print(f"# {op}: {options[op]}")

    return ham_data, ham, prop, trial, wave_data, sampler, options