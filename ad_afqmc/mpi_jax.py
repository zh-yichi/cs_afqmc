import argparse
import pickle
import time

import h5py
import numpy as np

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

from functools import partial

from jax import numpy as jnp

from ad_afqmc import driver, hamiltonian, propagation, sampling, wavefunctions, run_afqmc

print = partial(print, flush=True)

mo_file = run_afqmc.mo_file
amp_file = run_afqmc.amp_file
chol_file = run_afqmc.chol_file

def _prep_afqmc(options=None,option_file="options.bin",
                mo_file=mo_file,amp_file=amp_file,chol_file=chol_file):
    
    with h5py.File(chol_file, "r") as fh5:
        [nelec, nmo, ms, nchol] = fh5["header"]
        h0 = jnp.array(fh5.get("energy_core"))
        h1 = jnp.array(fh5.get("hcore")).reshape(nmo, nmo)
        chol = jnp.array(fh5.get("chol")).reshape(-1, nmo, nmo)
        h1_mod = jnp.array(fh5.get("hcore_mod")).reshape(nmo, nmo)

    assert type(ms) is np.int64
    assert type(nelec) is np.int64
    assert type(nmo) is np.int64
    assert type(nchol) is np.int64
    ms, nelec, nmo, nchol = int(ms), int(nelec), int(nmo), int(nchol)
    nelec_sp = ((nelec + abs(ms)) // 2, (nelec - abs(ms)) // 2)

    norb = nmo

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
    options["ad_mode"] = options.get("ad_mode", None)
    assert options["ad_mode"] in [None, "forward", "reverse", "2rdm"]
    options["orbital_rotation"] = options.get("orbital_rotation", True)
    options["do_sr"] = options.get("do_sr", True)
    options["walker_type"] = options.get("walker_type", "rhf")
    options["symmetry"] = options.get("symmetry", False)
    options["save_walkers"] = options.get("save_walkers", False)
    options["trial"] = options.get("trial", None)
    if options["trial"] is None:
        if rank == 0:
            print(f"# No trial specified in options.")
    options["ene0"] = options.get("ene0", 0.0)
    options["free_projection"] = options.get("free_projection", False)
    options["n_batch"] = options.get("n_batch", 1)
    options["LNO"] = options.get("LNO",False)
    if options["LNO"]:
        if rank == 0:
            print("# Using Local Natural Orbital Approximation")
    options['prjlo'] = options.get('prjlo',None)
    options["orbE"] = options.get("orbE",0)
    options['maxError'] = options.get('maxError',1e-3)

    if options['use_gpu']:
        config.afqmc_config["use_gpu"] = True

    # config.setup_jax()
    # MPI = config.setup_comm()
    # comm = MPI.COMM_WORLD
    # size = comm.Get_size()
    # rank = comm.Get_rank()
    if rank == 0:
        print(f"# Number of MPI ranks: {size}\n#")

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
    mo_coeff = jnp.array(np.load(mo_file)["mo_coeff"])
    # mo_coeff = jnp.array([np.eye(norb),np.eye(norb)])
    wave_data["rdm1"] = jnp.array(
        [
            mo_coeff[0][:, : nelec_sp[0]] @ mo_coeff[0][:, : nelec_sp[0]].T,
            mo_coeff[1][:, : nelec_sp[1]] @ mo_coeff[1][:, : nelec_sp[1]].T,
        ]
    )

    if options["trial"] == "rhf":
        trial = wavefunctions.rhf(norb, nelec_sp, n_batch=options["n_batch"])
        wave_data["mo_coeff"] = mo_coeff[0][:, : nelec_sp[0]]
    elif options["trial"] == "uhf":
        trial = wavefunctions.uhf(norb, nelec_sp, n_batch=options["n_batch"])
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
        trial = wavefunctions.noci(
            norb, nelec_sp, ci_coeffs_dets[0].size, n_batch=options["n_batch"]
        )
    elif options["trial"] == "cisd":
        try:
            amplitudes = np.load(amp_file)
            ci1 = jnp.array(amplitudes["ci1"])
            ci2 = jnp.array(amplitudes["ci2"])
            trial_wave_data = {"ci1": ci1, "ci2": ci2}
            wave_data.update(trial_wave_data)
            trial = wavefunctions.cisd(norb, nelec_sp, n_batch=options["n_batch"])
        except:
            raise ValueError("Trial specified as cisd, but amplitudes.npz not found.")
    elif options["trial"] == "ucisd":
        try:
            amplitudes = np.load(amp_file)
            ci1a = jnp.array(amplitudes["ci1a"])
            ci1b = jnp.array(amplitudes["ci1b"])
            ci2aa = jnp.array(amplitudes["ci2aa"])
            ci2ab = jnp.array(amplitudes["ci2ab"])
            ci2bb = jnp.array(amplitudes["ci2bb"])
            trial_wave_data = {
                "ci1A": ci1a,
                "ci1B": ci1b,
                "ci2AA": ci2aa,
                "ci2AB": ci2ab,
                "ci2BB": ci2bb,
                "mo_coeff": mo_coeff,
            }
            wave_data.update(trial_wave_data)
            trial = wavefunctions.ucisd(norb, nelec_sp, n_batch=options["n_batch"])
        except:
            raise ValueError("Trial specified as ucisd, but amplitudes.npz not found.")
    elif options["trial"] == "ccsd_pt":
        amplitudes = np.load(amp_file)
        t1 = jnp.array(amplitudes["t1"])
        t2 = jnp.array(amplitudes["t2"])
        trial_wave_data = {"t1": t1, "t2": t2}
        wave_data.update(trial_wave_data)
        trial = wavefunctions.ccsd_pt(norb, nelec_sp, n_batch=options["n_batch"])
        wave_data["mo_coeff"] = np.eye(norb)[:,:nelec_sp[0]]
    elif options["trial"] == "ccsd_hf":
        amplitudes = np.load(amp_file)
        t1 = jnp.array(amplitudes["t1"])
        t2 = jnp.array(amplitudes["t2"])
        trial_wave_data = {"t1": t1, "t2": t2}
        wave_data.update(trial_wave_data)
        trial = wavefunctions.ccsd_hf(norb, nelec_sp, n_batch=options["n_batch"])
        wave_data["mo_coeff"] = np.eye(norb)[:,:nelec_sp[0]]
    elif options["trial"] == "ccsd_pt2":
        amplitudes = np.load(amp_file)
        t1 = jnp.array(amplitudes["t1"])
        t2 = jnp.array(amplitudes["t2"])
        trial_wave_data = {"t1": t1, "t2": t2}
        wave_data.update(trial_wave_data)
        trial = wavefunctions.ccsd_pt2(norb, nelec_sp, n_batch=options["n_batch"])
        wave_data["mo_coeff"] = np.eye(norb)[:,:nelec_sp[0]]
    elif options["trial"] == "ccsd_pt_ad":
        ham_data['h1_mod'] = h1_mod
        amplitudes = np.load(amp_file)
        t1 = jnp.array(amplitudes["t1"])
        t2 = jnp.array(amplitudes["t2"])
        trial_wave_data = {"t1": t1, "t2": t2}
        wave_data.update(trial_wave_data)
        wave_data["mo_coeff"] = np.eye(norb)[:,:nelec_sp[0]]
        trial = wavefunctions.ccsd_pt_ad(norb, nelec_sp, n_batch=options["n_batch"])
    # elif options["trial"] == "ccsd_pt2_ad":
    #     ham_data['h1_mod'] = h1_mod
    #     amplitudes = np.load(amp_file)
    #     t1 = jnp.array(amplitudes["t1"])
    #     t2 = jnp.array(amplitudes["t2"])
    #     trial_wave_data = {"t1": t1, "t2": t2}
    #     wave_data.update(trial_wave_data)
    #     trial = wavefunctions.ccsd_pt2_ad(norb, nelec_sp, n_batch=options["n_batch"])
    #     nocc = nelec_sp[0]
    #     trial_mo = trial.thouless_trans(t1)
    #     # trial_mo = np.eye(norb)[:,:nocc]
    #     wave_data['mo_coeff'] = trial_mo[:,:nocc]
    #     rot_t2 = jnp.einsum('il,jk,lakb->iajb',trial_mo[:nocc,:nocc].T,
    #                trial_mo[:nocc,:nocc].T,t2)
    #     wave_data['rot_t2'] = rot_t2
    elif options["trial"] == "ccsd_pt2_ad":
        ham_data['h1_mod'] = h1_mod
        amplitudes = np.load(amp_file)
        t1 = jnp.array(amplitudes["t1"])
        t2 = jnp.array(amplitudes["t2"])
        trial_wave_data = {"t1": t1, "t2": t2}
        wave_data.update(trial_wave_data)
        trial = wavefunctions.ccsd_pt2_ad(
            norb, nelec_sp, n_batch=options["n_batch"])
        nocc = nelec_sp[0]
        mo_t = trial.thouless_trans(t1)[:,:nocc]
        wave_data['mo_t'] = mo_t
        wave_data['mo_coeff'] = np.eye(norb)[:,:nocc]
        rot_t2 = jnp.einsum('il,jk,lakb->iajb',mo_t[:nocc,:nocc].T,
                   mo_t[:nocc,:nocc].T,t2)
        wave_data['rot_t2'] = rot_t2
    elif options["trial"] == "uccsd_pt_ad":
        trial = wavefunctions.uccsd_pt_ad(
            norb, nelec_sp, n_batch=options["n_batch"])
        wave_data["mo_coeff"] = [
            mo_coeff[0][:, : nelec_sp[0]],
            mo_coeff[1][:, : nelec_sp[1]],
        ]
        wave_data["mo_B"] = mo_coeff[1]
        ham_data['h1_mod'] = h1_mod
        amplitudes = np.load(amp_file)
        t1a = jnp.array(amplitudes["t1a"])
        t1b = jnp.array(amplitudes["t1b"])
        t2aa = jnp.array(amplitudes["t2aa"])
        t2ab = jnp.array(amplitudes["t2ab"])
        t2bb = jnp.array(amplitudes["t2bb"])
        mo_a_A = wave_data['mo_coeff'][0]
        mo_b_B = wave_data["mo_B"].T @ wave_data['mo_coeff'][1]
        noccA, noccB = trial.nelec[0], trial.nelec[1]
        wave_data["rot_t1A"] = mo_a_A[:noccA,:noccA].T @ t1a
        wave_data["rot_t2AA"] = jnp.einsum('ik,jl,kalb->iajb',
            mo_a_A[:noccA,:noccA].T,mo_a_A[:noccA,:noccA].T,t2aa)
        wave_data["rot_t1B"] = mo_b_B[:noccB,:noccB].T @ t1b
        wave_data["rot_t2BB"] = jnp.einsum('ik,jl,kalb->iajb',
            mo_b_B[:noccB,:noccB].T,mo_b_B[:noccB,:noccB].T,t2bb)
        wave_data["rot_t2AB"] = jnp.einsum('ik,jl,kalb->iajb',
            mo_a_A[:noccA,:noccA].T,mo_b_B[:noccB,:noccB].T,t2ab)
    elif options["trial"] == "uccsd_pt2_ad":
        trial = wavefunctions.uccsd_pt2_ad(
            norb, nelec_sp, n_batch=options["n_batch"])
        noccA, noccB = trial.nelec[0], trial.nelec[1]
        wave_data["mo_coeff"] = [
            mo_coeff[0][:, : noccA],
            mo_coeff[1][:, : noccB],
        ]
        wave_data["mo_A2B"] = mo_coeff[1].T # <B_p|A_q>
        ham_data['h1_mod'] = h1_mod
        amplitudes = np.load(amp_file)
        t1a = jnp.array(amplitudes["t1a"])
        t1b = jnp.array(amplitudes["t1b"])
        t2aa = jnp.array(amplitudes["t2aa"])
        t2ab = jnp.array(amplitudes["t2ab"])
        t2bb = jnp.array(amplitudes["t2bb"])
        mo_ta = trial.thouless_trans(t1a)[:,:noccA]
        mo_tb = trial.thouless_trans(t1b)[:,:noccB]
        wave_data['mo_ta'] = mo_ta
        wave_data['mo_tb'] = mo_tb
        wave_data['mo_tb_A'] = wave_data["mo_A2B"].T @ mo_tb
        wave_data["rot_t2AA"] = jnp.einsum('ik,jl,kalb->iajb',
            mo_ta[:noccA,:noccA].T,mo_ta[:noccA,:noccA].T,t2aa)
        wave_data["rot_t2BB"] = jnp.einsum('ik,jl,kalb->iajb',
            mo_tb[:noccB,:noccB].T,mo_tb[:noccB,:noccB].T,t2bb)
        wave_data["rot_t2AB"] = jnp.einsum('ik,jl,kalb->iajb',
            mo_ta[:noccA,:noccA].T,mo_tb[:noccB,:noccB].T,t2ab)
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

    if options["walker_type"] == "rhf":
        if options["symmetry"]:
            ham_data["mask"] = jnp.where(jnp.abs(ham_data["h1"]) > 1.0e-10, 1.0, 0.0)
        else:
            ham_data["mask"] = jnp.ones(ham_data["h1"].shape)
        #print(f'using {options["n_exp_terms"]} exp_terms')
        prop = propagation.propagator_restricted(
            options["dt"], 
            options["n_walkers"], 
            options["n_exp_terms"],
            options["n_batch"]
        )

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
                n_batch=options["n_batch"],
            )
        else:
            prop = propagation.propagator_unrestricted(
                options["dt"],
                options["n_walkers"],
                n_batch=options["n_batch"],
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

    return ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI


if __name__ == "__main__":
    ham_data, ham, prop, trial, wave_data, sampler, observable, options, _ = (
        _prep_afqmc()
    )

    assert trial is not None
    init = time.time()
    comm.Barrier()
    e_afqmc, err_afqmc = 0.0, 0.0
    if options["free_projection"]:
        driver.fp_afqmc(
        ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI
        )
    else:
        if options["LNO"]:
            e_afqmc, err_afqmc = driver.LNOafqmc(
            ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI
        )
        else:
            e_afqmc, err_afqmc = driver.afqmc(
            ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI
            )
    comm.Barrier()
    end = time.time()
    if rank == 0:
        print(f"ph_afqmc walltime: {end - init}", flush=True)
        np.savetxt("ene_err.txt", np.array([e_afqmc, err_afqmc]))
    comm.Barrier()
