from jax import numpy as jnp
from functools import partial

from functools import partial
from typing import Optional, Union
import h5py
import jax.numpy as jnp
import numpy as np
from pyscf import __config__, mcscf, scf
from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD
from ad_afqmc import pyscf_interface

def prep_afqmc(
    mf_or_cc: Union[scf.uhf.UHF, scf.rhf.RHF, CCSD, UCCSD],
    options: dict,
    basis_coeff: Optional[np.ndarray] = None,
    norb_frozen: int = 0,
    chol_cut: float = 1e-5,
    amp_file = "amplitudes.npz",
    chol_file = "FCIDUMP_chol"
):
    """Prepare AFQMC calculation with mean field trial wavefunction. Writes integrals and mo coefficients to disk.

    Args:
        mf (Union[scf.uhf.UHF, scf.rhf.RHF, mcscf.mc1step.CASSCF]): pyscf mean field object. Used for generating integrals (if not provided) and trial.
        basis_coeff (np.ndarray, optional): Orthonormal basis used for afqmc, given in the basis of ao's. If not provided mo_coeff of mf is used as the basis.
        norb_frozen (int, optional): Number of frozen orbitals. Not supported for custom integrals.
        chol_cut (float, optional): Cholesky decomposition cutoff.
        integrals (dict, optional): Dictionary of integrals in an orthonormal basis, {"h0": enuc, "h1": h1e, "h2": eri}.
    """

    print("#\n# Preparing AFQMC calculation")

    trial = options['trial']

    if isinstance(mf_or_cc, (CCSD, UCCSD)):
        # raise warning about mpi
        print(
            "# If you import pyscf cc modules and use MPI for AFQMC in the same script, finalize MPI before calling the AFQMC driver."
        )
        mf = mf_or_cc._scf
        cc = mf_or_cc
        if cc.frozen is not None:
            norb_frozen = cc.frozen
        
        # if 'ci' in trial.lower():
        # # build ci from cc #
        #     if isinstance(cc, UCCSD):
        #         ci2aa = cc.t2[0] + 2 * np.einsum("ia,jb->ijab", cc.t1[0], cc.t1[0])
        #         ci2aa = (ci2aa - ci2aa.transpose(0, 1, 3, 2)) / 2
        #         ci2aa = ci2aa.transpose(0, 2, 1, 3)
        #         ci2bb = cc.t2[2] + 2 * np.einsum("ia,jb->ijab", cc.t1[1], cc.t1[1])
        #         ci2bb = (ci2bb - ci2bb.transpose(0, 1, 3, 2)) / 2
        #         ci2bb = ci2bb.transpose(0, 2, 1, 3)
        #         ci2ab = cc.t2[1] + np.einsum("ia,jb->ijab", cc.t1[0], cc.t1[1])
        #         ci2ab = ci2ab.transpose(0, 2, 1, 3)
        #         ci1a = np.array(cc.t1[0])
        #         ci1b = np.array(cc.t1[1])
                
        #         np.savez(
        #             amp_file,
        #             ci1a=ci1a,
        #             ci1b=ci1b,
        #             ci2aa=ci2aa,
        #             ci2ab=ci2ab,
        #             ci2bb=ci2bb,
        #         )
            # else:
            #     ci2 = cc.t2 + np.einsum("ia,jb->ijab", np.array(cc.t1), np.array(cc.t1))
            #     ci2 = ci2.transpose(0, 2, 1, 3)
            #     ci1 = np.array(cc.t1)
            #     np.savez(amp_file, ci1=ci1, ci2=ci2)
        
        # elif 'cc' in trial.lower():
        # ccsd trial #
        if isinstance(cc, UCCSD):
            t1a = np.array(cc.t1[0])
            t1b = np.array(cc.t1[1])
            t2aa, t2ab, t2bb = cc.t2
            t2aa = (t2aa - t2aa.transpose(0, 1, 3, 2)) / 2
            t2bb = (t2bb - t2bb.transpose(0, 1, 3, 2)) / 2
            t2aa = t2aa.transpose(0, 2, 1, 3)
            t2bb = t2bb.transpose(0, 2, 1, 3)
            t2ab = t2ab.transpose(0, 2, 1, 3)
            np.savez(
                amp_file,
                t1a=t1a,
                t1b=t1b,
                t2aa=t2aa,
                t2ab=t2ab,
                t2bb=t2bb,
            )
        elif isinstance(cc, CCSD):
            t2 = cc.t2
            t2 = t2.transpose(0, 2, 1, 3)
            t1 = np.array(cc.t1)
            np.savez(amp_file, t1=t1, t2=t2)
    else:
        mf = mf_or_cc

    mol = mf.mol
    # choose the orbital basis
    if basis_coeff is None:
        if isinstance(mf, scf.uhf.UHF):
            basis_coeff = mf.mo_coeff[0]
        else:
            basis_coeff = mf.mo_coeff

    # calculate cholesky integrals
    print("# Calculating Cholesky integrals")
    
    DFbas = None
    if getattr(mf, "with_df", None) is not None:
        print('# Decomposing ERI with DF')
        DFbas = mf.with_df.auxmol.basis 

    # assert norb_frozen * 2 < sum(
    #     nelec
    # ), "Frozen orbitals exceed number of electrons"
    if 'u' not in trial.lower():
        mc = mcscf.CASSCF(
            mf, mol.nao - norb_frozen, mol.nelectron - 2 * norb_frozen
        )
        nelec = mc.nelecas
        mc.mo_coeff = basis_coeff
        h1e, enuc = mc.get_h1eff()
        _, chol, _, _ = \
            pyscf_interface.generate_integrals(
            mol, mf.get_hcore(), basis_coeff, chol_cut, DFbas=DFbas)
        nao = mf.mol.nao
        chol = chol.reshape((-1, nao, nao))
        chol = chol[:, mc.ncore : mc.ncore + mc.ncas, 
                    mc.ncore : mc.ncore + mc.ncas]
        nbasis = mc.ncas
        print("# Finished calculating Cholesky integrals\n#")
        print("# Size of the correlation space:")
        print(f"# Number of electrons: {nelec}")
        print(f"# Number of basis functions: {nbasis}")
        print(f"# Number of Cholesky vectors: {chol.shape[0]}\n#")
        chol = chol.reshape((-1, nbasis, nbasis))
        v0 = 0.5 * jnp.einsum("nik,njk->ij", chol, chol, optimize="optimal")
        h1e_mod = h1e - v0
        chol = chol.reshape((chol.shape[0], -1))
            
    elif 'u' in trial.lower():
        mc = mcscf.UCASSCF(
            mf, mol.nao - norb_frozen,
            mol.nelectron - 2 * norb_frozen)
        nelec = mc.nelecas
        mc.mo_coeff = mf.mo_coeff
        h1e, enuc = mc.get_h1eff()

        _, chol_a, _, _ = pyscf_interface.generate_integrals(
            mol, mf.get_hcore(), mf.mo_coeff[0], chol_cut, DFbas=DFbas
        )
        # _, chol_b, _, _ = pyscf_interface.generate_integrals(
        #     mol, mf.get_hcore(), mf.mo_coeff[1], chol_cut, DFbas=DFbas
        # )
        nbasis = mc.ncas
        nao = mf.mol.nao
        chol_a = chol_a.reshape((-1, nao, nao))
        # a2b = <B|A>
        # <B|L|B> = <B|A><A|L|A><A|B>
        # if separate build chol might not be the same number
        s1e = mf.get_ovlp()
        a2b = mf.mo_coeff[1].T @ s1e @ mf.mo_coeff[0]
        chol_b = jnp.einsum('pr,grs,sq->gpq',a2b,chol_a,a2b.T)
        chol_a = chol_a[:, mc.ncore[0] : mc.ncore[0] + mc.ncas,
                        mc.ncore[0] : mc.ncore[0] + mc.ncas]
        chol_b = chol_b.reshape((-1, nao, nao))
        chol_b = chol_b[:, mc.ncore[1] : mc.ncore[1] + mc.ncas, 
                        mc.ncore[1] : mc.ncore[1] + mc.ncas]
        v0_a = 0.5 * jnp.einsum("nik,njk->ij", chol_a, chol_a, optimize="optimal")
        v0_b = 0.5 * jnp.einsum("nik,njk->ij", chol_b, chol_b, optimize="optimal")
        h1e = jnp.array(h1e)
        h1e_mod = jnp.array(h1e - jnp.array([v0_a,v0_b]))

        chol = jnp.array(
            [chol_a.reshape(chol_a.shape[0], -1),
             chol_b.reshape(chol_b.shape[0], -1)])
        print("# Finished calculating Cholesky integrals\n#")
        print("# Size of the correlation space:")
        print(f"# Number of electrons: {nelec}")
        print(f"# Number of basis functions: {nbasis}")
        print(f"# Number of Cholesky vectors: {chol.shape[-2]}\n#")

    
    pyscf_interface.write_dqmc(
        h1e,
        h1e_mod,
        chol,
        sum(nelec),
        nbasis,
        enuc,
        ms=mol.spin,
        filename=chol_file,
    )

import pickle
import h5py
import numpy as np
from ad_afqmc import config
from functools import partial
from ad_afqmc import hamiltonian
from ad_afqmc.prop_unrestricted import propagation, sampling
from ad_afqmc.prop_unrestricted import wavefunctions, wavefunctions_restricted

print = partial(print, flush=True)

def _prep_afqmc(options=None,
                option_file="options.bin",
                amp_file="amplitudes.npz",
                chol_file="FCIDUMP_chol"):
    
    if options is None:
        try:
            with open(option_file, "rb") as f:
                options = pickle.load(f)
        except:
            options = {}
    
    trial = options['trial']
    if 'u' not in trial.lower():
        with h5py.File(chol_file, "r") as fh5:
            [nelec, nmo, ms, nchol] = fh5["header"]
            h0 = jnp.array(fh5.get("energy_core"))
            h1 = jnp.array(fh5.get("hcore")).reshape(nmo, nmo)
            chol = jnp.array(fh5.get("chol")).reshape(-1, nmo, nmo)
            h1_mod = jnp.array(fh5.get("hcore_mod")).reshape(nmo, nmo)
    elif 'u' in trial.lower():
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
    options["ene0"] = options.get("ene0", 0.0)
    options["free_projection"] = options.get("free_projection", False)
    options["n_batch"] = options.get("n_batch", 1)
    options["LNO"] = options.get("LNO",False)
    options['prjlo'] = options.get('prjlo',None)
    options["orbE"] = options.get("orbE",0)
    options['maxError'] = options.get('maxError',1e-3)

    if options['use_gpu']:
        config.afqmc_config["use_gpu"] = True

    config.setup_jax()
    MPI = config.setup_comm()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

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

    if 'u' not in trial.lower():
        ham_data["h1"] = jnp.array([h1, h1])
        ham_data["h1_mod"] = jnp.array(h1_mod)
        nchol = chol.shape[0]
        ham_data["chol"] = jnp.array(chol.reshape(chol.shape[0], -1))
    elif 'u' in trial.lower():
        ham_data["h1"] = jnp.array(h1)
        ham_data["h1_mod"] = jnp.array(h1_mod)
        nchol = chol[0].shape[0]
        ham_data["chol"] = jnp.array([chol[0].reshape(chol[0].shape[0], -1),
                                    chol[1].reshape(chol[1].shape[0], -1)])

    ham_data["ene0"] = options["ene0"]

    wave_data = {}
    mo_coeff = jnp.array([np.eye(norb),np.eye(norb)])

    if options["trial"] == "rhf":
        trial = wavefunctions_restricted.rhf(norb, nelec_sp, n_batch=options["n_batch"])
        wave_data["mo_coeff"] = mo_coeff[0][:, : nelec_sp[0]]

    elif options["trial"] == "uhf":
        trial = wavefunctions.uhf(norb, nelec_sp, n_batch=options["n_batch"])
        wave_data["mo_coeff"] = [
            mo_coeff[0][:, : nelec_sp[0]],
            mo_coeff[1][:, : nelec_sp[1]],
        ]

    elif options["trial"] == "cisd":
        try:
            amplitudes = np.load(amp_file)
            t1 = jnp.array(amplitudes["t1"])
            t2 = jnp.array(amplitudes["t2"])
            ci2 = t2 + jnp.einsum("ia,jb->iajb", t1, t1)
            trial_wave_data = {"ci1": t1, "ci2": ci2, 
                               "mo_coeff": mo_coeff[0][:, : nelec_sp[0]]}
            wave_data.update(trial_wave_data)
            trial = wavefunctions_restricted.cisd(norb, nelec_sp, n_batch=options["n_batch"])
            if "pt" in options["trial"]:
                trial = wavefunctions_restricted.cisd_pt(norb, nelec_sp, n_batch=options["n_batch"])
            if "hf1" in options["trial"]:
                trial = wavefunctions_restricted.cisd_hf1(norb, nelec_sp, n_batch=options["n_batch"])
            if "hf2" in options["trial"]:
                trial = wavefunctions_restricted.cisd_hf2(norb, nelec_sp, n_batch=options["n_batch"])
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
        wave_data["mo_coeff"] = mo_coeff[0][:,:nelec_sp[0]]
        trial = wavefunctions.ccsd_pt(norb, nelec_sp, n_batch=options["n_batch"])
        if "ad" in options["trial"]:
            trial = wavefunctions_restricted.ccsd_pt_ad(norb, nelec_sp, n_batch=options["n_batch"])
    elif options["trial"] == "ccsd_pt2":
        nocc = nelec_sp[0]
        amplitudes = np.load(amp_file)
        t1 = jnp.array(amplitudes["t1"])
        t2 = jnp.array(amplitudes["t2"])
        trial_wave_data = {"t1": t1, "t2": t2}
        wave_data.update(trial_wave_data)
        mo_t = trial.thouless_trans(t1)[:,:nocc]
        wave_data['mo_t'] = mo_t
        wave_data["mo_coeff"] = mo_coeff[0][:,:nelec_sp[0]]
        trial = wavefunctions.ccsd_pt2(norb, nelec_sp, n_batch=options["n_batch"])

    elif options["trial"] == "ccsd_pt2_ad":
        trial = wavefunctions.ccsd_pt2_ad(norb, nelec_sp, n_batch=options["n_batch"])
        ham_data['h1_mod'] = h1_mod
        amplitudes = np.load(amp_file)
        t1 = jnp.array(amplitudes["t1"])
        t2 = jnp.array(amplitudes["t2"])
        trial_wave_data = {"t1": t1, "t2": t2}
        wave_data.update(trial_wave_data)
        nocc = nelec_sp[0]
        mo_t = trial.thouless_trans(t1)[:,:nocc]
        wave_data['mo_t'] = mo_t
        wave_data['mo_coeff'] = mo_coeff[0][:,:nocc]
        rot_t2 = jnp.einsum('il,jk,lakb->iajb',
                            mo_t[:nocc,:nocc].T,mo_t[:nocc,:nocc].T,t2)
        wave_data['rot_t2'] = rot_t2

    elif options["trial"] == "uccsd_pt_ad":
        trial = wavefunctions.uccsd_pt_ad(norb, nelec_sp, n_batch=options["n_batch"])
        wave_data["mo_coeff"] = [
            mo_coeff[0][:, : nelec_sp[0]],
            mo_coeff[1][:, : nelec_sp[1]],
        ]
        ham_data['h1_mod'] = h1_mod
        amplitudes = np.load(amp_file)
        t1a = jnp.array(amplitudes["t1a"])
        t1b = jnp.array(amplitudes["t1b"])
        t2aa = jnp.array(amplitudes["t2aa"])
        t2ab = jnp.array(amplitudes["t2ab"])
        t2bb = jnp.array(amplitudes["t2bb"])
        mo_a_A = wave_data['mo_coeff'][0]
        mo_b_B = wave_data['mo_coeff'][1]
        noccA, noccB = trial.nelec[0], trial.nelec[1]
        wave_data["rot_t1A"] = mo_a_A[:noccA,:noccA].T @ t1a
        wave_data["rot_t2AA"] = jnp.einsum('ik,jl,kalb->iajb',
            mo_a_A[:noccA,:noccA].T,mo_a_A[:noccA,:noccA].T,t2aa)
        wave_data["rot_t1B"] = mo_b_B[:noccB,:noccB].T @ t1b
        wave_data["rot_t2BB"] = jnp.einsum('ik,jl,kalb->iajb',
            mo_b_B[:noccB,:noccB].T,mo_b_B[:noccB,:noccB].T,t2bb)
        wave_data["rot_t2AB"] = jnp.einsum('ik,jl,kalb->iajb',
            mo_a_A[:noccA,:noccA].T,mo_b_B[:noccB,:noccB].T,t2ab)
    elif options["trial"] == "uccsd_pt":
        trial = wavefunctions.uccsd_pt(norb, nelec_sp, n_batch = options["n_batch"])
        noccA, noccB = trial.nelec[0], trial.nelec[1]
        wave_data["mo_coeff"] = [
            mo_coeff[0][:, : noccA],
            mo_coeff[1][:, : noccB],
        ]
        ham_data['h1_mod'] = h1_mod
        amplitudes = np.load(amp_file)
        t1a = jnp.array(amplitudes["t1a"])
        t1b = jnp.array(amplitudes["t1b"])
        t2aa = jnp.array(amplitudes["t2aa"])
        t2ab = jnp.array(amplitudes["t2ab"])
        t2bb = jnp.array(amplitudes["t2bb"])
        wave_data['t1a'] = t1a
        wave_data['t1b'] = t1b
        wave_data["t2aa"] = t2aa
        wave_data["t2bb"] = t2bb
        wave_data["t2ab"] = t2ab

    elif options["trial"] == "uccsd_pt2":
        trial = wavefunctions.uccsd_pt2(norb, nelec_sp, n_batch = options["n_batch"])
        noccA, noccB = trial.nelec[0], trial.nelec[1]
        wave_data["mo_coeff"] = [
            mo_coeff[0][:, : noccA],
            mo_coeff[1][:, : noccB],
        ]
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
        wave_data["t2aa"] = t2aa
        wave_data["t2bb"] = t2bb
        wave_data["t2ab"] = t2ab

    elif options["trial"] == "uccsd_pt2_ad":
        trial = wavefunctions.uccsd_pt2_ad(norb, nelec_sp, n_batch=options["n_batch"])
        noccA, noccB = trial.nelec[0], trial.nelec[1]
        wave_data["mo_coeff"] = [
            mo_coeff[0][:, : noccA],
            mo_coeff[1][:, : noccB],
        ]
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
        wave_data["rot_t2AA"] = jnp.einsum('ik,jl,kalb->iajb',
            mo_ta[:noccA,:noccA].T,mo_ta[:noccA,:noccA].T,t2aa)
        wave_data["rot_t2BB"] = jnp.einsum('ik,jl,kalb->iajb',
            mo_tb[:noccB,:noccB].T,mo_tb[:noccB,:noccB].T,t2bb)
        wave_data["rot_t2AB"] = jnp.einsum('ik,jl,kalb->iajb',
            mo_ta[:noccA,:noccA].T,mo_tb[:noccB,:noccB].T,t2ab)

    elif options["trial"] == "uccsd_pt2_true_ad":
        trial = wavefunctions.uccsd_pt2_true_ad(norb, nelec_sp, n_batch=options["n_batch"])
        noccA, noccB = trial.nelec[0], trial.nelec[1]
        wave_data["mo_coeff"] = [
            mo_coeff[0][:, : noccA],
            mo_coeff[1][:, : noccB],
        ]
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
        wave_data["rot_t2AA"] = jnp.einsum('ik,jl,kalb->iajb',
            mo_ta[:noccA,:noccA].T,mo_ta[:noccA,:noccA].T,t2aa)
        wave_data["rot_t2BB"] = jnp.einsum('ik,jl,kalb->iajb',
            mo_tb[:noccB,:noccB].T,mo_tb[:noccB,:noccB].T,t2bb)
        wave_data["rot_t2AB"] = jnp.einsum('ik,jl,kalb->iajb',
            mo_ta[:noccA,:noccA].T,mo_tb[:noccB,:noccB].T,t2ab)

    if options["walker_type"] == "rhf":
        if options["symmetry"]:
            ham_data["mask"] = jnp.where(jnp.abs(ham_data["h1"]) > 1.0e-10, 1.0, 0.0)
        else:
            ham_data["mask"] = jnp.ones(ham_data["h1"].shape)
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
    if  'pt' in options['trial'] and 'cc' in options['trial']:
        if '2' in options['trial']:
            sampler = sampling.sampler_pt2(
                options["n_prop_steps"],
                options["n_ene_blocks"],
                options["n_sr_blocks"],
                options["n_blocks"],
                nchol,)
        else:
            sampler = sampling.sampler_pt(
                options["n_prop_steps"],
                options["n_ene_blocks"],
                options["n_sr_blocks"],
                options["n_blocks"],
                nchol,)
    elif  'cisd_hf1' in options['trial']:
        sampler = sampling.sampler_cisd_hf1(
                options["n_prop_steps"],
                options["n_ene_blocks"],
                options["n_sr_blocks"],
                options["n_blocks"],
                nchol,)
    elif  'cisd_hf2' in options['trial']:
        sampler = sampling.sampler_cisd_hf2(
                options["n_prop_steps"],
                options["n_ene_blocks"],
                options["n_sr_blocks"],
                options["n_blocks"],
                nchol,)
    else:
        sampler = sampling.sampler(
                options["n_prop_steps"],
                options["n_ene_blocks"],
                options["n_sr_blocks"],
                options["n_blocks"],
                nchol,)

    if rank == 0:
        print(f"# norb: {norb}")
        print(f"# nelec: {nelec_sp}")
        print("#")
        for op in options:
            if options[op] is not None:
                print(f"# {op}: {options[op]}")
        print("#")

    return ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI

import os
def run_afqmc(options,nproc=None,dbg=False,
              option_file='options.bin'):

    with open(option_file, 'wb') as f:
        pickle.dump(options, f)

    use_gpu = options["use_gpu"]
    if use_gpu:
        config.afqmc_config["use_gpu"] = True
        config.setup_jax()
        print(f'# running AFQMC on GPU')
        gpu_flag = "--use_gpu"
        mpi_prefix = ""
    else:
        print(f'# running AFQMC on CPU')
        gpu_flag = ""
        mpi_prefix = "mpirun "
        if nproc is not None:
            mpi_prefix += f"-np {nproc} "
    if  'pt' in options['trial'] and 'cc' in options['trial']:
        if '2' in options['trial']:
            if dbg:
                script='run_afqmc_ccsd_pt2_dbg.py'
            else:
                script='run_afqmc_ccsd_pt2.py'
        else:
            script='run_afqmc_ccsd_pt.py'
    elif  'cisd_hf' in options['trial']:
        if '1' in options['trial']:
            script='run_afqmc_cisd_hf1.py'
        elif '2' in options['trial']:
            script='run_afqmc_cisd_hf2.py'
    else:
        script='run_unrestricted_test.py'
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    script = f"{dir_path}/{script}"
    print(f'# AFQMC script: {script}')
    
    os.system(
        f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1;"
        f"{mpi_prefix} python {script} {gpu_flag} |tee afqmc.out"
    )
