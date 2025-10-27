from jax import scipy as jsp
from jax import numpy as jnp
from jax import lax, jit, vmap, random
from functools import partial

@partial(jit, static_argnums=(0, 2))
def _build_propagation_intermediates(self, ham_data, trial, wave_data):
    rdm1 = wave_data["rdm1"]
    mf_shift_a = 1.0j * vmap(
        lambda x: jnp.sum(x.reshape(trial.norb, trial.norb)
                          * rdm1[0]))(ham_data["chol"][0])
    mf_shift_b = 1.0j * vmap(
        lambda x: jnp.sum(x.reshape(trial.norb, trial.norb) 
                          * rdm1[1]))(ham_data["chol"][1])
    ham_data["mf_shifts"] = mf_shift_a + mf_shift_b
    ham_data["h0_prop"] = (
        - ham_data["h0"] - jnp.sum(ham_data["mf_shifts"]**2) / 2.0
                                   )
    # alpha
    v0_a = 0.5 * jnp.einsum(
        "gik,gjk->ij",
        ham_data["chol"][0].reshape(-1, trial.norb, trial.norb),
        ham_data["chol"][0].reshape(-1, trial.norb, trial.norb),
        optimize="optimal",
    )
    # beta
    v0_b = 0.5 * jnp.einsum(
        "gik,gjk->ij",
        ham_data["chol"][1].reshape(-1, trial.norb, trial.norb),
        ham_data["chol"][1].reshape(-1, trial.norb, trial.norb),
        optimize="optimal",
    )
    # alpha
    v1_a = jnp.real(
        1.0j
        * jnp.einsum(
            "g,gik->ik",ham_data["mf_shifts"],
            ham_data["chol"][0].reshape(-1, trial.norb, trial.norb),
        )
    )
    # beta
    v1_b = jnp.real(
        1.0j
        * jnp.einsum(
            "g,gik->ik",ham_data["mf_shifts"],
            ham_data["chol"][1].reshape(-1, trial.norb, trial.norb),
        )
    )
    h1_mod = ham_data["h1"] - jnp.array([v0_a + v1_a, v0_b + v1_b])
    ham_data["exp_h1"] = jnp.array(
        [
            jsp.linalg.expm(-self.dt * h1_mod[0] / 2.0),
            jsp.linalg.expm(-self.dt * h1_mod[1] / 2.0),
        ]
    )
    return ham_data

@partial(jit, static_argnums=(0,))
def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
    ham_data["h1"] = (
        ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
    )
    ham_data["h1"] = (
        ham_data["h1"].at[1].set((ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0)
    )
    ham_data["rot_h1"] = [
        wave_data["mo_coeff"][0].T.conj() @ ham_data["h1"][0],
        wave_data["mo_coeff"][1].T.conj() @ ham_data["h1"][1],
    ]
    ham_data["rot_chol"] = [
        jnp.einsum(
            "pi,gij->gpj",
            wave_data["mo_coeff"][0].T.conj(),
            ham_data["chol"][0].reshape(-1, self.norb, self.norb),
        ),
        jnp.einsum(
            "pi,gij->gpj",
            wave_data["mo_coeff"][1].T.conj(),
            ham_data["chol"][1].reshape(-1, self.norb, self.norb),
        ),
    ]
    return ham_data

@partial(jit, static_argnums=(0,))
def _apply_trotprop(
    self, ham_data, walkers, fields
):
    n_walkers = walkers[0].shape[0]
    batch_size = n_walkers // self.n_batch
    # print('### running this trot ###')

    def scanned_fun(carry, batch):
        field_batch, walker_batch_0, walker_batch_1 = batch
        # alpha
        vhs_a = (
            1.0j
            * jnp.sqrt(self.dt)
            * field_batch.dot(ham_data["chol"][0]).reshape(
                batch_size, walkers[0].shape[1], walkers[0].shape[1]
            )
        )
        # beta
        vhs_b = (
            1.0j
            * jnp.sqrt(self.dt)
            * field_batch.dot(ham_data["chol"][1]).reshape(
                batch_size, walkers[0].shape[1], walkers[0].shape[1]
            )
        )
        walkers_new_0 = vmap(self._apply_trotprop_det, in_axes=(None, 0, 0))(
            ham_data["exp_h1"][0], vhs_a, walker_batch_0
        )
        walkers_new_1 = vmap(self._apply_trotprop_det, in_axes=(None, 0, 0))(
            ham_data["exp_h1"][1], vhs_b, walker_batch_1
        )
        return carry, [walkers_new_0, walkers_new_1]

    _, walkers_new = lax.scan(
        scanned_fun,
        None,
        (
            fields.reshape(self.n_batch, batch_size, -1),
            walkers[0].reshape(
                self.n_batch, batch_size, walkers[0].shape[1], walkers[0].shape[2]
            ),
            walkers[1].reshape(
                self.n_batch, batch_size, walkers[1].shape[1], walkers[1].shape[2]
            ),
        ),
    )
    walkers = [
        walkers_new[0].reshape(n_walkers, walkers[0].shape[1], walkers[0].shape[2]),
        walkers_new[1].reshape(n_walkers, walkers[1].shape[1], walkers[1].shape[2]),
    ]
    return walkers

@partial(jit, static_argnums=(0, 4, 5))
def _block_scan(self,prop_data,_x,ham_data,prop,trial,wave_data):
    """Block scan function. Propagation and energy calculation."""
    prop_data["key"], subkey = random.split(prop_data["key"])
    fields = random.normal(
        subkey,
        shape=(
            self.n_prop_steps,
            prop.n_walkers,
            ham_data["chol"][0].shape[0],
        ),
    )
    _step_scan_wrapper = lambda x, y: self._step_scan(
        x, y, ham_data, prop, trial, wave_data
    )
    prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)
    prop_data["n_killed_walkers"] += prop_data["weights"].size - jnp.count_nonzero(
        prop_data["weights"]
    )
    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
    energy_samples = jnp.real(
        trial.calc_energy(prop_data["walkers"], ham_data, wave_data)
    )
    energy_samples = jnp.where(
        jnp.abs(energy_samples - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt),
        prop_data["e_estimate"],
        energy_samples,
    )
    block_weight = jnp.sum(prop_data["weights"])
    block_energy = jnp.sum(energy_samples * prop_data["weights"]) / block_weight
    prop_data["pop_control_ene_shift"] = (
        0.9 * prop_data["pop_control_ene_shift"] + 0.1 * block_energy
    )
    return prop_data, (block_energy, block_weight)

import struct
import time
from functools import partial
from typing import Optional, Union
import h5py
import jax.numpy as jnp
import numpy as np
import scipy
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
    mo_file = "mo_coeff.npz",
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
        # print(mf.mo_coeff)
        # mo_copy = mf.mo_coeff.copy()
        # print(mo_copy)
        cc = mf_or_cc
        if cc.frozen is not None:
            norb_frozen = cc.frozen
        
        if 'ci' in trial.lower():
        # build ci from cc #
            if isinstance(cc, UCCSD):
                ci2aa = cc.t2[0] + 2 * np.einsum("ia,jb->ijab", cc.t1[0], cc.t1[0])
                ci2aa = (ci2aa - ci2aa.transpose(0, 1, 3, 2)) / 2
                ci2aa = ci2aa.transpose(0, 2, 1, 3)
                ci2bb = cc.t2[2] + 2 * np.einsum("ia,jb->ijab", cc.t1[1], cc.t1[1])
                ci2bb = (ci2bb - ci2bb.transpose(0, 1, 3, 2)) / 2
                ci2bb = ci2bb.transpose(0, 2, 1, 3)
                ci2ab = cc.t2[1] + np.einsum("ia,jb->ijab", cc.t1[0], cc.t1[1])
                ci2ab = ci2ab.transpose(0, 2, 1, 3)
                ci1a = np.array(cc.t1[0])
                ci1b = np.array(cc.t1[1])
                
                np.savez(
                    amp_file,
                    ci1a=ci1a,
                    ci1b=ci1b,
                    ci2aa=ci2aa,
                    ci2ab=ci2ab,
                    ci2bb=ci2bb,
                )
            else:
                ci2 = cc.t2 + np.einsum("ia,jb->ijab", np.array(cc.t1), np.array(cc.t1))
                ci2 = ci2.transpose(0, 2, 1, 3)
                ci1 = np.array(cc.t1)
                np.savez(amp_file, ci1=ci1, ci2=ci2)
        
        elif 'cc' in trial.lower():
        # ccsd trial #
            if isinstance(cc, UCCSD):
                t2aa = cc.t2[0] # + 2 * np.einsum("ia,jb->ijab", cc.t1[0], cc.t1[0])
                # t2aa = (t2aa - t2aa.transpose(0, 1, 3, 2)) / 2
                t2aa = t2aa.transpose(0, 2, 1, 3)
                t2bb = cc.t2[2] # + 2 * np.einsum("ia,jb->ijab", cc.t1[1], cc.t1[1])
                # t2bb = (t2bb - t2bb.transpose(0, 1, 3, 2)) / 2
                t2bb = t2bb.transpose(0, 2, 1, 3)
                t2ab = cc.t2[1] # + np.einsum("ia,jb->ijab", cc.t1[0], cc.t1[1])
                t2ab = t2ab.transpose(0, 2, 1, 3)
                t1a = np.array(cc.t1[0])
                t1b = np.array(cc.t1[1])
                
                np.savez(
                    amp_file,
                    t1a=t1a,
                    t1b=t1b,
                    t2aa=t2aa,
                    t2ab=t2ab,
                    t2bb=t2bb,
                )
            else:
                t2 = cc.t2
                t2 = t2.transpose(0, 2, 1, 3)
                t1 = np.array(cc.t1)
                np.savez(amp_file, t1=t1, t2=t2)
    else:
        mf = mf_or_cc

    mol = mf.mol
    mo_copy = mf.mo_coeff.copy()
    # choose the orbital basis
    if basis_coeff is None:
        if isinstance(mf, scf.uhf.UHF):
            basis_coeff = mf.mo_coeff[0].copy()
        else:
            basis_coeff = mf.mo_coeff.copy()

    # calculate cholesky integrals
    print("# Calculating Cholesky integrals")
    # h1e, chol, nelec, enuc, nbasis = [None] * 5
    
    DFbas = None
    if getattr(mf, "with_df", None) is not None:
        print('# Decomposing ERI with DF')
        DFbas = mf.with_df.auxmol.basis  # type: ignore

    # assert norb_frozen * 2 < sum(
    #     nelec
    # ), "Frozen orbitals exceed number of electrons"
    if 'r' in trial.lower():
        mc = mcscf.CASSCF(
            mf, mol.nao - norb_frozen, mol.nelectron - 2 * norb_frozen
        )
        nelec = mc.nelecas  # type: ignore
        mc.mo_coeff = basis_coeff  # type: ignore
        h1e, enuc = mc.get_h1eff()  # type: ignore
        chol = chol.reshape((-1, nbasis, nbasis))
        chol = chol[:, mc.ncore : mc.ncore + mc.ncas, 
                    mc.ncore : mc.ncore + mc.ncas]
        _, chol, _, _ = \
            pyscf_interface.generate_integrals(
            mol, mf.get_hcore(), basis_coeff, chol_cut, DFbas=DFbas)
        print("# Finished calculating Cholesky integrals\n#")
        nbasis = h1e.shape[-1]
        print("# Size of the correlation space:")
        print(f"# Number of electrons: {nelec}")
        print(f"# Number of basis functions: {nbasis}")
        print(f"# Number of Cholesky vectors: {chol.shape[0]}\n#")
        chol = chol.reshape((-1, nbasis, nbasis))
        v0 = 0.5 * np.einsum("nik,njk->ij", chol, chol, optimize="optimal")
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
        chol_b = np.einsum('pr,grs,sq->gpq',a2b,chol_a,a2b.T)
        chol_a = chol_a[:, mc.ncore[0] : mc.ncore[0] + mc.ncas,
                        mc.ncore[0] : mc.ncore[0] + mc.ncas]
        chol_b = chol_b.reshape((-1, nao, nao))
        chol_b = chol_b[:, mc.ncore[1] : mc.ncore[1] + mc.ncas, 
                        mc.ncore[1] : mc.ncore[1] + mc.ncas]
        v0_a = 0.5 * np.einsum("nik,njk->ij", chol_a, chol_a, optimize="optimal")
        v0_b = 0.5 * np.einsum("nik,njk->ij", chol_b, chol_b, optimize="optimal")
        h1e = np.array(h1e)
        h1e_mod = np.array(h1e - np.array([v0_a,v0_b]))
        # print(h1e[1]-h1e[0])
        # print(h1e_mod[1]-h1e_mod[0])
        chol = np.array(
            [chol_a.reshape(chol_a.shape[0], -1),
             chol_b.reshape(chol_b.shape[0], -1)])
        print("# Finished calculating Cholesky integrals\n#")
        print("# Size of the correlation space:")
        print(f"# Number of electrons: {nelec}")
        print(f"# Number of basis functions: {nbasis}")
        print(f"# Number of Cholesky vectors: {chol_a.shape[0]}\n#")

    # # write trial mo coefficients
    # trial_coeffs = np.empty((2, nbasis, nbasis))
    # overlap = mf.get_ovlp()
    # if isinstance(mf, (scf.uhf.UHF, scf.rohf.ROHF)):
    #     uhfCoeffs = np.empty((nbasis, 2 * nbasis))
    #     if isinstance(mf, scf.uhf.UHF):
    #         # alpha
    #         # print(mf.mo_coeff)
    #         q, r = np.linalg.qr(
    #             basis_coeff[:, norb_frozen:]
    #             .T.dot(overlap)
    #             .dot(mo_copy[0][:, norb_frozen:])
    #         )
    #         sgn = np.sign(r.diagonal())
    #         q = np.einsum("ij,j->ij", q, sgn)
    #         uhfCoeffs[:, :nbasis] = q
    #         # beta
    #         q, r = np.linalg.qr(
    #             basis_coeff[:, norb_frozen:]
    #             .T.dot(overlap)
    #             .dot(mo_copy[1][:, norb_frozen:])
    #         )
    #         sgn = np.sign(r.diagonal())
    #         q = np.einsum("ij,j->ij", q, sgn)
    #         uhfCoeffs[:, nbasis:] = q
    #         trial_coeffs[0] = uhfCoeffs[:, :nbasis]
    #         trial_coeffs[1] = uhfCoeffs[:, nbasis:]
    #         # print(mf.mo_coeff[0])
    #         # print(mf.mo_coeff[1])
    #         # print(mo_copy)
    #         # mo_a = np.eye(nbasis)
    #         # mo_b_A = (mo_copy[0][:, norb_frozen:].T
    #         #     @ overlap
    #         #     @ mo_copy[1][:, norb_frozen:])
    #         # print(mo_b_A)
    #         # trial_coeffs[0] = mo_a
    #         # trial_coeffs[1] = mo_b_A
    #         np.savez(mo_file, mo_coeff=trial_coeffs)
    #     else:
    #         q, r = np.linalg.qr(
    #             basis_coeff[:, norb_frozen:]
    #             .T.dot(overlap)
    #             .dot(mf.mo_coeff[:, norb_frozen:])
    #         )
    #         sgn = np.sign(r.diagonal())
    #         q = np.einsum("ij,j->ij", q, sgn)
    #         uhfCoeffs[:, :nbasis] = q
    #         uhfCoeffs[:, nbasis:] = q

    #         trial_coeffs[0] = uhfCoeffs[:, :nbasis]
    #         trial_coeffs[1] = uhfCoeffs[:, nbasis:]
    #     # np.savetxt("uhf.txt", uhfCoeffs)
    #         np.savez(mo_file, mo_coeff=trial_coeffs)

    # elif isinstance(mf, scf.rhf.RHF):
    #     q, _ = np.linalg.qr(
    #         basis_coeff[:, norb_frozen:]
    #         .T.dot(overlap)
    #         .dot(mf.mo_coeff[:, norb_frozen:])
    #     )
    #     trial_coeffs[0] = q
    #     trial_coeffs[1] = q
    #     np.savez(mo_file, mo_coeff=trial_coeffs)
    
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
import time
import h5py
import numpy as np
from ad_afqmc import config
from functools import partial
from ad_afqmc import hamiltonian, propagation, sampling
from ad_afqmc.prop_unrestricted import wavefunctions
print = partial(print, flush=True)

def _prep_afqmc(options=None,
                option_file="options.bin",
                amp_file="amplitudes.npz",
                chol_file="FCIDUMP_chol"):
    
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
    ham_data["h1"] = jnp.array(h1)
    ham_data["h1_mod"] = jnp.array(h1_mod)
    nchol = chol[0].shape[0]
    ham_data["chol"] = jnp.array([chol[0].reshape(chol[0].shape[0], -1),
                                  chol[1].reshape(chol[1].shape[0], -1)])
    ham_data["ene0"] = options["ene0"]

    wave_data = {}
    mo_coeff = jnp.array([np.eye(norb),np.eye(norb)])
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
        wave_data["mo_coeff"] = mo_coeff[0][:,:nelec_sp[0]]
    elif options["trial"] == "ccsd_pt2":
        amplitudes = np.load(amp_file)
        t1 = jnp.array(amplitudes["t1"])
        t2 = jnp.array(amplitudes["t2"])
        trial_wave_data = {"t1": t1, "t2": t2}
        wave_data.update(trial_wave_data)
        trial = wavefunctions.ccsd_pt2(norb, nelec_sp, n_batch=options["n_batch"])
        wave_data["mo_coeff"] = mo_coeff[0][:,:nelec_sp[0]]
    elif options["trial"] == "ccsd_pt_ad":
        ham_data['h1_mod'] = h1_mod
        amplitudes = np.load(amp_file)
        t1 = jnp.array(amplitudes["t1"])
        t2 = jnp.array(amplitudes["t2"])
        trial_wave_data = {"t1": t1, "t2": t2}
        wave_data.update(trial_wave_data)
        wave_data["mo_coeff"] = np.eye(norb)[:,:nelec_sp[0]]
        trial = wavefunctions.ccsd_pt_ad(norb, nelec_sp, n_batch=options["n_batch"])
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
        wave_data['mo_coeff'] = mo_coeff[0][:,:nocc]
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
    sampler.n_chol = nchol

    if rank == 0:
        print(f"# norb: {norb}")
        print(f"# nelec: {nelec_sp}")
        print("#")
        for op in options:
            if options[op] is not None:
                print(f"# {op}: {options[op]}")
        print("#")

    # from ad_afqmc.ccsd_pt import prop_unrestricted
    import types

    trial._build_measurement_intermediates = types.MethodType(
        _build_measurement_intermediates, trial)

    prop._build_propagation_intermediates = types.MethodType(
        _build_propagation_intermediates, prop)

    prop._apply_trotprop = types.MethodType(
        _apply_trotprop, prop)
    
    sampler._block_scan = types.MethodType(
        _block_scan, sampler)

    return ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI

import os
def run_afqmc(options,nproc=None,
              option_file='options.bin',
              script='run_afqmc_ccsd_pt2.py'):

    with open(option_file, 'wb') as f:
        pickle.dump(options, f)

    use_gpu = options["use_gpu"]
    if use_gpu:
        print(f'# running AFQMC on GPU')
        gpu_flag = "--use_gpu"
        mpi_prefix = ""
    else:
        print(f'# running AFQMC on CPU')
        gpu_flag = ""
        mpi_prefix = "mpirun "
        if nproc is not None:
            mpi_prefix += f"-np {nproc} "
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)  
    script = f"{dir_path}/{script}"
    print(f'# AFQMC script: {script}')
    
    os.system(
        f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1;"
        f"{mpi_prefix} python {script} {gpu_flag} |tee afqmc_unrestricted.out"
    )

