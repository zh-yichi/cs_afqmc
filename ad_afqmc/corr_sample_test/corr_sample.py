import time
from functools import partial
from typing import List, Optional, Tuple, Union

import pickle
import jax
import jax.numpy as jnp
from jax import random, lax, jit
import numpy as np
#import scipy
from pyscf import __config__,  df,  lib, mcscf, scf
from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD
from ad_afqmc import pyscf_interface


from ad_afqmc.propagation import propagator
from ad_afqmc.wavefunctions import wave_function
from ad_afqmc import sampling

modified_cholesky = pyscf_interface.modified_cholesky
generate_integrals = pyscf_interface.generate_integrals
ao2mo_chol_copy = pyscf_interface.ao2mo_chol_copy
ao2mo_chol = pyscf_interface.ao2mo_chol
write_dqmc = pyscf_interface.write_dqmc

#@jit
def fix_len_chunked_cholesky(mol, chol_len, max_error=1e-6, verbose=False, cmax=10):
    """Modified cholesky decomposition of certain length from pyscf eris."""

    nao = mol.nao_nr()
    diag = np.zeros(nao * nao)
    nchol_max = cmax * nao
    # This shape is more convenient for pauxy.
    chol_vecs = np.zeros((nchol_max, nao * nao))
    ndiag = 0
    dims = [0]
    nao_per_i = 0
    for i in range(0, mol.nbas):
        l = mol.bas_angular(i)
        nc = mol.bas_nctr(i)
        nao_per_i += (2 * l + 1) * nc
        dims.append(nao_per_i)
    # print (dims)
    for i in range(0, mol.nbas):
        shls = (i, i + 1, 0, mol.nbas, i, i + 1, 0, mol.nbas)
        buf = mol.intor("int2e_sph", shls_slice=shls)
        di, dk, dj, dl = buf.shape
        diag[ndiag : ndiag + di * nao] = buf.reshape(di * nao, di * nao).diagonal()
        ndiag += di * nao
    nu = np.argmax(diag)
    delta_max = diag[nu]
    if verbose:
        print("# Generating Cholesky decomposition of ERIs." % nchol_max)
        print("# max number of cholesky vectors = %d" % nchol_max)
        print("# iteration %5d: delta_max = %f" % (0, delta_max))
    j = nu // nao
    l = nu % nao
    sj = np.searchsorted(dims, j)
    sl = np.searchsorted(dims, l)
    if dims[sj] != j and j != 0:
        sj -= 1
    if dims[sl] != l and l != 0:
        sl -= 1
    Mapprox = np.zeros(nao * nao)
    # ERI[:,jl]
    eri_col = mol.intor(
        "int2e_sph", shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1)
    )
    cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
    chol_vecs[0] = np.copy(eri_col[:, :, cj, cl].reshape(nao * nao)) / delta_max**0.5

    nchol = 0
    while abs(delta_max) > max_error:
        # Update cholesky vector
        start = time.time()
        # M'_ii = L_i^x L_i^x
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
        # D_ii = M_ii - M'_ii
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        # Compute ERI chunk.
        # shls_slice computes shells of integrals as determined by the angular
        # momentum of the basis function and the number of contraction
        # coefficients. Need to search for AO index within this shell indexing
        # scheme.
        # AO index.
        j = nu // nao
        l = nu % nao
        # Associated shell index.
        sj = np.searchsorted(dims, j)
        sl = np.searchsorted(dims, l)
        if dims[sj] != j and j != 0:
            sj -= 1
        if dims[sl] != l and l != 0:
            sl -= 1
        # Compute ERI chunk.
        eri_col = mol.intor(
            "int2e_sph", shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1)
        )
        # Select correct ERI chunk from shell.
        cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
        Munu0 = eri_col[:, :, cj, cl].reshape(nao * nao)
        # Updated residual = \sum_x L_i^x L_nu^x
        R = np.dot(chol_vecs[: nchol + 1, nu], chol_vecs[: nchol + 1, :])
        chol_vecs[nchol + 1] = (Munu0 - R) / (delta_max) ** 0.5
        nchol += 1
        if verbose:
            step_time = time.time() - start
            info = (nchol, delta_max, step_time)
            print("# iteration %5d: delta_max = %13.8e: time = %13.8e" % info)

    if chol_len > nchol:
        raise ValueError(f"given cholesky vector length {chol_len} exceeded the \n"
                         f"decompostion {nchol} lower chol_len or higher the chol_cut")

    return chol_vecs[:chol_len]

def fix_len_generate_integrals(mol, hcore, X, chol_len, chol_cut=1e-6, verbose=False, DFbas=None):
    # Unpack SCF data.
    # Step 1. Rotate core Hamiltonian to orthogonal basis.
    if verbose:
        print(" # Transforming hcore and eri to ortho AO basis.")
    if len(X.shape) == 2:
        h1e = np.dot(X.T, np.dot(hcore, X))
    elif len(X.shape) == 3:
        h1e = np.dot(X[0].T, np.dot(hcore, X[0]))

    if DFbas is not None:
        chol_vecs = df.incore.cholesky_eri(mol, auxbasis=DFbas)
        chol_vecs = lib.unpack_tril(chol_vecs).reshape(chol_vecs.shape[0], -1)
    else:  # do cholesky
        # nbasis = h1e.shape[-1]
        # Step 2. Genrate Cholesky decomposed ERIs in non-orthogonal AO basis.
        if verbose:
            print(" # Performing modified Cholesky decomposition on ERI tensor.")
        chol_vecs = fix_len_chunked_cholesky(mol,chol_len,max_error=chol_cut,verbose=verbose)

    if verbose:
        print(" # Orthogonalising Cholesky vectors.")
    start = time.time()

    # Step 2.a Orthogonalise Cholesky vectors.
    if len(X.shape) == 2 and X.shape[0] != X.shape[1]:
        chol_vecs = ao2mo_chol_copy(chol_vecs, X)
    elif len(X.shape) == 2:
        ao2mo_chol(chol_vecs, X)
    elif len(X.shape) == 3:
        ao2mo_chol(chol_vecs, X[0])
    if verbose:
        print(" # Time to orthogonalise: %f" % (time.time() - start))
    enuc = mol.energy_nuc()
    # Step 3. (Optionally) freeze core / virtuals.
    nelec = mol.nelec
    return h1e, chol_vecs, nelec, enuc

def fix_len_chol_prep(
    mf_or_cc: Union[scf.uhf.UHF, scf.rhf.RHF, CCSD, UCCSD],
    chol_len,
    basis_coeff: Optional[np.ndarray] = None,
    norb_frozen: int = 0,
    chol_cut: float = 1e-7,
    mo_file = "mo_coeff.npz",
    chol_file = "FCIDUMP_chol"
):
    
    mf = mf_or_cc
    mol = mf.mol

    if basis_coeff is None:
        if isinstance(mf, scf.uhf.UHF):
            basis_coeff = mf.mo_coeff[0]
        else:
            basis_coeff = mf.mo_coeff
            
    print("# Calculating Cholesky integrals")

    h1e, chol, nelec, enuc, nbasis,_ = [None] * 6

    h1e, chol, nelec, enuc = \
        fix_len_generate_integrals(mol, mf.get_hcore(), basis_coeff, chol_len, chol_cut)
    nbasis = h1e.shape[-1]
    nelec = mol.nelec

    if norb_frozen > 0:
        assert norb_frozen * 2 < sum(
            nelec
        ), "Frozen orbitals exceed number of electrons"
        mc = mcscf.CASSCF(
            mf, mol.nao - norb_frozen, mol.nelectron - 2 * norb_frozen
        )
        nelec = mc.nelecas  # type: ignore
        mc.mo_coeff = basis_coeff  # type: ignore
        h1e, enuc = mc.get_h1eff()  # type: ignore
        chol = chol.reshape((-1, nbasis, nbasis))
        chol = chol[:, mc.ncore : mc.ncore + mc.ncas, mc.ncore : mc.ncore + mc.ncas]  # type: ignore

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

    # write trial mo coefficients
    trial_coeffs = np.empty((2, nbasis, nbasis))
    overlap = mf.get_ovlp(mol)
    if isinstance(mf, (scf.uhf.UHF, scf.rohf.ROHF)):
        uhfCoeffs = np.empty((nbasis, 2 * nbasis))
        if isinstance(mf, scf.uhf.UHF):
            q, r = np.linalg.qr(
                basis_coeff[:, norb_frozen:]
                .T.dot(overlap)
                .dot(mf.mo_coeff[0][:, norb_frozen:])
            )
            sgn = np.sign(r.diagonal())
            q = np.einsum("ij,j->ij", q, sgn)
            # q2 = basis_coeff[:, norb_frozen:].T.dot(overlap).dot(mf.mo_coeff[0][:, norb_frozen:])
            # print("max err a", np.max(abs(q-q2)))
            # q, _ = np.linalg.qr(
            #    basis_coeff[:, norb_frozen:]
            #    .T.dot(overlap)
            #    .dot(mf.mo_coeff[0][:, norb_frozen:])
            # )
            uhfCoeffs[:, :nbasis] = q
            q, r = np.linalg.qr(
                basis_coeff[:, norb_frozen:]
                .T.dot(overlap)
                .dot(mf.mo_coeff[1][:, norb_frozen:])
            )
            sgn = np.sign(r.diagonal())
            q = np.einsum("ij,j->ij", q, sgn)
            # q2 = basis_coeff[:, norb_frozen:].T.dot(overlap).dot(mf.mo_coeff[1][:, norb_frozen:])
            # print("max err b", np.max(abs(q-q2)))
            # import pdb
            # pdb.set_trace()
            # q, _ = np.linalg.qr(
            #     basis_coeff[:, norb_frozen:]
            #     .T.dot(overlap)
            #     .dot(mf.mo_coeff[1][:, norb_frozen:])
            # )
            uhfCoeffs[:, nbasis:] = q
        else:
            q, r = np.linalg.qr(
                basis_coeff[:, norb_frozen:]
                .T.dot(overlap)
                .dot(mf.mo_coeff[:, norb_frozen:])
            )
            sgn = np.sign(r.diagonal())
            q = np.einsum("ij,j->ij", q, sgn)
            uhfCoeffs[:, :nbasis] = q
            uhfCoeffs[:, nbasis:] = q

        trial_coeffs[0] = uhfCoeffs[:, :nbasis]
        trial_coeffs[1] = uhfCoeffs[:, nbasis:]
        # np.savetxt("uhf.txt", uhfCoeffs)
        np.savez(mo_file, mo_coeff=trial_coeffs)

    elif isinstance(mf, scf.rhf.RHF):
        q, _ = np.linalg.qr(
            basis_coeff[:, norb_frozen:]
            .T.dot(overlap)
            .dot(mf.mo_coeff[:, norb_frozen:])
        )
        trial_coeffs[0] = q
        trial_coeffs[1] = q
        np.savez(mo_file, mo_coeff=trial_coeffs)

    write_dqmc(
        h1e,
        h1e_mod,
        chol,
        sum(nelec),
        nbasis,
        enuc,
        ms=mol.spin,
        filename=chol_file,
        mo_coeffs=trial_coeffs,
    )


sampler_eq = sampling.sampler(n_prop_steps=50, n_ene_blocks=5, n_sr_blocks=10)

def init_prop(ham_data, ham, prop, trial, wave_data, seed, MPI):
    comm = MPI.COMM_WORLD
    #size = comm.Get_size()
    rank = comm.Get_rank()
    #seed = options["seed"]
    #neql = options["n_eql"]
    init_walkers: Optional[Union[List, jax.Array]] = None
    trial_rdm1 = trial.get_rdm1(wave_data)
    if "rdm1" not in wave_data:
        wave_data["rdm1"] = trial_rdm1
    ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
    ham_data = ham.build_propagation_intermediates(ham_data, prop, trial, wave_data)
    prop_data = prop.init_prop_data(trial, wave_data, ham_data, init_walkers)
    prop_data["key"] = random.PRNGKey(seed + rank)
    prop_data["n_killed_walkers"] = 0
    #print(f"# initial energy: {prop_data['e_estimate']:.9e}")
    
    return prop_data, ham_data

def en_samples(prop_data,ham_data,prop,trial,wave_data):
    energy_samples = jnp.real(
        trial.calc_energy(prop_data["walkers"], ham_data, wave_data)
    )
    energy_samples = jnp.where(
        jnp.abs(energy_samples - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt),
        prop_data["e_estimate"],
        energy_samples,
    )
    return energy_samples

def block_en_weight(prop_data,ham_data,prop,trial,wave_data):

    energy_samples = en_samples(prop_data,ham_data,prop,wave_data,trial)

    block_weight = jnp.sum(prop_data["weights"])
    block_energy = jnp.sum(energy_samples * prop_data["weights"]) / block_weight
    return block_energy, block_weight

@partial(jit, static_argnums=(3,4))
def field_block_scan(
        prop_data: dict,
        fields,
        ham_data: dict,
        prop: propagator,
        trial: wave_function,
        wave_data: dict,
        ) -> Tuple[dict, Tuple[jax.Array, jax.Array]]:
    """Block scan function for a given field"""
    with open("options.pkl", "rb") as file:
        options = pickle.load(file)
    if options["free_proj"]:
        # print("free projection propagation")
        _step_scan_wrapper = lambda x, y: sampler_eq._step_scan_free(
            x, y, ham_data, prop, trial, wave_data
        )
        prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)
        # energy_samples = trial.calc_energy(prop_data["walkers"], ham_data, wave_data)
        # # energy_samples = jnp.where(jnp.abs(energy_samples - ham_data['ene0']) > jnp.sqrt(2./propagator.dt), ham_data['ene0'],     energy_samples)
        # block_energy = jnp.sum(energy_samples * prop_data["overlaps"]) / jnp.sum(
        #     prop_data["overlaps"]
        # )
        # #block_weight = jnp.sum(prop_data["overlaps"])
        pass
    else:
        # print("phaseless propagation")
        _step_scan_wrapper = lambda x, y: sampler_eq._step_scan(
            x, y, ham_data, prop, trial, wave_data
        )
        prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)
        prop_data["n_killed_walkers"] += prop_data["weights"].size - jnp.count_nonzero(
            prop_data["weights"]
        )
        prop_data = prop.orthonormalize_walkers(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)

        block_energy,_ = block_en_weight(prop_data,ham_data,prop,wave_data,trial)
        prop_data["pop_control_ene_shift"] = (
            0.9 * prop_data["pop_control_ene_shift"] + 0.1 * block_energy
        )
    return prop_data

@partial(jit, static_argnums=(2,3,7,8))
def cs_block_scan(
        prop_data1: dict,
        ham_data1: dict,
        prop1: propagator,
        trial1: wave_function,
        wave_data1: dict,
        prop_data2: dict,
        ham_data2: dict,
        prop2: propagator,
        trial2: wave_function,
        wave_data2: dict):
    '''correlated sampling of two blocks over the same field'''
    prop_data1["key"], subkey1 = random.split(prop_data1["key"])
    fields = random.normal(
        subkey1,
        shape=(
            sampler_eq.n_prop_steps,
            prop1.n_walkers,
            ham_data1["chol"].shape[0],
        )
    )
    prop_data1 = field_block_scan(prop_data1,fields,ham_data1,prop1,trial1,wave_data1)
    prop_data2 = field_block_scan(prop_data2,fields,ham_data2,prop2,trial2,wave_data2)

    return prop_data1, prop_data2

@partial(jit, static_argnums=(2,3,7,8))
def ucs_block_scan(
        prop_data1: dict,
        ham_data1: dict,
        prop1: propagator,
        trial1: wave_function,
        wave_data1: dict,
        prop_data2: dict,
        ham_data2: dict,
        prop2: propagator,
        trial2: wave_function,
        wave_data2: dict):
    '''correlated sampling of two blocks over the same field'''
    prop_data1["key"], subkey1 = random.split(prop_data1["key"])
    fields1 = random.normal(
        subkey1,
        shape=(
            sampler_eq.n_prop_steps,
            prop1.n_walkers,
            ham_data1["chol"].shape[0],
        )
    )
    prop_data1 = field_block_scan(prop_data1,fields1,ham_data1,prop1,trial1,wave_data1)

    prop_data2["key"], subkey2 = random.split(prop_data2["key"])
    fields2 = random.normal(
        subkey2,
        shape=(
            sampler_eq.n_prop_steps,
            prop2.n_walkers,
            ham_data2["chol"].shape[0],
        )
    )
    prop_data2 = field_block_scan(prop_data2,fields2,ham_data2,prop2,trial2,wave_data2)

    return prop_data1, prop_data2

@partial(jit, static_argnums=(0,3,4,8,9))
def cs_steps_scan(steps,
                  prop_data1,ham_data1,prop1,trial1,wave_data1,
                  prop_data2,ham_data2,prop2,trial2,wave_data2
                  ):

    cs_prop_data = (prop_data1,prop_data2)
    def cs_step(cs_prop_data,_):
        prop_data1,prop_data2= cs_prop_data
        prop_data1,prop_data2 = cs_block_scan(prop_data1,ham_data1,prop1,trial1,wave_data1,
                                              prop_data2,ham_data2,prop2,trial2,wave_data2)
        cs_prop_data = (prop_data1,prop_data2)
        loc_en_samples1 = en_samples(prop_data1,ham_data1,prop1,trial1,wave_data1)
        loc_en_samples2 = en_samples(prop_data2,ham_data2,prop2,trial2,wave_data2)
        loc_weight_sample1 = prop_data1["weights"]
        loc_weight1 = jnp.sum(loc_weight_sample1)
        loc_weight_sample2 = prop_data2["weights"]
        loc_weight2 = jnp.sum(loc_weight_sample2)
        loc_en_sample1 = loc_en_samples1*loc_weight_sample1
        loc_en_sample2 = loc_en_samples2*loc_weight_sample2
        loc_en1 = sum(loc_en_sample1) #not normalized
        loc_en2 = sum(loc_en_sample2) #not normalized
        return cs_prop_data, (loc_en1,loc_weight1,loc_en2,loc_weight2)

    cs_prop_data, (loc_en1,loc_weight1,loc_en2,loc_weight2) \
        = jax.lax.scan(cs_step,cs_prop_data,xs=None,length=steps)
    return cs_prop_data, (loc_en1,loc_weight1,loc_en2,loc_weight2)


@partial(jit, static_argnums=(0,3,4,8,9))
# def ucs_steps_scan(steps,
#                    prop_data1,ham_data1,prop1,trial1,wave_data1,
#                    prop_data2,ham_data2,prop2,trial2,wave_data2
#                    ):

#     ucs_prop_data = (prop_data1,prop_data2)
#     def ucs_step(ucs_prop_data,_):
#         prop_data1,prop_data2= ucs_prop_data
#         prop_data1,prop_data2 = ucs_block_scan(prop_data1,ham_data1,prop1,trial1,wave_data1,
#                                                prop_data2,ham_data2,prop2,trial2,wave_data2)
#         ucs_prop_data = (prop_data1,prop_data2)
#         return ucs_prop_data, None

#     (prop_data1,prop_data2),_ = jax.lax.scan(ucs_step,ucs_prop_data,xs=None,length=steps)
#     prop_data1 = prop1.stochastic_reconfiguration_local(prop_data1)
#     prop_data2 = prop2.stochastic_reconfiguration_local(prop_data2)
#     ucs_prop_data = (prop_data1,prop_data2)
#     return ucs_prop_data
def ucs_steps_scan(steps,
                  prop_data1,ham_data1,prop1,trial1,wave_data1,
                  prop_data2,ham_data2,prop2,trial2,wave_data2
                  ):

    ucs_prop_data = (prop_data1,prop_data2)
    def ucs_step(ucs_prop_data,_):
        prop_data1,prop_data2= ucs_prop_data
        prop_data1,prop_data2 = ucs_block_scan(prop_data1,ham_data1,prop1,trial1,wave_data1,
                                               prop_data2,ham_data2,prop2,trial2,wave_data2)
        ucs_prop_data = (prop_data1,prop_data2)
        loc_en_samples1 = en_samples(prop_data1,ham_data1,prop1,trial1,wave_data1)
        loc_en_samples2 = en_samples(prop_data2,ham_data2,prop2,trial2,wave_data2)
        loc_weight_sample1 = prop_data1["weights"]
        loc_weight1 = jnp.sum(loc_weight_sample1)
        loc_weight_sample2 = prop_data2["weights"]
        loc_weight2 = jnp.sum(loc_weight_sample2)
        loc_en_sample1 = loc_en_samples1*loc_weight_sample1
        loc_en_sample2 = loc_en_samples2*loc_weight_sample2
        loc_en1 = sum(loc_en_sample1) #not normalized
        loc_en2 = sum(loc_en_sample2) #not normalized
        return ucs_prop_data, (loc_en1,loc_weight1,loc_en2,loc_weight2)

    ucs_prop_data, (loc_en1,loc_weight1,loc_en2,loc_weight2) \
        = jax.lax.scan(ucs_step,ucs_prop_data,xs=None,length=steps)
    return ucs_prop_data, (loc_en1,loc_weight1,loc_en2,loc_weight2)

#@jit
def scan_seeds(seeds,eq_steps,
               prop_data1_init,ham_data1_init,prop1,trial1,wave_data1,
               prop_data2_init,ham_data2_init,prop2,trial2,wave_data2, 
               MPI):
    '''
    do a number of independent runs of given equilirium steps
    for a given array of seeds
    return local energy of system1, local weight of system1
    and the same for system2.
    the ensemble energy average for each system should be 
    loc_en/loc_weight
    '''

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    def _seed_cs(carry,seed):
        prop_data1_init["key"] = jax.random.PRNGKey(seed + rank)
        _,(loc_en1,loc_weight1,loc_en2,loc_weight2) \
            = cs_steps_scan(eq_steps,
                            prop_data1_init,ham_data1_init,prop1,trial1,wave_data1,
                            prop_data2_init,ham_data2_init,prop2,trial2,wave_data2)
        
        # loc_en_samples1 = en_samples(prop_data1,ham_data1_init,prop1,trial1,wave_data1)
        # loc_en_samples2 = en_samples(prop_data2,ham_data2_init,prop2,trial2,wave_data2)
        # loc_weight_sample1 = prop_data1["weights"]
        # loc_weight1 = jnp.sum(loc_weight_sample1)
        # loc_weight_sample2 = prop_data2["weights"]
        # loc_weight2 = jnp.sum(loc_weight_sample2)
        # loc_en_sample1 = loc_en_samples1*loc_weight_sample1
        # loc_en_sample2 = loc_en_samples2*loc_weight_sample2
        # loc_en1 = sum(loc_en_sample1) #not normalized
        # loc_en2 = sum(loc_en_sample2) #not normalized
        return carry, (loc_en1,loc_weight1,loc_en2,loc_weight2)
    
    _, (loc_en1,loc_weight1,loc_en2,loc_weight2) = jax.lax.scan(_seed_cs, None, seeds)

    return loc_en1,loc_weight1,loc_en2,loc_weight2

def ucs_scan_seeds(seeds,eq_steps,
                  prop_data1_init,ham_data1_init,prop1,trial1,wave_data1,
                  prop_data2_init,ham_data2_init,prop2,trial2,wave_data2, 
                  MPI):
    '''
    do a number of independent runs of given equilirium steps
    for a given array of seeds
    return local energy of system1, local weight of system1
    and the same for system2.
    the ensemble energy average for each system should be 
    loc_en/loc_weight
    '''

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    def _seed_ucs(carry,seed):
        (seed1,seed2) = seed
        prop_data1_init["key"] = jax.random.PRNGKey(seed1 + rank)
        prop_data2_init["key"] = jax.random.PRNGKey(seed2 + rank)
        _,(loc_en1,loc_weight1,loc_en2,loc_weight2) \
            = ucs_steps_scan(eq_steps,
                            prop_data1_init,ham_data1_init,prop1,trial1,wave_data1,
                            prop_data2_init,ham_data2_init,prop2,trial2,wave_data2)
        
        return carry, (loc_en1,loc_weight1,loc_en2,loc_weight2)
    
    _, (loc_en1,loc_weight1,loc_en2,loc_weight2) = jax.lax.scan(_seed_ucs, None, seeds)

    return loc_en1,loc_weight1,loc_en2,loc_weight2