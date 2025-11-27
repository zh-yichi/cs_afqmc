import pickle
import numpy as np
from pyscf.ci.cisd import CISD
from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD
from pyscf import lib, ao2mo, df, mcscf
from ad_afqmc import pyscf_interface
import h5py
from ad_afqmc import config
from functools import partial
from ad_afqmc import hamiltonian
from ad_afqmc.prop_unrestricted import propagation
from ad_afqmc.lno_afqmc import wavefunctions, sampling
from jax import numpy as jnp

print = partial(print, flush=True)

def prep_lnoafqmc(mf_cc,mo_coeff,options,norb_act,nelec_act,
                  prjlo=[],norb_frozen=[],ci1=None,ci2=None,chol_cut=1e-5,
                  option_file='options.bin',
                  mo_file="mo_coeff.npz",
                  amp_file="amplitudes.npz",
                  chol_file="FCIDUMP_chol"):
    
    
    with open(option_file, 'wb') as f:
        pickle.dump(options, f)
    
    if isinstance(mf_cc, (CCSD, UCCSD)):
        mf = mf_cc._scf
    else:
        mf = mf_cc
    
    ci2 = ci2.transpose(0, 2, 1, 3)
    if 'ci' in options['trial']:
        np.savez(amp_file, ci1=ci1, ci2=ci2)
    elif 'cc' in options['trial']:
        np.savez(amp_file, t1=ci1, t2=ci2)

    mol = mf.mol

    if isinstance(norb_frozen, (int, float)) and norb_frozen == 0:
        norb_frozen = []
    elif isinstance(norb_frozen, int):
        norb_frozen = np.arange(norb_frozen)
    act_idx = np.array([i for i in range(mol.nao) if i not in norb_frozen])
    
    print('\n')
    print('# Generating Cholesky Integrals')
    mc = mcscf.CASSCF(mf, norb_act, nelec_act)
    mc.frozen = norb_frozen
    nelec = mc.nelecas
    mc.mo_coeff = mo_coeff
    h1e, enuc = mc.get_h1eff()
    nbasis = h1e.shape[-1]

    if getattr(mf, "with_df", None) is not None:
        print("# Composing AO ERIs from DF basis")
        chol_df = df.incore.cholesky_eri(mol, mf.with_df.auxmol.basis)
        chol_df = lib.unpack_tril(chol_df).reshape(chol_df.shape[0], -1)
        chol_df = chol_df.reshape((-1, mol.nao, mol.nao))
        eri_ao_df = lib.einsum('lpq,lrs->pqrs', chol_df, chol_df, optimize='optimal')
        print("# Composing active space MO ERIs from AO ERIs")
        eri_mo_df = ao2mo.kernel(eri_ao_df,mo_coeff[:,act_idx],compact=False)
        eri_mo_df = eri_mo_df.reshape(nbasis**2,nbasis**2)
        print("# Decomposing MO ERIs to Cholesky vectors")
        print(f"# Cholesky cutoff is: {chol_cut}")
        chol = pyscf_interface.modified_cholesky(eri_mo_df,max_error=chol_cut)
    else:
        eri_mo = ao2mo.kernel(mf.mol,mo_coeff[:,act_idx],compact=False)
        chol = pyscf_interface.modified_cholesky(eri_mo,max_error=chol_cut)
    
    print("# Finished calculating Cholesky integrals")
    print('# Size of the correlation space')
    print(f'# Number of electrons: {nelec}')
    print(f'# Number of basis functions: {nbasis}')
    print(f'# Number of Cholesky vectors: {chol.shape[0]}')
    
    chol = chol.reshape((-1, nbasis, nbasis))
    v0 = 0.5 * lib.einsum('nik,njk->ij', chol, chol, optimize='optimal')
    h1e_mod = h1e - v0
    chol = chol.reshape((chol.shape[0], -1))

    np.savez(mo_file,prjlo=prjlo)

    write_dqmc(
        h1e,
        h1e_mod,
        chol,
        sum(nelec),
        nbasis,
        enuc,
        mf.e_tot,
        ms=mol.spin,
        filename=chol_file,
    )

    return None

def write_dqmc(
    hcore,
    hcore_mod,
    chol,
    nelec,
    nmo,
    enuc,
    emf,
    ms=0,
    filename="FCIDUMP_chol",
    mo_coeffs=None,
):
    # assert len(chol.shape) == 2
    with h5py.File(filename, "w") as fh5:
        fh5["header"] = np.array([nelec, nmo, ms, chol.shape[0]])
        fh5["hcore"] = hcore.flatten()
        fh5["hcore_mod"] = hcore_mod.flatten()
        fh5["chol"] = chol.flatten()
        fh5["energy_core"] = enuc
        fh5["emf"] = emf
        if mo_coeffs is not None:
            fh5["mo_coeffs_up"] = mo_coeffs[0]
            fh5["mo_coeffs_dn"] = mo_coeffs[1]


def _prep_afqmc(option_file="options.bin",
                mo_file="mo_coeff.npz",
                amp_file="amplitudes.npz",
                chol_file="FCIDUMP_chol"):
    
    try:
        with open(option_file, "rb") as f:
            options = pickle.load(f)
    except:
        print('# Using default options')
        options = {}

    options["dt"] = options.get("dt", 0.01)
    options["n_exp_terms"] = options.get("n_exp_terms",6)
    options["n_walkers"] = options.get("n_walkers", 50)
    options["n_prop_steps"] = options.get("n_prop_steps", 50)
    options["n_ene_blocks"] = options.get("n_ene_blocks", 10)
    options["n_sr_blocks"] = options.get("n_sr_blocks", 5)
    options["n_blocks"] = options.get("n_blocks", 50)
    options["seed"] = options.get("seed", np.random.randint(1, int(1e6)))
    options["n_eql"] = options.get("n_eql", 3)
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

    if options['use_gpu']:
        config.afqmc_config["use_gpu"] = True

    config.setup_jax()
    MPI = config.setup_comm()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print(f"# Number of MPI ranks: {size}\n#")

    trial = options['trial']
    if 'u' not in trial.lower():
        with h5py.File(chol_file, "r") as fh5:
            [nelec, nmo, ms, nchol] = fh5["header"]
            h0 = jnp.array(fh5.get("energy_core"))
            emf = jnp.array(fh5.get("emf"))
            h1 = jnp.array(fh5.get("hcore")).reshape(nmo, nmo)
            chol = jnp.array(fh5.get("chol")).reshape(-1, nmo, nmo)
            h1_mod = jnp.array(fh5.get("hcore_mod")).reshape(nmo, nmo)
    elif 'u' in trial.lower():
        with h5py.File(chol_file, "r") as fh5:
            [nelec, nmo, ms, nchol] = fh5["header"]
            h0 = jnp.array(fh5.get("energy_core"))
            emf = jnp.array(fh5.get("emf"))
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
    ham_data["E0"] = emf
    ham_data["ene0"] = options["ene0"]

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

    wave_data = {}
    prjlo = jnp.array(np.load(mo_file)["prjlo"])
    wave_data['prjlo'] = jnp.dot(prjlo.T,prjlo)
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
    elif options["trial"] == "cisd":
        try:
            amplitudes = np.load(amp_file)
            ci1 = jnp.array(amplitudes["ci1"])
            ci2 = jnp.array(amplitudes["ci2"])
            trial_wave_data = {"ci1": ci1, "ci2": ci2, 
                               "mo_coeff": mo_coeff[0][:, : nelec_sp[0]]}
            wave_data.update(trial_wave_data)
            trial = wavefunctions.cisd(norb, nelec_sp, n_batch=options["n_batch"])
        except:
            raise ValueError("Trial specified as cisd, but amplitudes.npz not found.")
    elif options["trial"] == "ccsd_pt_ad":
        trial = wavefunctions.ccsd_pt_ad(norb, nelec_sp, n_batch=options["n_batch"])
        amplitudes = np.load(amp_file)
        t1 = jnp.array(amplitudes["t1"])
        t2 = jnp.array(amplitudes["t2"])
        trial_wave_data = {"t1": t1, "t2": t2}
        wave_data.update(trial_wave_data)
        wave_data["mo_coeff"] = mo_coeff[0][:,:nelec_sp[0]]
    elif options["trial"] == "ccsd_pt2":
        trial = wavefunctions.ccsd_pt2(norb, nelec_sp, n_batch=options["n_batch"])
        nocc = nelec_sp[0]
        amplitudes = np.load(amp_file)
        t1 = jnp.array(amplitudes["t1"])
        t2 = jnp.array(amplitudes["t2"])
        trial_wave_data = {"t1": t1, "t2": t2}
        wave_data.update(trial_wave_data)
        mo_t = trial.thouless_trans(t1)[:,:nocc]
        wave_data['mo_t'] = mo_t
        wave_data["mo_coeff"] = mo_coeff[0][:,:nelec_sp[0]]
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
    if  'pt' in options['trial']:
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

    return ham_data, ham, prop, trial, wave_data, sampler, options, MPI

import os
def run_afqmc(options,nproc=None,
              option_file='options.bin'):

    with open(option_file, 'wb') as f:
        pickle.dump(options, f)

    use_gpu = options["use_gpu"]
    if use_gpu:
        print(f'# running AFQMC on GPU')
        config.afqmc_config = {"use_gpu": True}
        config.setup_jax()
        gpu_flag = "--use_gpu"
        mpi_prefix = ""
    else:
        print(f'# running AFQMC on CPU')
        gpu_flag = ""
        mpi_prefix = "mpirun "
        if nproc is not None:
            mpi_prefix += f"-np {nproc} "
    if  'pt' in options['trial']:
        if '2' in options['trial']:
            script='lno_afqmc_ccsd_pt2/run_lnoafqmc.py'
        else:
            script='lno_afqmc_ccsd_pt/run_lnoafqmc.py'
    else:
        script='lno_afqmc_cisd/run_lnoafqmc.py'

    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    script = f"{dir_path}/{script}"
    print(f'# AFQMC script: {script}')
    
    os.system(
        f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1;"
        f"{mpi_prefix} python {script} {gpu_flag} |tee lno_afqmc.out"
    )

from pyscf.ci.cisd import CISD
from pyscf.cc.ccsd import CCSD
from pyscf import lib
import jax
from ad_afqmc.lno.cc import LNOCCSD
from ad_afqmc.lno_afqmc import lno_maker, lno_afqmc
from ad_afqmc.lno.base import lno

def run_lnoafqmc(mfcc,options,frozen=None,
                 lno_thresh=1e-5,chol_cut=1e-5,
                 run_frg_list=None,nproc=None):

    if isinstance(mfcc, (CCSD, CISD)):
        mf = mfcc._scf
    else:
        mf = mfcc

    if isinstance(lno_thresh, list):
        thresh_occ, thresh_vir = lno_thresh
    else:
        thresh_occ = lno_thresh*10
        thresh_vir = lno_thresh

    lno_cc = LNOCCSD(mf, frozen=frozen)
    lno_cc.thresh_occ = thresh_occ
    lno_cc.thresh_vir = thresh_vir
    lno_cc.lo_type = 'boys'
    lno_cc.no_type = 'ie'
    no_type = 'ie'
    lno_cc.frag_lolist = '1o'
    lno_cc.force_outcore_ao2mo = True

    s1e = mf.get_ovlp()
    loc_occ = lno_cc.get_lo(lo_type='boys') # localized active occ orbitals
    # lococc,locvir = lno_maker.get_lo(lno_cc,lo_type) ### fix this for DF
    eris = lno_cc.ao2mo()

    frag_lolist = [[i] for i in range(loc_occ.shape[1])]
    print('localized occupied orbitals', frag_lolist)
    nfrag = len(frag_lolist)

    frozen_mask = lno_cc.get_frozen_mask()
    thresh_pno = [thresh_occ,thresh_vir]
    print(f'lno thresh {thresh_pno}')

    if run_frg_list is None:
        run_frg_list = range(nfrag)

    frag_nonvlist = None
    if frag_nonvlist is None: frag_nonvlist = lno_cc.frag_nonvlist
    if frag_nonvlist is None: frag_nonvlist = [[None,None]] * nfrag

    nelec_list = np.empty(len(run_frg_list),dtype='float64')
    norb_list = np.empty(len(run_frg_list),dtype='float64')
    eorb_p2 = np.empty(len(run_frg_list),dtype='float64')
    eorb_cc = np.empty(len(run_frg_list),dtype='float64')
        
    from jax import random
    seeds = random.randint(random.PRNGKey(options["seed"]),
                        shape=(len(run_frg_list),), minval=0, maxval=100*nfrag)
    options["max_error"] = options["max_error"]/np.sqrt(len(run_frg_list))

    for n,ifrag in enumerate(run_frg_list):
        print(f'\n########### running fragment {ifrag+1} ##########')

        fraglo = frag_lolist[ifrag]
        orbfragloc = loc_occ[:,fraglo]
        THRESH_INTERNAL = 1e-10
        # frag_target_nocc, frag_target_nvir = frag_nonvlist[ifrag]
        frzfrag, orbfrag, can_orbfrag \
            = lno.make_fpno1(lno_cc, eris, orbfragloc, no_type,
                                THRESH_INTERNAL, thresh_pno,
                                frozen_mask=frozen_mask,
                                frag_target_nocc=None,
                                frag_target_nvir=None,
                                canonicalize=True)

        mol = mf.mol
        nocc = mol.nelectron // 2 
        nao = mol.nao
        actfrag = np.array([i for i in range(nao) if i not in frzfrag])
        # frzocc = np.array([i for i in range(nocc) if i in frzfrag])
        actocc = np.array([i for i in range(nocc) if i in actfrag])
        actvir = np.array([i for i in range(nocc,nao) if i in actfrag])
        nactocc = len(actocc)
        nactocc = len(actocc)
        nactvir = len(actvir)
        prjlo = orbfragloc.T @ s1e @ orbfrag[:,actocc]
        nelec_act = nactocc*2
        norb_act = nactocc+nactvir

        print(f'# active orbitals: {actfrag}')
        print(f'# active occupied orbitals: {actocc}')
        print(f'# active virtual orbitals: {actvir}')
        print(f'# frozen orbitals: {frzfrag}')
        print(f'# number of active electrons: {nelec_act}')
        print(f'# number of active orbitals: {norb_act}')
        print(f'# number of frozen orbitals: {len(frzfrag)}')

        nelec_list[n] = nelec_act
        norb_list[n] = norb_act

        # mp2 is not invariant to lno transformation
        # needs to be done in canoical HF orbitals
        # which the globel mp2 is calculated in
        print('# running fragment MP2')
        ecorr_p2 = \
            lno_maker.lno_mp2_frg_e(mf,frzfrag,orbfragloc,can_orbfrag)
        eorb_p2[n] = ecorr_p2
        ecorr_p2 = f'{ecorr_p2:.8f}'
        print(f'# LNO-MP2 Orbital Energy: {ecorr_p2}')
        
        print('# running fragment CCSD')
        mcc, ecorr_cc = \
            lno_maker.lno_cc_solver(mf,orbfrag,orbfragloc,frozen=frzfrag)
        eorb_cc[n] = ecorr_cc
        ecorr_cc = f'{ecorr_cc:.8f}'
        print(f'# LNO-CCSD Energy: {mcc.e_tot}')
        print(f'# LNO-CCSD Orbital Energy: {ecorr_cc}')

        from mpi4py import MPI
        if not MPI.Is_finalized():
            MPI.Finalize()

        if 'ci' in options['trial']:
            ci1 = np.array(mcc.t1)
            ci2 = mcc.t2 + lib.einsum("ia,jb->ijab",ci1,ci1)
        elif 'cc' in options['trial']:
            ci1 = np.array(mcc.t1)
            ci2 = mcc.t2

        options["seed"] = seeds[n]
        lno_afqmc.prep_lnoafqmc(
            mf,orbfrag,options,
            norb_act=norb_act,nelec_act=nelec_act,
            prjlo=prjlo,norb_frozen=frzfrag,
            ci1=ci1,ci2=ci2,chol_cut=chol_cut,
            )
        lno_afqmc.run_afqmc(options,nproc)
        os.system(f'mv lno_afqmc.out lno_afqmc.out{ifrag+1}')

    from pyscf import mp
    mmp = mp.MP2(mf, frozen=frozen)
    e_mp2tot = mmp.kernel()[0]

    if 'ci' in options['trial']:
        eorb = np.empty(len(run_frg_list),dtype='float64')
        eorb_err = np.empty(len(run_frg_list),dtype='float64')
        run_time = np.empty(len(run_frg_list),dtype='float64')
        for i in run_frg_list:
            with open(f"lno_afqmc.out{i+1}", "r") as rf:
                for line in rf:
                    if "AFQMC/CISD E_Orbital" in line:
                        eorb[n] = float(line.split()[-3])
                        eorb_err[n] = float(line.split()[-1])
                    if "total run time" in line:
                        run_time[n] = float(line.split()[-1])
        nelec = np.mean(nelec_list)
        norb = np.mean(norb_list)
        e_mp2 = sum(eorb_p2)
        e_ccsd = sum(eorb_cc)
        e_afqmc_cisd = sum(eorb)
        e_afqmc_cisd_err = np.sqrt(sum(eorb_err**2))
        tot_time = sum(run_time)

        with open(f'lno_result.out', 'w') as out_file:
            print('# frag  eorb_mp2  eorb_ccsd  eorb_afqmc/cisd  nelec  norb  time',
                  file=out_file)
            for n,i in enumerate(run_frg_list):
                print(f'{i+1:3d}  '
                      f'{eorb_p2[n]:.8f}  {eorb_cc[n]:.8f}  '
                      f'{eorb[n]:.6f} +/- {eorb_err[n]:.6f}  '
                      f'{nelec_list[n]}  {norb_list[n]}  {run_time[n]:.2f}', file=out_file)
            print(f'# LNO Thresh: {thresh_pno}',file=out_file)
            print(f'# LNO Average Active Space: ({nelec:.1f},{norb:.1f})',file=out_file)
            print(f'# LNO-MP2 Energy: {e_mp2:.8f}',file=out_file)
            print(f'# LNO-CCSD Energy: {e_ccsd:.8f}',file=out_file)
            print(f'# LNO-AFQMC/CISD Energy: {e_afqmc_cisd:.6f} +/- {e_afqmc_cisd_err:.6f}',file=out_file)
            print(f'# MP2 Correction: {e_mp2tot-e_mp2:.8f}',file=out_file)
            print(f"# total run time: {tot_time:.2f}",file=out_file)

    if 'cc' in options['trial']:
        eorb0 = np.empty(len(run_frg_list),dtype='float64')
        eorb0_err = np.empty(len(run_frg_list),dtype='float64')
        eorb = np.empty(len(run_frg_list),dtype='float64')
        eorb_err = np.empty(len(run_frg_list),dtype='float64')
        run_time = np.empty(len(run_frg_list),dtype='float64')
        for n,i in enumerate(run_frg_list):
            with open(f"lno_afqmc.out{i+1}", "r") as rf:
                for line in rf:
                    if "AFQMC/HF E_Orbital" in line:
                        eorb0[n] = float(line.split()[-3])
                        eorb0_err[n] = float(line.split()[-1])
                    if "AFQMC/CCSD_PT E_Orbital" in line:
                        eorb[n] = float(line.split()[-3])
                        eorb_err[n] = float(line.split()[-1])
                    if "total run time" in line:
                        run_time[n] = float(line.split()[-1])

        nelec = np.mean(nelec_list)
        norb = np.mean(norb_list)
        e_mp2 = sum(eorb_p2)
        e_ccsd = sum(eorb_cc)
        e_afqmc_hf = sum(eorb0) 
        e_afqmc_hf_err = np.sqrt(sum(eorb0_err**2))
        e_afqmc_pt = sum(eorb)
        e_afqmc_pt_err = np.sqrt(sum(eorb_err**2))
        tot_time = sum(run_time)

        with open(f'lno_result.out', 'w') as out_file:
            print('# frag  eorb_mp2  eorb_ccsd  eorb_afqmc/hf  eorb_afqmc/ccsd_pt  nelec  norb  time',
                  file=out_file)
            for n,i in enumerate(run_frg_list):
                print(f'{i+1:3d}  '
                      f'{eorb_p2[n]:.8f}  {eorb_cc[n]:.8f}  '
                      f'{eorb0[n]:.6f} +/- {eorb0_err[n]:.6f}  '
                      f'{eorb[n]:.6f} +/- {eorb_err[n]:.6f}  '
                      f'{nelec_list[n]}  {norb_list[n]}  {run_time[n]:.2f}', file=out_file)
            print(f'# LNO Thresh: {thresh_pno}',file=out_file)
            print(f'# LNO Average Active Space: ({nelec:.1f},{norb:.1f})',file=out_file)
            print(f'# LNO-MP2 Energy: {e_mp2:.8f}',file=out_file)
            print(f'# LNO-CCSD Energy: {e_ccsd:.8f}',file=out_file)
            print(f'# LNO-AFQMC/HF Energy: {e_afqmc_hf:.6f} +/- {e_afqmc_hf_err:.6f}',file=out_file)
            print(f'# LNO-AFQMC/CCSD_PT Energy: {e_afqmc_pt:.6f} +/- {e_afqmc_pt_err:.6f}',file=out_file)
            print(f'# MP2 Correction: {e_mp2tot-e_mp2:.8f}',file=out_file)
            print(f"# total run time: {tot_time:.2f}",file=out_file)
    return None