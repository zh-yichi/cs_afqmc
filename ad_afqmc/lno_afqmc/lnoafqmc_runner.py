import sys, os
import pickle
import h5py
import numpy as np
from jax import numpy as jnp
from jax import random
from pyscf.ci.cisd import CISD
from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD
from pyscf import ao2mo, mcscf, scf, lib
from ad_afqmc import pyscf_interface, wavefunctions, config
from ad_afqmc import hamiltonian, propagation, sampling
from ad_afqmc.lno.cc import LNOCCSD
from ad_afqmc.lno.base import lno
from ad_afqmc.lno_afqmc import lno_maker, afqmc_maker, data_maker

def write_dqmc(E0,hcore,hcore_mod,chol,nelec,nmo,enuc,ms=0,
    filename="FCIDUMP_chol",mo_coeffs=None):
    assert len(chol.shape) == 2
    with h5py.File(filename, "w") as fh5:
        fh5["header"] = np.array([nelec, nmo, ms, chol.shape[0]])
        fh5["hcore"] = hcore.flatten()
        fh5["hcore_mod"] = hcore_mod.flatten()
        fh5["chol"] = chol.flatten()
        fh5["energy_core"] = enuc
        fh5["E0"] = E0
        if mo_coeffs is not None:
            fh5["mo_coeffs_up"] = mo_coeffs[0]
            fh5["mo_coeffs_dn"] = mo_coeffs[1]

def prep_lnoafqmc_file(mf_cc,mo_coeff,options,norb_act,nelec_act,
                       prjlo=[],norb_frozen=[],ci1=None,ci2=None,
                       use_df_vecs=False,chol_cut=1e-6,
                       option_file='options.bin',
                       mo_file="mo_coeff.npz",
                       amp_file="amplitudes.npz",
                       chol_file="FCIDUMP_chol"):
    
    
    with open(option_file, 'wb') as f:
        pickle.dump(options, f)
    
    # write ccsd amplitudes
    if isinstance(mf_cc, (CCSD, UCCSD)):
        mf = mf_cc._scf
    else:
        mf = mf_cc

    if options["trial"] == "cisd":
        np.savez(amp_file, ci1=ci1, ci2=ci2)

    mol = mf.mol
    # calculate cholesky vectors
    h1e, chol, nelec, enuc, nbasis, nchol = [ None ] * 6
    print('# Generating Cholesky Integrals')

    mc = mcscf.CASSCF(mf, norb_act, nelec_act) 
    mc.frozen = norb_frozen
    nelec = mc.nelecas
    mc.mo_coeff = mo_coeff
    h1e, enuc = mc.get_h1eff()

    nbasis = h1e.shape[-1]
    if isinstance(norb_frozen, (int, float)) and norb_frozen == 0:
        norb_frozen = []
    elif isinstance(norb_frozen, int):
        norb_frozen = np.arange(norb_frozen)
    print(f'# frozen orbitals: {norb_frozen}')
    act = np.array([i for i in range(mol.nao) if i not in norb_frozen])
    print(f'# local active orbitals: {act}') #yichi
    print(f'# local active space size: {len(act)}') #yichi
    
    if getattr(mf, "with_df", None) is not None:
        if use_df_vecs:
            print('# use DF vectors as Cholesky vectors')
            _, chol, _, _ = pyscf_interface.generate_integrals(
                mol,mf.get_hcore(),mo_coeff[:,act],DFbas=mf.with_df.auxmol.basis)
        else:
            print("# composing ERIs from DF basis")
            from pyscf import df
            chol_df = df.incore.cholesky_eri(mol, mf.with_df.auxmol.basis)
            chol_df = lib.unpack_tril(chol_df).reshape(chol_df.shape[0], -1)
            chol_df = chol_df.reshape((-1, mol.nao, mol.nao))
            eri_ao_df = lib.einsum('lpq,lrs->pqrs', chol_df, chol_df)
            #eri_df.reshape((nao*nao, nao*nao))
            print("# decomposing ERIs to Cholesky vectors")
            print(f"# Cholesky cutoff is: {chol_cut}")
            eri_mo_df = ao2mo.kernel(eri_ao_df,mo_coeff[:,act],compact=False)
            eri_mo_df = eri_mo_df.reshape(nbasis**2,nbasis**2)
            chol = pyscf_interface.modified_cholesky(eri_mo_df,max_error=chol_cut)
    else:
        eri_mo = ao2mo.kernel(mf.mol,mo_coeff[:,act],compact=False)
        chol = pyscf_interface.modified_cholesky(eri_mo,max_error=chol_cut)
    
    print("# Finished calculating Cholesky integrals\n")
    print('# Size of the correlation space')
    print(f'# Number of electrons: {nelec}')
    print(f'# Number of basis functions: {nbasis}')
    print(f'# Number of Cholesky vectors: {chol.shape[0]}\n')
    
    chol = chol.reshape((-1, nbasis, nbasis))
    v0 = 0.5 * lib.einsum('nik,njk->ij', chol, chol, optimize='optimal')
    h1e_mod = h1e - v0
    chol = chol.reshape((chol.shape[0], -1))

    # write mo coefficients
    trial_coeffs = np.empty((2, nbasis, nbasis))
    # overlap = mf.get_ovlp(mol)
    if isinstance(mf, (scf.uhf.UHF, scf.rohf.ROHF)):
        uhfCoeffs = np.empty((nbasis, 2 * nbasis))
        overlap = mf.get_ovlp(mol)
        if isinstance(mf, scf.uhf.UHF):
            q, r = np.linalg.qr(mo_coeff[:, norb_frozen:].T.dot(overlap).dot(mf.mo_coeff[0][:, norb_frozen:]))
            uhfCoeffs[:, :nbasis] = q
            q, r = np.linalg.qr(mo_coeff[:, norb_frozen:].T.dot(overlap).dot(mf.mo_coeff[1][:, norb_frozen:]))
            uhfCoeffs[:, nbasis:] = q
        else:
            q, r = np.linalg.qr(mo_coeff[:, norb_frozen:].T.dot(overlap).dot(mf.mo_coeff[:, norb_frozen:]))
            uhfCoeffs[:, :nbasis] = q
            uhfCoeffs[:, nbasis:] = q

        trial_coeffs[0] = uhfCoeffs[:, :nbasis]
        trial_coeffs[1] = uhfCoeffs[:, nbasis:]
        np.savez(mo_file,mo_coeff=trial_coeffs,prjlo=prjlo)

    elif isinstance(mf, scf.rhf.RHF):
        #q, r = np.linalg.qr(mo_coeff[:, norb_frozen:].T.dot(overlap).dot(mf.mo_coeff[:, norb_frozen:]))
        q = np.eye(mol.nao- len(norb_frozen)) # in mo basis
        trial_coeffs[0] = q
        trial_coeffs[1] = q
        np.savez(mo_file,mo_coeff=trial_coeffs,prjlo=prjlo)

    write_dqmc(mf.e_tot,h1e,h1e_mod,chol,sum(nelec),nbasis,enuc,ms=mol.spin,
               filename=chol_file,mo_coeffs=trial_coeffs)
    
    return None

def prep_lnoafqmc_run(options=None,prjlo=True,
                      option_file="options.bin",
                      mo_file="mo_coeff.npz",
                      amp_file="amplitudes.npz",
                      chol_file="FCIDUMP_chol"):
    
    if options is None:
        try:
            with open(option_file, "rb") as f:
                options = pickle.load(f)
        except:
            options = {}
            
    options["dt"] = options.get("dt", 0.005)
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
    options["trial"] = options.get("trial", "cisd")
    options["free_projection"] = options.get("free_projection", False)
    options["n_batch"] = options.get("n_batch", 1)
    options["ene0"] = options.get("ene0",0)
    options["use_gpu"] = options.get("use_gpu", False)

    if options["use_gpu"]:
        config.afqmc_config["use_gpu"] = True
    config.setup_jax()
    MPI = config.setup_comm()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    if rank == 0:
        print(f"# Number of MPI ranks: {size}\n#")

    with h5py.File(chol_file, "r") as fh5:
        [nelec, nmo, ms, nchol] = fh5["header"]
        h0 = jnp.array(fh5.get("energy_core"))
        e0 = jnp.array(fh5.get("E0"))
        h1 = jnp.array(fh5.get("hcore")).reshape(nmo, nmo)
        chol = jnp.array(fh5.get("chol")).reshape(-1, nmo, nmo)

    assert type(ms) is np.int64
    assert type(nelec) is np.int64
    assert type(nmo) is np.int64
    assert type(nchol) is np.int64
    ms, nelec, nmo, nchol = int(ms), int(nelec), int(nmo), int(nchol)
    nelec_sp = ((nelec + abs(ms)) // 2, (nelec - abs(ms)) // 2)
    norb = nmo

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
    ham_data["E0"] = e0
    ham_data["h1"] = jnp.array([h1, h1])
    ham_data["chol"] = chol.reshape(nchol, -1)
    ham_data["ene0"] = options["ene0"]

    wave_data = {}
    if prjlo is True:
        prjlo = jnp.array(np.load(mo_file)["prjlo"])
    else: prjlo = None
    wave_data["prjlo"] = prjlo
    mo_coeff = jnp.array(np.load(mo_file)["mo_coeff"])
    wave_data["rdm1"] = jnp.array(
        [
            mo_coeff[0][:, : nelec_sp[0]] @ mo_coeff[0][:, : nelec_sp[0]].T,
            mo_coeff[1][:, : nelec_sp[1]] @ mo_coeff[1][:, : nelec_sp[1]].T,
        ]
    )

    if options["trial"] == "rhf":
        trial = wavefunctions.rhf(norb, nelec_sp, n_batch=options["n_batch"])
        wave_data["mo_coeff"] = mo_coeff[0][:, : nelec_sp[0]]
    # wave_data["mo_coeff"] = mo_coeff[0][:, : nelec_sp[0]]
    elif options["trial"] == "uhf":
        trial = wavefunctions.uhf(norb, nelec_sp, n_batch=options["n_batch"])
        wave_data["mo_coeff"] = [
            mo_coeff[0][:, : nelec_sp[0]],
            mo_coeff[1][:, : nelec_sp[1]],
        ]
    
    elif options["trial"] == "cisd":
        wave_data["mo_coeff"] = mo_coeff[0][:, : nelec_sp[0]]
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
        print(f"# nchol: {nchol}")
        print("#")
        for op in options:
            if options[op] is not None:
                print(f"# {op}: {options[op]}")
        print("#")

    return ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI

def run_lno_afqmc(mfcc,thresh,frozen=None,options=None,
                  lo_type='boys',chol_cut=1e-6,nproc=None,
                  run_frg_list=None,use_df_vecs=False,mp2=True,
                  debug=False,t2_0=False):
    '''
    mfcc: pyscf mean-field object
    thresh: lno thresh
    frozen: frozen orbitals
    options: afqmc options
    chol_cut: Cholesky Decomposition cutoff
    nproc: number of processors
    run_frg_list: list of the fragments to run
    '''

    if isinstance(mfcc, (CCSD, CISD)):
        full_cisd = True
        mf = mfcc._scf
    else:
        full_cisd = False
        mf = mfcc

    no_type = 'ie' # cim

    if isinstance(thresh, list):
        thresh_occ, thresh_vir = thresh
    else:
        thresh_occ = thresh*10
        thresh_vir = thresh

    lno_cc = LNOCCSD(mf, thresh=thresh, frozen=frozen)
    lno_cc.thresh_occ = thresh_occ
    lno_cc.thresh_vir = thresh_vir
    lno_cc.lo_type = lo_type
    lno_cc.no_type = no_type
    lno_cc.force_outcore_ao2mo = True
    lno_cc.frag_lolist = '1o'

    s1e = mf.get_ovlp()
    eris = lno_cc.ao2mo()
    lococc = lno_cc.get_lo(lo_type=lo_type) # localized active occ orbitals
    # lococc,locvir = lno_maker.get_lo(lno_cc,lo_type) ### fix this for DF

    frag_lolist = [[i] for i in range(lococc.shape[1])]
    nfrag = len(frag_lolist)

    frozen_mask = lno_cc.get_frozen_mask()
    thresh_pno = [thresh_occ,thresh_vir]
    print(f'# lno thresh {thresh_pno}')
    
    if run_frg_list is None:
        run_frg_list = range(nfrag)
    
    seeds = random.randint(random.PRNGKey(options["seed"]),
                        shape=(len(run_frg_list),), minval=0, maxval=100*nfrag)
    
    for ifrag in run_frg_list:
        print(f'\n########### running fragment {ifrag+1} ##########')

        fraglo = frag_lolist[ifrag]
        orbfragloc = lococc[:,fraglo]
        THRESH_INTERNAL = 1e-10
        # frzfrag, orbfrag, can_orbfrag \
        #     = lno_maker.make_lno(
        #         lno_cc,orbfragloc,THRESH_INTERNAL,thresh_pno
        #         )
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

        print(f'# active orbitals: {actfrag}')
        print(f'# active occupied orbitals: {actocc}')
        print(f'# active virtual orbitals: {actvir}')
        print(f'# frozen orbitals: {frzfrag}')

        if options["trial"] == "cisd":
            if full_cisd:
                # print('# This method is not size-extensive')
                frz_mo_idx = np.where(np.array(frozen_mask) == False)[0]
                act_mo_occ = np.array([i for i in range(nocc) if i not in frz_mo_idx])
                act_mo_vir = np.array([i for i in range(nocc,nao) if i not in frz_mo_idx])
                prj_no2mo = afqmc_maker.no2mo(mf.mo_coeff,s1e,orbfrag)
                prj_oo_act = prj_no2mo[np.ix_(act_mo_occ,actocc)]
                prj_vv_act = prj_no2mo[np.ix_(act_mo_vir,actvir)]
                if isinstance(mfcc, CCSD):
                    print('# Use full CCSD wavefunction')
                    print('# Project CC amplitudes from MO to NO')
                    t1 = mfcc.t1
                    t2 = mfcc.t2
                    if t2_0:
                        t2 = np.zeros(t2.shape)
                    # project to active no
                    t1 = lib.einsum("ij,ia,ba->jb",prj_oo_act,t1,prj_vv_act.T)
                    t2 = lib.einsum("ik,jl,ijab,db,ca->klcd",
                            prj_oo_act,prj_oo_act,t2,prj_vv_act.T,prj_vv_act.T)
                    ci1 = np.array(t1)
                    ci2 = t2 + lib.einsum("ia,jb->ijab",ci1,ci1)
                    ci2 = ci2.transpose(0, 2, 1, 3)
                if isinstance(mfcc, CISD):
                    print('# Use full CISD wavefunction')
                    print('# Project CI coefficients from MO to NO')
                    v_ci = mfcc.ci
                    ci0,ci1,ci2 = mfcc.cisdvec_to_amplitudes(v_ci)
                    ci1 = ci1/ci0
                    ci2 = ci2/ci0
                    ci1 = lib.einsum("ij,ia,ba->jb",prj_oo_act,ci1,prj_vv_act.T)
                    ci2 = lib.einsum("ik,jl,ijab,db,ca->klcd",
                            prj_oo_act,prj_oo_act,ci2,prj_vv_act.T,prj_vv_act.T)
                    ci2 = ci2.transpose(0, 2, 1, 3)
                print('# Finished MO to NO projection')
                ecorr_cc = '  None  '
            else:
                print('# Solving LNO-CCSD')
                ecorr_cc,t1,t2 = lno_maker.lno_cc_solver(
                        mf,orbfrag,orbfragloc,frozen=frzfrag,eris=None
                        )
                if t2_0:
                    t2 = np.zeros(t2.shape)
                ci1 = np.array(t1)
                ci2 = t2 + lib.einsum("ia,jb->ijab",ci1,ci1)
                ci2 = ci2.transpose(0, 2, 1, 3)
                ecorr_cc = f'{ecorr_cc:.8f}'
                from mpi4py import MPI
                if not MPI.Is_finalized():
                    MPI.Finalize() # CCSD initializes MPI
        #MP2 correction 
        if mp2:
            ## mp2 is not invariant to lno transformation
            ## needs to be done in canoical HF orbitals
            ## which the globel mp2 is calculated in
            print('# running fragment MP2')
            ecorr_p2 = \
                lno_maker.lno_mp2_frg_e(mf,frzfrag,orbfragloc,can_orbfrag)
            ecorr_p2 = f'{ecorr_p2:.8f}'
        else:
            ecorr_p2 = '  None  '

        nelec_act = nactocc*2
        norb_act = nactocc+nactvir
        
        print(f'# number of active electrons: {nelec_act}')
        print(f'# number of active orbitals: {norb_act}')
        print(f'# number of frozen orbitals: {len(frzfrag)}')
        print(f'# lno-mp2 fragment correlation energy: {ecorr_p2}')
        print(f'# lno-ccsd fragment correlation energy: {ecorr_cc}')

        options["seed"] = seeds[ifrag]
        prep_lnoafqmc_file(
            mf,orbfrag,options,
            norb_act=norb_act,nelec_act=nelec_act,
            prjlo=prjlo,norb_frozen=frzfrag,
            ci1=ci1,ci2=ci2,use_df_vecs=use_df_vecs,
            chol_cut=chol_cut,
            )
        
        # Run AFQMC
        use_gpu = options["use_gpu"]
        if use_gpu:
            print(f'# running AFQMC on GPU')
            gpu_flag = "--use_gpu"
            mpi_prefix = ""
            nproc = None
        else:
            print(f'# running AFQMC on CPU')
            gpu_flag = ""
            mpi_prefix = "mpirun "
            if nproc is not None:
                mpi_prefix += f"-np {nproc} "
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)   
        if debug:
            script = f"{dir_path}/run_lnocc_frg_dbg.py"
        else:
            script = f"{dir_path}/run_lnocc_frg.py"
        os.system(
            f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1;"
            f"{mpi_prefix} python {script} {gpu_flag} |tee frg_{ifrag+1}.out"
        )

        with open(f'frg_{ifrag+1}.out', 'a') as out_file:
            print(f"lno-mp2 orb_corr: {ecorr_p2}",file=out_file)
            print(f"lno-ccsd orb_corr: {ecorr_cc}",file=out_file)
            print(f"number of active electrons: {nelec_act}",file=out_file)
            print(f"number of active orbitals: {norb_act}",file=out_file)

    # from pyscf import mp
    # mmp = mp.MP2(mf, frozen=frozen)
    # e_mp2 = mmp.kernel()[0]

    if debug:
        data_maker.frg2result_dbg(thresh_pno,len(run_frg_list),mf.e_tot,e_mp2=0)
    else:
        data_maker.frg2result(thresh_pno,len(run_frg_list),mf.e_tot,e_mp2=0)

    return None