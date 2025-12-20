import os, h5py, pickle
import numpy as np
from jax import numpy as jnp
from jax import random
from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD
from pyscf import lib, ao2mo, df, mp
from pyscf.lno import ulnoccsd
from ad_afqmc import config, pyscf_interface
from functools import partial
from ad_afqmc.lno_afqmc import wavefunctions, propagation, sampling
from collections.abc import Iterable
print = partial(print, flush=True)
from functools import reduce

def ulno_ccsd(mcc, mo_coeff, uocc_loc, mo_occ, maskact, ccsd_t=False):

    occidxa = mo_occ[0]>1e-10
    occidxb = mo_occ[1]>1e-10
    nmo = mo_occ[0].size, mo_occ[1].size
    moidxa, moidxb = maskact

    orbfrzocca = mo_coeff[0][:, ~moidxa &  occidxa]
    orbactocca = mo_coeff[0][:,  moidxa &  occidxa]
    orbactvira = mo_coeff[0][:,  moidxa & ~occidxa]
    orbfrzvira = mo_coeff[0][:, ~moidxa & ~occidxa]
    nfrzocca, nactocca, nactvira, nfrzvira = [orb.shape[1]
                                              for orb in [orbfrzocca,orbactocca,
                                                          orbactvira,orbfrzvira]]
    orbfrzoccb = mo_coeff[1][:, ~moidxb &  occidxb]
    orbactoccb = mo_coeff[1][:,  moidxb &  occidxb]
    orbactvirb = mo_coeff[1][:,  moidxb & ~occidxb]
    orbfrzvirb = mo_coeff[1][:, ~moidxb & ~occidxb]
    nfrzoccb, nactoccb, nactvirb, nfrzvirb = [orb.shape[1]
                                              for orb in [orbfrzoccb,orbactoccb,
                                                          orbactvirb,orbfrzvirb]]
    nlo = [uocc_loc[0].shape[1], uocc_loc[1].shape[1]]
    prjlo = [uocc_loc[0].T.conj(), uocc_loc[1].T.conj()]
    if nactocca * nactvira == 0 and nactoccb * nactvirb == 0:
        elcorr_pt2 = lib.tag_array(0., spin_comp=np.array((0., 0.)))
        elcorr_cc = lib.tag_array(0., spin_comp=np.array((0., 0.)))
        elcorr_cc_t = 0.
    else:
        # solve impurity problem
        imp_eris = mcc.ao2mo()
        # MP2 fragment energy
        t1, t2 = mcc.init_amps(eris=imp_eris)[1:]
        elcorr_pt2 = ulnoccsd.get_fragment_energy(imp_eris, t1, t2, prjlo)
        # CCSD fragment energy
        t1, t2 = mcc.kernel(eris=imp_eris, t1=t1, t2=t2)[1:]
        elcorr_cc = ulnoccsd.get_fragment_energy(imp_eris, t1, t2, prjlo)
        if ccsd_t:
            from pyscf.lno.ulnoccsd_t_slow import kernel as UCCSD_T
            elcorr_cc_t = UCCSD_T(mcc, imp_eris, prjlo, t1=t1, t2=t2)
        else:
            elcorr_cc_t = 0.

        t1_0 = [t1[0]*0, t1[1]*0]
        ecct1_0 = ulnoccsd.get_fragment_energy(imp_eris, t1_0, t2, prjlo)
        print('# LNO-UCCSD (T1 = 0) fragment energy:', ecct1_0)
        t2_0 = [t2[0]*0, t2[1]*0, t2[2]*0]
        ecct2_0 = ulnoccsd.get_fragment_energy(imp_eris, t1, t2_0, prjlo)
        print('# LNO-UCCSD (T2 = 0) fragment energy:', ecct2_0)

    imp_eris = None
    t1_0 = t2_0 = None

    return (elcorr_pt2, elcorr_cc, elcorr_cc_t ,ecct1_0, ecct2_0), t1, t2

def get_veff(mf, dm):
    mol = mf.mol
    vj, vk = mf.get_jk(mol, dm, hermi=1)
    return vj[0]+vj[1] - vk

def h1e_uas(mf, mo_coeff, ncas, ncore):
    '''
    effective one-electron integral for unrestricted active space
    ncas = (ncas_a, ncas_b) size of active space
    ncore = (ncore_a, ncore_b) number of core electrons
    '''
    # mf = mf.undo_df() ucasci undo DF

    mo_core =(mo_coeff[0][:,:ncore[0]], mo_coeff[1][:,:ncore[1]])
    mo_cas = (mo_coeff[0][:,ncore[0]:ncore[0]+ncas[0]],
              mo_coeff[1][:,ncore[1]:ncore[1]+ncas[1]])

    hcore = mf.get_hcore()
    hcore = [hcore, hcore]
    energy_core = mf.energy_nuc()
    if mo_core[0].size == 0 and mo_core[1].size == 0:
        corevhf = (0,0)
    else:
        core_dm = (np.dot(mo_core[0], mo_core[0].T),
                   np.dot(mo_core[1], mo_core[1].T))
        corevhf = get_veff(mf, core_dm)
        energy_core += np.einsum('ij,ji', core_dm[0], hcore[0])
        energy_core += np.einsum('ij,ji', core_dm[1], hcore[1])
        energy_core += np.einsum('ij,ji', core_dm[0], corevhf[0]) * .5
        energy_core += np.einsum('ij,ji', core_dm[1], corevhf[1]) * .5
    h1eff = (reduce(np.dot, (mo_cas[0].T, hcore[0]+corevhf[0], mo_cas[0])),
             reduce(np.dot, (mo_cas[1].T, hcore[1]+corevhf[1], mo_cas[1])))
    return h1eff, energy_core

def prjmo(prj,s1e,mo):
    # prj and reconstruct mo
    # e.g. |B_p> = |A_q><A_q|B_p>
    #            = C^A_mq C^A(T)_qn|m><n|s> C^B_sp
    mo_rec = prj @ prj.T @ s1e @ mo
    return mo_rec

def common_las(mf, lno_coeff, ncas, ncore, torr=1e-6):
    print("# Constracting cLAS that span both Alpha and Beta active space")
    s1e = mf.get_ovlp()
    lno_acta = lno_coeff[0][:,ncore[0]:ncore[0]+ncas[0]]
    lno_actb = lno_coeff[1][:,ncore[1]:ncore[1]+ncas[1]]
    lno_actaa = lno_coeff[0].T @ s1e @ lno_acta
    lno_actba = lno_coeff[0].T @ s1e @ lno_actb
    m = np.hstack([lno_actaa,lno_actba])
    u, s, _ = np.linalg.svd(m)
    print(f'# Common Active Space SVD Singular values:')
    print(s)
    print(f"# cLAS projection torr: {torr}")
    for idx in range(1,m.shape[1]+1):
        prj = lno_coeff[0] @ u[:,:idx]
        prj_acta = prjmo(prj,s1e,lno_actb)
        prj_actb = prjmo(prj,s1e,lno_acta)
        losa = abs(prj_acta-lno_actb).max()
        losb = abs(prj_actb-lno_acta).max()
        print(f"# cLAS projection loss: ({losa:.2e}, {losb:.2e})")
        if losa < torr and losb < torr:
            break
    print(f"# Minimum size of cLAS to span both Alpha and Beta LAS: {idx}")
    clas_coeff = lno_coeff[0] @ u[:,:idx]
    a2c = clas_coeff.T @ s1e @ lno_acta
    b2c = clas_coeff.T @ s1e @ lno_actb
    return clas_coeff, a2c, b2c


def prep_afqmc(mf_cc,mo_coeff,t1,t2,frozen,prjlo,
               options,chol_cut=1e-5,
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

    if 'ci' in options['trial']:
        ci2aa = t2[0] + 2 * np.einsum("ia,jb->ijab", t1[0], t1[0])
        ci2aa = (ci2aa - ci2aa.transpose(0, 1, 3, 2)) / 2
        ci2aa = ci2aa.transpose(0, 2, 1, 3)
        ci2bb = t2[2] + 2 * np.einsum("ia,jb->ijab", t1[1], t1[1])
        ci2bb = (ci2bb - ci2bb.transpose(0, 1, 3, 2)) / 2
        ci2bb = ci2bb.transpose(0, 2, 1, 3)
        ci2ab = t2[1] + np.einsum("ia,jb->ijab", t1[0], t1[1])
        ci2ab = ci2ab.transpose(0, 2, 1, 3)
        ci1a = np.array(t1[0])
        ci1b = np.array(t1[1])
        np.savez(amp_file,
                 ci1a=ci1a,
                 ci1b=ci1b,
                 ci2aa=ci2aa,
                 ci2ab=ci2ab,
                 ci2bb=ci2bb)
    elif 'cc' in options['trial']:
        t2aa = t2[0]
        t2aa = (t2aa - t2aa.transpose(0, 1, 3, 2)) / 2
        t2aa = t2aa.transpose(0, 2, 1, 3)
        t2bb = t2[2]
        t2bb = (t2bb - t2bb.transpose(0, 1, 3, 2)) / 2
        t2bb = t2bb.transpose(0, 2, 1, 3)
        t2ab = t2[1]
        t2ab = t2ab.transpose(0, 2, 1, 3)
        t1a = np.array(t1[0])
        t1b = np.array(t1[1])
        np.savez(amp_file,
                 t1a=t1a,
                 t1b=t1b,
                 t2aa=t2aa,
                 t2ab=t2ab,
                 t2bb=t2bb)

    print('# Calculating Effective Active Space One-electron Integrals')
    mol = mf.mol
    nocc_a = int(sum(mf.mo_occ[0]))
    actfrag_a = np.array([i for i in range(mol.nao) if i not in frozen[0]])
    frzocc_a = np.array([i for i in range(nocc_a) if i in frozen[0]])
    actocc_a = np.array([i for i in range(nocc_a) if i in actfrag_a])
    actvir_a = np.array([i for i in range(nocc_a,mol.nao) if i in actfrag_a])
    nfrzocc_a = len(frzocc_a)
    nactocc_a = len(actocc_a)
    nactvir_a = len(actvir_a)
    nactorb_a = len(actfrag_a)
    nocc_b = int(sum(mf.mo_occ[1]))
    actfrag_b = np.array([i for i in range(mol.nao) if i not in frozen[1]])
    frzocc_b = np.array([i for i in range(nocc_b) if i in frozen[1]])
    actocc_b = np.array([i for i in range(nocc_b) if i in actfrag_b])
    actvir_b = np.array([i for i in range(nocc_b,mol.nao) if i in actfrag_b])
    nfrzocc_b = len(frzocc_b)
    nactocc_b = len(actocc_b)
    nactvir_b = len(actvir_b)
    nactorb_b = len(actfrag_b)

    ncas = (nactorb_a, nactorb_b)
    ncore = (nfrzocc_a, nfrzocc_b)
    nelec = (nactocc_a, nactocc_b)
    h1e, enuc = h1e_uas(mf, mo_coeff, ncas, ncore)

    print('# Generating Cholesky Integrals')
    nao = mf.mol.nao
    # lno_acta = mo_coeff[0][:,ncore[0]:ncore[0]+ncas[0]]
    # lno_actb = mo_coeff[1][:,ncore[1]:ncore[1]+ncas[1]]
    # s1e = mf.get_ovlp()
    # a2b = mo_coeff[1].T @ s1e @ mo_coeff[0]
    clas_coeff, a2c, b2c = common_las(mf, mo_coeff, ncas, ncore, chol_cut)
    nclas = clas_coeff.shape[1]

    if getattr(mf, "with_df", None) is not None:
        # decompose eri in MO to achieve linear scale
        print("# Composing AO ERIs from DF basis")
        chol_df = df.incore.cholesky_eri(mol, mf.with_df.auxmol.basis)
        chol_df = lib.unpack_tril(chol_df).reshape(chol_df.shape[0], -1)
        chol_df = chol_df.reshape((-1, nao, nao))
        print(f'# DF Tensors shape: {chol_df.shape}')
        chol_df_clas = lib.einsum('pr,grs,sq->gpq',clas_coeff.T,chol_df,clas_coeff)
        eri_clas = lib.einsum('lpr,lqs->prqs', chol_df_clas, chol_df_clas, optimize='optimal')
        chol_df_clas = chol_df = None
        print("# Composing LAS ERIs from AO ERIs")
        # find the minimum common Local Active Space that spans both a and b
        # eri_clas = ao2mo.kernel(eri_ao,clas_coeff,compact=False,max_memory=mf.mol.max_memory)
        # eri_clas_half = lib.einsum("pu,xr,uxvw->prvw",clas_coeff.T,clas_coeff,eri_ao)
        # eri_clas = lib.einsum("prvw,qv,ws->prqs",eri_clas_half,clas_coeff.T,clas_coeff)
        print("# Finished Composing LAS ERIs")
        eri_clas = eri_clas.reshape(nclas**2,nclas**2)
        print("# Decomposing MO ERIs to Cholesky vectors")
        print(f"# Cholesky cutoff is: {chol_cut}")
        chol_clas = pyscf_interface.modified_cholesky(eri_clas,max_error=chol_cut)
        chol_clas = chol_clas.reshape((-1, nclas, nclas))
        chola = jnp.einsum('pr,grs,sq->gpq',a2c.T,chol_clas,a2c)
        cholb = jnp.einsum('pr,grs,sq->gpq',b2c.T,chol_clas,b2c)
        eri_clas = chol_clas = None
        # # # a2b = <B|A>
        # # # <B|L|B> = <B|A><A|L|A><A|B>
        # # # transform Lb from La so they have the same number of vectors
        # # chol_b = jnp.einsum('pr,grs,sq->gpq',a2b,chol_a,a2b.T)
        # # chol_b = chol_b.reshape((-1, nao, nao))
        # # <Ap|L|Aq> = <Ap|mu><mu|L|nu><nu|Aq>
        # chola = lib.einsum('pr,grs,sq->gpq',mo_coeff[0].T,chol_df,mo_coeff[0])
        # cholb = lib.einsum('pr,grs,sq->gpq',mo_coeff[1].T,chol_df,mo_coeff[1])
        # chola = chola[:,ncore[0]:ncore[0]+ncas[0],ncore[0]:ncore[0]+ncas[0]]
        # cholb = cholb[:,ncore[1]:ncore[1]+ncas[1],ncore[1]:ncore[1]+ncas[1]]
    else:
        raise  NotImplementedError('Use DF Only!')
        # eri_clas = ao2mo.kernel(mf.mol,clas_coeff,compact=False)
        # chol_clas = pyscf_interface.modified_cholesky(eri_clas,max_error=chol_cut)
        # chol_clas = chol_clas.reshape((-1, nclas, nclas))
        # chol_a = lib.einsum('pr,grs,sq->gpq',a2c.T,chol_clas,a2c)
        # chol_b = lib.einsum('pr,grs,sq->gpq',b2c.T,chol_clas,b2c)
    
    v0_a = 0.5 * jnp.einsum("nik,njk->ij", chola, chola, optimize="optimal")
    v0_b = 0.5 * jnp.einsum("nik,njk->ij", cholb, cholb, optimize="optimal")
    # h1e = jnp.array(h1e)
    h1mod_a = jnp.array(h1e[0] - v0_a)
    h1mod_b = jnp.array(h1e[1] - v0_b)

    print("# Finished calculating Cholesky integrals")
    print('# Size of the correlation space')
    print(f'# Number of electrons: {nelec}')
    print(f'# Number of basis functions: {ncas}')
    print(f'# Alpha Basis Cholesky shape: {chola.shape}')
    print(f'#  Beta Basis Cholesky shape: {cholb.shape}')
    
    chola.reshape(chola.shape[0], -1)
    cholb.reshape(cholb.shape[0], -1)
    
    np.savez(mo_file,prja=prjlo[0],prjb=prjlo[1])

    write_dqmc(h1e,[h1mod_a,h1mod_b],[chola, cholb],
               nelec,ncas,enuc,mf.e_tot,filename=chol_file)

    return nelec, ncas

def write_dqmc(
    h1e,
    h1e_mod,
    chol,
    nelec,
    nmo,
    enuc,
    emf,
    filename="FCIDUMP_chol"
):
    h1e_a, h1e_b = h1e
    h1mod_a, h1mod_b = h1e_mod
    chol_a, chol_b = chol
    with h5py.File(filename, "w") as fh5:
        fh5["header"] = np.array([nelec[0], nelec[1], nmo[0], nmo[1], chol_a.shape[0]])
        fh5["h1e_a"] = h1e_a.flatten()
        fh5["h1e_b"] = h1e_b.flatten()
        fh5["h1mod_a"] = h1mod_a.flatten()
        fh5["h1mod_b"] = h1mod_b.flatten()
        fh5["chol_a"] = chol_a.flatten()
        fh5["chol_b"] = chol_b.flatten()
        fh5["energy_core"] = enuc
        fh5["emf"] = emf


def _prep_afqmc(option_file="options.bin",
                mo_file="mo_coeff.npz",
                amp_file="amplitudes.npz",
                chol_file="FCIDUMP_chol"):
    
    # try:
    with open(option_file, "rb") as f:
        options = pickle.load(f)
    # except:
    #     print('# Using default options')
    #     options = {}

    options["dt"] = options.get("dt", 0.005)
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
    options['use_gpu'] = options.get("use_gpu", True)

    if options['use_gpu']:
        config.afqmc_config["use_gpu"] = True

    config.setup_jax()
    MPI = config.setup_comm()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print(f"# Number of MPI ranks: {size}\n#")

    with h5py.File(chol_file, "r") as fh5:
        [nelec_a,nelec_b,nmo_a,nmo_b,nchol] = fh5["header"]
        h0 = jnp.array(fh5.get("energy_core"))
        emf = jnp.array(fh5.get("emf"))
        h1_a = jnp.array(fh5.get("h1e_a")).reshape(nmo_a, nmo_a)
        h1_b = jnp.array(fh5.get("h1e_b")).reshape(nmo_b, nmo_b)
        h1mod_a = jnp.array(fh5.get("h1mod_a")).reshape(nmo_a, nmo_a)
        h1mod_b = jnp.array(fh5.get("h1mod_b")).reshape(nmo_b, nmo_b)
        chol_a = jnp.array(fh5.get("chol_a")).reshape(-1, nmo_a, nmo_a)
        chol_b = jnp.array(fh5.get("chol_b")).reshape(-1, nmo_b, nmo_b)

    assert chol_a.shape[0] == chol_b.shape[0]

    nelec_a, nelec_b, nmo_a, nmo_b, nchol \
        = int(nelec_a), int(nelec_b), int(nmo_a), int(nmo_b), int(nchol)
    nelec = (nelec_a, nelec_b)
    norb = (nmo_a, nmo_b)

    # ham = hamiltonian.hamiltonian(norb) # FIX
    ham_data = {}
    ham_data["h0"] = h0
    ham_data["E0"] = emf
    ham_data["ene0"] = options["ene0"]

    ham_data["h1"] = [jnp.array(h1_a), jnp.array(h1_b)]
    ham_data["h1_mod"] = [jnp.array(h1mod_a), jnp.array(h1mod_b)]
    ham_data["chol"] = [chol_a.reshape(chol_a.shape[0], -1),
                        chol_b.reshape(chol_b.shape[0], -1)]

    wave_data = {}
    prja = jnp.array(np.load(mo_file)["prja"])
    prjb = jnp.array(np.load(mo_file)["prjb"])
    wave_data['prjlo'] = [prja,prjb]
    mo_coeff_a = jnp.array(np.eye(nmo_a))
    mo_coeff_b = jnp.array(np.eye(nmo_b))
    wave_data["mo_coeff"] = [
            mo_coeff_a[:, : nelec[0]],
            mo_coeff_b[:, : nelec[1]],
        ]

    if options["trial"] == "uhf":
        trial = wavefunctions.uhf(norb, nelec, n_batch=options["n_batch"])
    elif options["trial"] == "uccsd_pt_ad":
        trial = wavefunctions.uccsd_pt_ad(norb, nelec, n_batch = options["n_batch"])
        amplitudes = np.load(amp_file)
        t1a = jnp.array(amplitudes["t1a"])
        t1b = jnp.array(amplitudes["t1b"])
        t2aa = jnp.array(amplitudes["t2aa"])
        t2ab = jnp.array(amplitudes["t2ab"])
        t2bb = jnp.array(amplitudes["t2bb"])
        prja, prjb = wave_data['prjlo']
        wave_data["t1a"] = jnp.einsum('ia,ik->ka',t1a,prja)
        wave_data["t1b"] = jnp.einsum('ia,ik->ka',t1b,prjb)
        wave_data["t2aa"] = jnp.einsum('iajb,ik->kajb',t2aa,prja)
        wave_data["t2ab"] = jnp.einsum('iajb,ik->kajb',t2ab,prja)
        wave_data["t2ba"] = jnp.einsum('jbia,ik->kajb',t2ab,prjb)
        wave_data["t2bb"] = jnp.einsum('iajb,ik->kajb',t2bb,prjb)
    elif options["trial"] == "uccsd_pt":
        trial = wavefunctions.uccsd_pt(norb, nelec, n_batch = options["n_batch"])
        amplitudes = np.load(amp_file)
        t1a = jnp.array(amplitudes["t1a"])
        t1b = jnp.array(amplitudes["t1b"])
        t2aa = jnp.array(amplitudes["t2aa"])
        t2ab = jnp.array(amplitudes["t2ab"])
        t2bb = jnp.array(amplitudes["t2bb"])
        prja, prjb = wave_data['prjlo']
        wave_data["t1a"] = jnp.einsum('ia,ik->ka',t1a,prja)
        wave_data["t1b"] = jnp.einsum('ia,ik->ka',t1b,prjb)
        wave_data["t2aa"] = jnp.einsum('iajb,ik->kajb',t2aa,prja)
        wave_data["t2ab"] = jnp.einsum('iajb,ik->kajb',t2ab,prja)
        wave_data["t2ba"] = jnp.einsum('jbia,ik->kajb',t2ab,prjb)
        wave_data["t2bb"] = jnp.einsum('iajb,ik->kajb',t2bb,prjb)
    elif "uccsd_pt2" in options["trial"]:
        nocca, noccb = nelec
        norba, norbb = norb
        amplitudes = np.load(amp_file)
        t1a = jnp.array(amplitudes["t1a"])
        t1b = jnp.array(amplitudes["t1b"])
        t2aa = jnp.array(amplitudes["t2aa"])
        t2ab = jnp.array(amplitudes["t2ab"])
        t2bb = jnp.array(amplitudes["t2bb"])
        t1a_full = np.zeros((norba, norba))
        t1a_full[:nocca, nocca:] = t1a
        t1b_full = np.zeros((norbb, norbb))
        t1b_full[:noccb, noccb:] = t1b
        from jax import scipy as jsp
        wave_data['exp_t1a'] = jsp.linalg.expm(t1a_full)
        wave_data['exp_mt1a'] = jsp.linalg.expm(-t1a_full)
        wave_data['exp_t1b'] = jsp.linalg.expm(t1b_full)
        wave_data['exp_mt1b'] = jsp.linalg.expm(-t1b_full)
        lt1a = jnp.einsum('ia,gja->gij', t1a, chol_a[:, :nocca, nocca:])
        lt1b = jnp.einsum('ia,gja->gij', t1b, chol_b[:, :noccb, noccb:])
        # e0t1orb = <exp(T1)HF|H|HF>_i
        e0t1orb_aa = (jnp.einsum('gik,ik,gjj->',lt1a, prja, lt1a) 
                    - jnp.einsum('gij,gjk,ik->',lt1a, lt1a, prja)) * 0.5
        e0t1orb_ab = jnp.einsum('gik,ik,gjj->',lt1a, prja, lt1b) * 0.5
        e0t1orb_ba = jnp.einsum('gik,ik,gjj->',lt1b, prjb, lt1a) * 0.5
        e0t1orb_bb = (jnp.einsum('gik,ik,gjj->',lt1b, prjb, lt1b)
                    - jnp.einsum('gij,gjk,ik->',lt1b, lt1b, prjb)) * 0.5
        ham_data['e0t1orb'] = e0t1orb_aa + e0t1orb_ab + e0t1orb_ba + e0t1orb_bb
        if "ad" in options["trial"]:
            trial = wavefunctions.uccsd_pt2_ad(norb, nelec, n_batch = options["n_batch"])
            wave_data["t2aa"] = jnp.einsum('iajb,ik->kajb',t2aa,prja)
            wave_data["t2ab"] = jnp.einsum('iajb,ik->kajb',t2ab,prja)
            wave_data["t2ba"] = jnp.einsum('jbia,ik->kajb',t2ab,prjb)
            wave_data["t2bb"] = jnp.einsum('iajb,ik->kajb',t2bb,prjb)
        elif "alpha" in options["trial"]:
            trial = wavefunctions.uccsd_pt2_alpha(norb, nelec, n_batch = options["n_batch"])
            wave_data["t2aa"] = jnp.einsum('iajb,ik->kajb',t2aa,prja)
            wave_data["t2ab"] = jnp.einsum('iajb,ik->kajb',t2ab,prja)
        elif "beta" in options["trial"]:
            trial = wavefunctions.uccsd_pt2_beta(norb, nelec, n_batch = options["n_batch"])
            wave_data["t2ba"] = jnp.einsum('jbia,ik->kajb',t2ab,prjb)
            wave_data["t2bb"] = jnp.einsum('iajb,ik->kajb',t2bb,prjb)
        else:
            trial = wavefunctions.uccsd_pt2(norb, nelec, n_batch = options["n_batch"])
            wave_data["t2aa"] = jnp.einsum('iajb,ik->kajb',t2aa,prja)
            wave_data["t2ab"] = jnp.einsum('iajb,ik->kajb',t2ab,prja)
            wave_data["t2ba"] = jnp.einsum('jbia,ik->kajb',t2ab,prjb)
            wave_data["t2bb"] = jnp.einsum('iajb,ik->kajb',t2bb,prjb)

    if options["walker_type"] == "rhf":
        prop = propagation.propagator_restricted(
            options["dt"], 
            options["n_walkers"], 
            options["n_exp_terms"],
            options["n_batch"]
        )

    elif options["walker_type"] == "uhf":
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
        print(f"# nelec: {nelec}")
        print("#")
        for op in options:
            if options[op] is not None:
                print(f"# {op}: {options[op]}")
        print("#")

    return ham_data, prop, trial, wave_data, sampler, options, MPI

import os
def run_lnoafqmc(options,nproc=None,
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
    # if  'cc' in options['trial'] and 'pt' in options['trial']:
    if 'pt2' in options['trial']:
        script='ccsd_pt2/run_uafqmc.py'
    elif 'pt' in options['trial']:
        script='ccsd_pt/run_uafqmc.py'
    else:
        raise NotImplementedError("Only support CCSD_pt and CCSD_pt2 trial.")
    
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    script = f"{dir_path}/{script}"
    print(f'# AFQMC script: {script}')
    
    os.system(
        f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1;"
        f"{mpi_prefix} python {script} {gpu_flag} |tee afqmc.out"
    )

def run_afqmc(mf, options, lo_coeff, frag_lolist,
              nfrozen = 0, thresh = 1e-6, chol_cut = 1e-5,
              lno_type = ['1h']*2, run_frg_list = None, nproc = None):
    
    mlno = ulnoccsd.ULNOCCSD_T(mf, lo_coeff, frag_lolist, frozen=nfrozen).set(verbose=0)
    mlno.lno_thresh = [thresh*10,thresh]
    lno_thresh = mlno.lno_thresh
    lno_type = ['1h','1h'] if lno_type is None else lno_type
    lno_thresh = [1e-5, 1e-6] if lno_thresh is None else lno_thresh
    lno_pct_occ = None
    lno_norb = None
    # lo_proj_thresh = 1e-10
    # lo_proj_thresh_active = 0.1
    eris = None

    if run_frg_list is None:
        nfrag = len(frag_lolist)
        run_frg_list = range(nfrag)
    
    frag_lolist = [frag_lolist[i] for i in run_frg_list]
    nfrag = len(frag_lolist)
    print(f'# Number of LNO-FRAGMENT: {nfrag}')
    if lno_pct_occ is None:
        lno_pct_occ = [None, None]
    if lno_norb is None:
        lno_norb = [[None,None]] * nfrag
    mf = mlno._scf

    if eris is None: eris = mlno.ao2mo()

    seeds = random.randint(random.PRNGKey(options["seed"]),
                        shape=(nfrag,), minval=0, maxval=100*nfrag)
    options["max_error"] = options["max_error"]/np.sqrt(nfrag)

    nelec_list = [None] * nfrag
    norb_list = [None] * nfrag
    eorb_mp2_cc = [None] * nfrag
    # Loop over fragment
    for ifrag, loidx in enumerate(frag_lolist):
        print(f'\n ########### RUNNING LNO-FRAGMENT {run_frg_list[ifrag]+1}/{nfrag} ###########')
        if len(loidx) == 2 and isinstance(loidx[0], Iterable): # Unrestricted
            orbloc = [lo_coeff[0][:,loidx[0]], lo_coeff[1][:,loidx[1]]]
            lno_param = [
                [
                    {
                        'thresh': (
                            lno_thresh[i][s] if isinstance(lno_thresh[i], Iterable)
                            else lno_thresh[i]
                        ),
                        'pct_occ': (
                            lno_pct_occ[i][s] if isinstance(lno_pct_occ[i], Iterable)
                            else lno_pct_occ[i]
                        ),
                        'norb': (
                            lno_norb[ifrag][i][s] if isinstance(lno_norb[ifrag][i], Iterable)
                            else lno_norb[ifrag][i]
                        ),
                    } for i in [0, 1]
                ] for s in range(2)
            ]

        else:
            orbloc = lo_coeff[:,loidx]
            lno_param = [{'thresh': lno_thresh[i], 'pct_occ': lno_pct_occ[i],
                            'norb': lno_norb[ifrag][i]} for i in [0,1]]

        lno_coeff, frozen, uocc_loc, _ = mlno.make_las(eris, orbloc, lno_type, lno_param)

        if uocc_loc[0].size > 0 and uocc_loc[1].size == 0:
            lno_elec_type = 'alpha'
        elif uocc_loc[0].size == 0 and uocc_loc[1].size > 0:
            lno_elec_type = 'beta'
        else: lno_elec_type = 'How could it be???'
        print(f'# LNO Electron Type: {lno_elec_type}')

        mo_occ = mlno.mo_occ
        frozen, maskact = ulnoccsd.get_maskact(frozen, [mo_occ[0].size, mo_occ[1].size])
        mcc = ulnoccsd.UCCSD(mf, mo_coeff=lno_coeff, frozen=frozen).set(verbose=4)
        mcc._s1e = mlno._s1e
        mcc._h1e = mlno._h1e
        mcc._vhf = mlno._vhf
        if mlno.kwargs_imp is not None:
            mcc = mcc.set(**mlno.kwargs_imp)
        eorb_mp2_cc[ifrag], t1, t2 =\
            ulno_ccsd(mcc, lno_coeff, uocc_loc, mo_occ, maskact, ccsd_t=True)
        
        prja = uocc_loc[0] @ uocc_loc[0].T.conj()
        prjb = uocc_loc[1] @ uocc_loc[1].T.conj()
        prjlo = [prja, prjb]

        print(f'# LNO-MP2 Orbital Energy: {eorb_mp2_cc[ifrag][0]:.8f}')
        print(f'# LNO-CCSD Orbital Energy: {eorb_mp2_cc[ifrag][1]:.8f}')
        print(f'# LNO-CCSD(T) Orbital Energy: {eorb_mp2_cc[ifrag][2]:.8f}')
        
        from mpi4py import MPI
        if not MPI.Is_finalized():
            MPI.Finalize()

        if 'ad' not in options["trial"]:
            if lno_elec_type == 'alpha':
                options["trial"] += '_alpha'
            elif lno_elec_type == 'beta':
                options["trial"] += '_beta'

        options["seed"] = seeds[ifrag]
        nelec_list[ifrag], norb_list[ifrag] \
            = prep_afqmc(mf,lno_coeff,t1,t2,frozen,prjlo,options,chol_cut=chol_cut)
        run_lnoafqmc(options,nproc)
        os.system(f'mv afqmc.out lnoafqmc.out{run_frg_list[ifrag]+1}')

    # finish lno loop
    mmp = mp.MP2(mf, frozen=nfrozen)
    emp2_tot = mmp.kernel()[0]

    eorb_pt2 = np.empty(nfrag,dtype='float64')
    eorb_pt2_err = np.empty(nfrag,dtype='float64')
    run_time = np.empty(nfrag,dtype='float64')
    for n, i in enumerate(run_frg_list):
        with open(f"lnoafqmc.out{i+1}", "r") as rf:
            for line in rf:
                if "Ept (direct observation)" in line:
                    eorb_pt2[n] = float(line.split()[-3])
                    eorb_pt2_err[n] = float(line.split()[-1])
                if "total run time" in line:
                    run_time[n] = float(line.split()[-1])

    nelec_list = np.array(nelec_list)
    norb_list = np.array(norb_list)
    eorb_mp2_cc = np.array(eorb_mp2_cc)
    nelec = (np.mean(nelec_list[:,0]),np.mean(nelec_list[:,1]))
    norb = (np.mean(norb_list[:,0]),np.mean(norb_list[:,1]))
    e_mp2 = sum(eorb_mp2_cc[:,0])
    e_ccsd = sum(eorb_mp2_cc[:,1])
    e_ccsd_pt = e_ccsd + sum(eorb_mp2_cc[:,2])
    e_afqmc_pt2 = sum(eorb_pt2)
    e_afqmc_pt2_err = np.sqrt(sum(eorb_pt2_err**2))
    tot_time = sum(run_time)

    with open(f'lno_result.out', 'w') as out_file:
        print('# frag  eorb_mp2  eorb_ccsd  eorb_ccsd(t) ' \
              '  eorb_afqmc/ccsd_pt2  nelec  norb  time',
                file=out_file)
        for n, i in enumerate(run_frg_list):
            print(f'{i+1:3d}  '
                    f'{eorb_mp2_cc[n,0]:.8f}  {eorb_mp2_cc[n,1]:.8f}  {eorb_mp2_cc[n,2]:.8f}  '
                    f'{eorb_pt2[n]:.6f} +/- {eorb_pt2_err[n]:.6f}  '
                    f'{nelec_list[n]}  {norb_list[n]}  {run_time[n]:.2f}', file=out_file)
        print(f'# LNO Thresh: {lno_thresh}',file=out_file)
        print(f'# LNO Average Number of Electrons: ({nelec[0]:.1f},{nelec[1]:.1f})',file=out_file)
        print(f'# LNO Average Number of Basis: ({norb[0]:.1f},{norb[1]:.1f})',file=out_file)
        print(f'# LNO-MP2 Energy: {e_mp2:.8f}',file=out_file)
        print(f'# LNO-CCSD Energy: {e_ccsd:.8f}',file=out_file)
        print(f'# LNO-CCSD(T) Energy: {e_ccsd_pt:.8f}',file=out_file)
        print(f'# LNO-AFQMC/CCSD_PT Energy: {e_afqmc_pt2:.6f} +/- {e_afqmc_pt2_err:.6f}',file=out_file)
        print(f'# MP2 Correction: {emp2_tot-e_mp2:.8f}',file=out_file)
        print(f"# total run time: {tot_time:.2f}",file=out_file)

    return None