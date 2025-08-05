import pickle
import os
import numpy as np
from ad_afqmc import pyscf_interface
from ad_afqmc.lno.afqmc import LNOAFQMC
from pyscf import mcscf
from functools import reduce
_fdot = np.dot
fdot = lambda *args: reduce(_fdot, args)


def prep_lnocs_files(options,mf,mo_coeff,lo_coeff,frozen=None,
                     option_file='options.bin',
                     mo_file='mo_coeff.npz',
                     chol_file='FCIDUMP_chol'):
    
    print("# Preparing CS-LNO-AFQMC calculation")
    mol = mf.mol
    maskocc = mf.mo_occ>1e-10
    nmo = mf.mo_occ.size
    # Convert frozen to 0 bc PySCF solvers do not support frozen=None or empty list
    if frozen is None:
        frozen = 0
    elif isinstance(frozen, (list,tuple,np.ndarray)) and len(frozen) == 0:
        frozen = 0
    if isinstance(frozen, (int,np.integer)):
        maskact = np.hstack([np.zeros(frozen,dtype=bool),
                                np.ones(nmo-frozen,dtype=bool)])
    elif isinstance(frozen, (list,tuple,np.ndarray)):
        maskact = np.array([i not in frozen for i in range(nmo)])
        
    orbfrzocc = mo_coeff[:,~maskact& maskocc]
    orbactocc = mo_coeff[:, maskact& maskocc]
    orbactvir = mo_coeff[:, maskact&~maskocc]
    orbfrzvir = mo_coeff[:,~maskact&~maskocc]
    _, nactocc, nactvir, _ = \
        [orb.shape[1] for orb in [orbfrzocc,orbactocc,orbactvir,orbfrzvir]]
    norb_act = (nactocc+nactvir)
    nelec_act = nactocc*2
    norb_frozen = frozen

    # s1e = mf.get_ovlp()
    # prjlo = fdot(lo_coeff.T, s1e, orbactocc)
    # options["prjlo"] = prjlo
    import pickle
    with open(option_file, 'wb') as f:
        pickle.dump(options, f)

    # calculate cholesky integrals
    print('# Generating Cholesky Integrals')
    nao = mol.nao
    act_idx = [i for i in range(nao) if i not in norb_frozen]
    _, chol, _, _ = pyscf_interface.generate_integrals(
            mol,mf.get_hcore(),mo_coeff[:,act_idx],DFbas=mf.with_df.auxmol.basis)

    mc = mcscf.CASSCF(mf, norb_act, nelec_act) 
    mc.frozen = norb_frozen
    nelec = mc.nelecas
    mc.mo_coeff = mo_coeff
    h1e, enuc = mc.get_h1eff()

    print(f'# local active orbitals are {act_idx}') #yichi
    print(f'# local active space size {len(act_idx)}') #yichi
    print(f'# chol shape: {chol.shape}') #yichi

    nbasis = h1e.shape[-1]
    print("# Finished calculating Cholesky integrals\n")
    print('# Size of the correlation space:')
    print(f'# Number of electrons: {nelec}')
    print(f'# Number of basis functions: {nbasis}')
    print(f'# Number of Cholesky vectors: {chol.shape[0]}\n')
    chol = chol.reshape((-1, nbasis, nbasis))
    v0 = 0.5 * np.einsum('nik,njk->ij', chol, chol, optimize='optimal')
    h1e_mod = h1e - v0
    chol = chol.reshape((chol.shape[0], -1))

    # write mo coefficients
    trial_coeffs = np.empty((2, nbasis, nbasis))
    q = np.eye(mol.nao- len(norb_frozen))
    trial_coeffs[0] = q
    trial_coeffs[1] = q
    s1e = mf.get_ovlp()
    prjlo = fdot(lo_coeff.T, s1e, orbactocc)
    # np.savez(mo_file,mo_coeff=trial_coeffs)
    np.savez(mo_file,mo_coeff=trial_coeffs,prjlo=prjlo)
    pyscf_interface.write_dqmc(h1e,h1e_mod,chol,sum(nelec),nbasis,enuc,ms=mol.spin,
                               filename=chol_file,mo_coeffs=trial_coeffs)
    
    return None


def run_lnocs_afqmc(nproc=None,use_gpu=False):
    
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    script = f"{dir_path}/run_lnocs.py"
    
    if use_gpu:
        mpi_prefix = ""
        gpu_flag = "--use_gpu"
    else:
        mpi_prefix = "mpirun "
        gpu_flag = ""

    if nproc is not None:
        mpi_prefix += f"-np {nproc} "
    os.system(
        f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; {mpi_prefix} python {script} {gpu_flag} |tee lnocs_afqmc.out"
    )


def run_cs_frags(mf1,mf2,frozen,options,nproc=None,
                 chol_cut=1e-6,lno_thresh=1e-5):
    
    no_type='ie'
    lo_type="pm"

    mfcc1 = LNOAFQMC(mf1,thresh=lno_thresh,frozen=frozen)
    mfcc1.thresh_occ = lno_thresh*10
    mfcc1.thresh_vir = lno_thresh
    mfcc1.nwalk_per_proc = options["n_walkers"]
    mfcc1.chol_cut = chol_cut

    mfcc2 = LNOAFQMC(mf2,thresh=lno_thresh,frozen=frozen)
    mfcc2.thresh_occ = lno_thresh*10
    mfcc2.thresh_vir = lno_thresh
    mfcc2.nwalk_per_proc = options["n_walkers"]
    mfcc2.chol_cut = chol_cut

    eris1 = mfcc1.ao2mo()
    orbloc1 = mfcc1.get_lo(lo_type=lo_type)
    frag_lolist1 = [[i] for i in range(orbloc1.shape[1])]
    frag_nonvlist1 = mfcc1.frag_nonvlist
    nfrag1 = len(frag_lolist1)
    if frag_nonvlist1 is None: frag_nonvlist1 = [[None,None]] * nfrag1

    eris2 = mfcc2.ao2mo()
    orbloc2 = mfcc2.get_lo(lo_type=lo_type)
    frag_lolist2 = [[i] for i in range(orbloc2.shape[1])]
    frag_nonvlist2 = mfcc2.frag_nonvlist
    nfrag2 = len(frag_lolist2)
    if frag_nonvlist2 is None: frag_nonvlist2 = [[None,None]] * nfrag2

    if nfrag1 != nfrag2: 
        raise ValueError("number of fragments are different in two system!")

    from ad_afqmc.lno.base import lno
    for ifrag in range(0,nfrag1):
        frag_target_nocc1, frag_target_nvir1 = frag_nonvlist1[ifrag]
        fraglo1 = frag_lolist1[ifrag]
        # frag_res1 = [None] * nfrag1
        orbfragloc1 = orbloc1[:,fraglo1]
        frag_target_nocc2, frag_target_nvir2 = frag_nonvlist2[ifrag]
        fraglo2 = frag_lolist2[ifrag]
        # frag_res2 = [None] * nfrag2
        orbfragloc2 = orbloc2[:,fraglo2]

        # make fpno
        THRESH_INTERNAL = 1e-10
        thresh_pno1 = [mfcc1.thresh_occ, mfcc1.thresh_vir]
        frozen_mask1 = mfcc1.get_frozen_mask()
        thresh_pno2 = [mfcc2.thresh_occ, mfcc2.thresh_vir]
        frozen_mask2 = mfcc2.get_frozen_mask()
        
        frzfrag1, orbfrag1, _ = lno.make_fpno1(mfcc1,eris1,orbfragloc1,no_type,
                                               THRESH_INTERNAL,thresh_pno1,
                                               frozen_mask=frozen_mask1,
                                               frag_target_nocc=frag_target_nocc1,
                                               frag_target_nvir=frag_target_nvir1,
                                               canonicalize=False)
        
        frzfrag2, orbfrag2, _ = lno.make_fpno1(mfcc2,eris2,orbfragloc2,no_type,
                                                THRESH_INTERNAL, thresh_pno2,
                                                frozen_mask=frozen_mask2,
                                                frag_target_nocc=frag_target_nocc2,
                                                frag_target_nvir=frag_target_nvir2,
                                                canonicalize=False)
        
        #take the larger active space
        if len(frzfrag1) > len(frzfrag2):
            frzfrag = frzfrag2
        else:
            frzfrag = frzfrag1

        prep_lnocs_files(options,mf1,orbfrag1,orbfragloc1,frozen=frzfrag,eris=eris1,
                  option_file='option.bin',mo_file='mo1.npz',chol_file='chol1')
        prep_lnocs_files(options,mf2,orbfrag2,orbfragloc2,frozen=frzfrag,eris=eris2,
                  option_file='option.bin',mo_file='mo2.npz',chol_file='chol2')
        
        use_gpu = options["use_gpu"]
        run_lnocs_afqmc(nproc,use_gpu)
        os.system(f"mv lnocs_afqmc.out cs_frag_{ifrag}.out")
        
    return None #can_orbfrag1,can_orbfrag2 for mp2