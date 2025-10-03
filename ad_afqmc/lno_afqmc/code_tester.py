from jax import numpy as jnp
from jax import vmap, jit, lax, random
import jax
from ad_afqmc.lno.base import lno
from ad_afqmc.lno_afqmc import lno_maker, afqmc_maker, lnoafqmc_runner, data_maker
import numpy as np
from pyscf import lib, ao2mo
from functools import partial
from ad_afqmc import wavefunctions, propagation, sampling
from typing import Tuple
import os

def get_lo(lnocc,lo_type='boys'):
    '''get the localized orbitals in the active space'''
    from pyscf import lo
    mol = lnocc._scf.mol
    _,actocc,actvir,_ = lnocc.split_mo()
    if lo_type == 'boys':
        lococc = lo.Boys(mol,actocc).kernel()
        locvir = lo.Boys(mol,actvir).kernel()
    if lo_type == 'pm':
        lococc = lo.PM(mol,actocc).kernel()
        locvir = lo.PM(mol,actvir).kernel()
    print('(loc_occ,loc_vir) span the same space as (occ,vir): ',
          lno_maker.check_lo_span(lnocc,lococc,locvir))
    return lococc, locvir

def make_lno(mfcc,orbfragloc,lococc,locvir,thresh_internal,thresh_external):

    if isinstance(thresh_external,(float,int)):
        thresh_ext_occ = thresh_ext_vir = thresh_external
    else:
        thresh_ext_occ, thresh_ext_vir  = thresh_external

    mf = mfcc._scf
    if getattr(mf,'with_df',None) is not None:
        print('Using DF integrals')
        eris = mfcc.ao2mo()
        s1e = eris.s1e if eris.s1e is None else mf.get_ovlp()
        fock = eris.fock  if eris.fock is None else mf.get_fock()
    else:
        print('Using true 4-index integrals')
        eris = None
        s1e = mf.get_ovlp()
        fock = mf.get_fock()
        
    nocc = np.count_nonzero(mf.mo_occ>1e-10)
    nmo = mf.mo_occ.size
    # orbocc0, orbocc1, orbvir1, orbvir0 = mfcc.split_mo()
    frzocc, _, _, frzvir = mfcc.split_mo() # frz_occ, act_occ, act_vir, frz_vir
    _, moeocc1, moevir1, _ = mfcc.split_moe() # split energy

    lovir = abs(orbfragloc.T @ s1e @ locvir).max() > 1e-10
    m = orbfragloc.T @ s1e @ lococc # overlap with all loc act_occs
    uocc1, uocc2 = lno.projection_construction(m, thresh_internal)
    moefragocc1, orbfragocc1 = lno.subspace_eigh(fock, lococc@uocc1)
    if lovir:
        m = orbfragloc.T @ s1e @ locvir
        uvir1, uvir2 = lno.projection_construction(m, thresh_internal)
        _, orbfragvir1 = lno.subspace_eigh(fock, locvir@uvir1)

    def moe_Ov(moefragocc):
        return (moefragocc[:,None] - moevir1).reshape(-1)

    eov = moe_Ov(moeocc1)
    # Construct PT2 dm_vv
    u = lococc.T @ s1e @ orbfragocc1
    if getattr(mf,'with_df',None) is not None:
        print('Using DF integrals')
        ovov = eris.get_Ovov(u)
    else:
        print('Using true 4-index integrals')
        eri_ao = mf._eri
        nao = mf.mol.nao
        eri_ao = ao2mo.restore(1,eri_ao,nao)
        eri_mo = lno_maker.get_eri_mo(eri_ao,lococc,locvir)
        ovov = lib.einsum('iI,iajb->Iajb',u,eri_mo)
    eia = moe_Ov(moefragocc1)
    ejb = eov
    e1_or_e2 = 'e1'
    swapidx = 'ab'

    eiajb = (eia[:,None]+ejb).reshape(*ovov.shape)
    t2 = ovov / eiajb

    dmvv = lno.make_rdm1_mp2(t2, 'vv', e1_or_e2, swapidx)
   
    if lovir:
        dmvv = uvir2.T @ dmvv @uvir2

    # Construct PT2 dm_oo
    e1_or_e2 = 'e2'
    swapidx = 'ab'

    dmoo = lno.make_rdm1_mp2(t2, 'oo', e1_or_e2, swapidx)
    dmoo = uocc2.T @ dmoo @ uocc2

    t2 = ovov = eiajb = None
    orbfragocc2, orbfragocc0 \
        = lno.natorb_compression(dmoo, lococc, thresh_ext_occ, uocc2)

    can_orbfragocc12 = lno.subspace_eigh(fock, np.hstack([orbfragocc2, orbfragocc1]))[1]
    orbfragocc12 = np.hstack([orbfragocc2, orbfragocc1])
    if lovir:
        orbfragvir2, orbfragvir0 \
            = lno.natorb_compression(dmvv,locvir,thresh_ext_vir,uvir2)

        can_orbfragvir12 = lno.subspace_eigh(fock, np.hstack([orbfragvir2, orbfragvir1]))[1]
        orbfragvir12 = np.hstack([orbfragvir2, orbfragvir1])
    else: 
        orbfragvir2, orbfragvir0 = lno.natorb_compression(dmvv,locvir,thresh_ext_vir)

        can_orbfragvir12 = lno.subspace_eigh(fock, orbfragvir2)[1]
        orbfragvir12 = orbfragvir2

    lno_orb = np.hstack([frzocc, orbfragocc0, orbfragocc12,
                         orbfragvir12, orbfragvir0, frzvir])
    can_orbfrag = np.hstack([frzocc, orbfragocc0, can_orbfragocc12,
                        can_orbfragvir12, orbfragvir0, frzvir])
    
    frzfrag = np.hstack([np.arange(frzocc.shape[1]+orbfragocc0.shape[1]),
                         np.arange(nocc+orbfragvir12.shape[1],nmo)])

    return frzfrag, lno_orb, can_orbfrag

@jax.jit
def walker_rhf_fock_energy(walker,ham_data,wave_data):
    '''one-body part of rhf walker's energy with 
    non-orthonal trial C_ia <psi|H|psi_ia>
    should be (close to) zero for orthogonal trial!!!'''

    rot_h1,rot_chol = ham_data['rot_h1'],ham_data['rot_chol']
    trial_mo = wave_data['mo_coeff']
    nocc = rot_h1.shape[0]
    gw = (walker.dot(jnp.linalg.inv(trial_mo.T.conj() @ walker))).T
    f = jnp.einsum('gja,ai->gji',rot_chol[:,:nocc,nocc:],
                    gw.T[nocc:,:nocc],optimize='optimal')
    tr_f = vmap(jnp.trace)(f)
    tr_chol = vmap(jnp.trace)(rot_chol[:,:nocc,:nocc])
    e_fock1 = 2*jnp.einsum('ia,ia->',gw[:nocc,nocc:],rot_h1[:nocc,nocc:])
    e_fock2 = 4*jnp.einsum('g,g->',tr_f,tr_chol)\
              - 2*jnp.einsum('gji,gij->',f,rot_chol[:,:nocc,:nocc])
    # ene12 = 2*jnp.sum(tf*tc) \
    #         - jnp.einsum('gik,gki->',f,rot_chol[:,:nocc,:nocc])
    return e_fock1+e_fock2,e_fock1,e_fock2

@jax.jit
def walker_rhf_coexc(walker,ham_data,wave_data):
    '''two-body part of rhf walker's energy with 
    non-orthonal trial C_ijab <psi|H|psi_{ia}^{jb}>'''

    rot_h1,rot_chol = ham_data['rot_h1'],ham_data['rot_chol']
    trial_mo = wave_data['mo_coeff']
    nocc = rot_h1.shape[0]
    gw = (walker.dot(jnp.linalg.inv(trial_mo.T.conj() @ walker))).T
    f = jnp.einsum('gja,ai->gji',rot_chol[:,:nocc,nocc:],
                    gw.T[nocc:,:nocc],optimize='optimal')
    tr_f = vmap(jnp.trace)(f)
    e_col = 2*jnp.einsum('g,g->',tr_f,tr_f)
    e_exc = -jnp.einsum('gij,gji->',f,f)
    # e_col = 2*jnp.einsum('ia,jb,gia,gjb->',
    #                      gf[:nocc,nocc:],gf[:nocc,nocc:],
    #                      rot_chol[:,:nocc,nocc:],rot_chol[:,:nocc,nocc:])
    # e_exc = - jnp.einsum('ia,jb,gib,gja->',
    #                      gf[:nocc,nocc:],gf[:nocc,nocc:],
    #                      rot_chol[:,:nocc,nocc:],rot_chol[:,:nocc,nocc:])
    return e_col+e_exc,e_col,e_exc

@jax.jit
def walker_norhf_energy(walker, ham_data, wave_data):
    rot_h1,rot_chol = ham_data['rot_h1'],ham_data['rot_chol']
    trial_mo = wave_data['mo_coeff']
    nocc = rot_h1.shape[0]
    gw = (walker.dot(jnp.linalg.inv(trial_mo.T.conj() @ walker))).T
    f = jnp.einsum('gja,ai->gji',rot_chol[:,:nocc,nocc:],
                    gw.T[nocc:,:nocc],optimize='optimal')
    tr_f = vmap(jnp.trace)(f)
    # one-bady part
    tr_chol = vmap(jnp.trace)(rot_chol[:,:nocc,:nocc])
    e_fock1 = 2*jnp.einsum('ia,ia->',gw[:nocc,nocc:],rot_h1[:nocc,nocc:])
    e_fock2 = 4*jnp.einsum('g,g->',tr_f,tr_chol)\
              - 2*jnp.einsum('gji,gij->',f,rot_chol[:,:nocc,:nocc])
    ene1 = e_fock1+e_fock2
    # two-body part
    e_col = 2*jnp.einsum('g,g->',tr_f,tr_f)
    e_exc = -jnp.einsum('gij,gji->',f,f)
    ene2 = e_col+e_exc
    return ene1+ene2

@jax.jit
def walker_rhf_fock_eorb(walker,ham_data,wave_data):
    '''one-body part of rhf walker's energy with 
    non-orthonal trial C_ia <psi|H|psi_ia>
    should be (close to) zero for orthogonal trial!!!'''
    m = jnp.dot(wave_data["prjlo"].T,wave_data["prjlo"])
    trial_mo = wave_data['mo_coeff']
    rot_h1,rot_chol = ham_data['rot_h1'],ham_data['rot_chol']
    nocc = rot_h1.shape[0]
    gw = (walker.dot(jnp.linalg.inv(trial_mo.T.conj() @ walker))).T
    f = jnp.einsum('gja,ai->gji',rot_chol[:,:nocc,nocc:],
                    gw.T[nocc:,:nocc],optimize='optimal')
    # tr_f = vmap(jnp.trace)(f)
    tr_chol = vmap(jnp.trace)(rot_chol[:,:nocc,:nocc])
    e_fock1 = 2*jnp.einsum('ia,ka,ik->',gw[:nocc,nocc:],rot_h1[:nocc,nocc:],m)
    e_fock2 = 4*jnp.einsum('gik,ik,g->',f,m,tr_chol)\
              - 2*jnp.einsum('gij,gjk,ik->',f,rot_chol[:,:nocc,:nocc],m)
    return e_fock1+e_fock2

@jax.jit
def walker_rhf_ceorb(walker,ham_data,wave_data):
    '''two-body part of rhf walker's energy with 
    non-orthonal trial C_ijab <psi|H|psi_{ia}^{jb}>'''
    m = jnp.dot(wave_data["prjlo"].T,wave_data["prjlo"])
    trial_mo = wave_data['mo_coeff']
    rot_h1,rot_chol = ham_data['rot_h1'],ham_data['rot_chol']
    nocc = rot_h1.shape[0]
    gw = (walker.dot(jnp.linalg.inv(trial_mo.T.conj() @ walker))).T
    f = jnp.einsum('gja,ai->gji',rot_chol[:,:nocc,nocc:],
                    gw.T[nocc:,:nocc],optimize='optimal')
    tr_f = vmap(jnp.trace)(f)
    e_col = 2*jnp.einsum('gik,ik,g->',f,m,tr_f)
    e_exc = -jnp.einsum('gij,gjk,ik->',f,f,m)
    return e_col+e_exc

@jax.jit
def _frg_norhf_eorb(walker,ham_data,wave_data):
    m = jnp.dot(wave_data["prjlo"].T,wave_data["prjlo"])
    trial_mo = wave_data['mo_coeff']
    rot_h1,rot_chol = ham_data['rot_h1'],ham_data['rot_chol']
    nocc = rot_h1.shape[0]
    gw = (walker.dot(jnp.linalg.inv(trial_mo.T.conj() @ walker))).T
    f = jnp.einsum('gja,ai->gji',rot_chol[:,:nocc,nocc:],
                    gw.T[nocc:,:nocc],optimize='optimal')
    # tr_f = vmap(jnp.trace)(f)
    tr_chol = vmap(jnp.trace)(rot_chol[:,:nocc,:nocc])
    e_fock1 = 2*jnp.einsum('ia,ka,ik->',gw[:nocc,nocc:],rot_h1[:nocc,nocc:],m)
    e_fock2 = 4*jnp.einsum('gik,ik,g->',f,m,tr_chol)\
              - 2*jnp.einsum('gij,gjk,ik->',f,rot_chol[:,:nocc,:nocc],m)
    ene1 = e_fock1+e_fock2
    tr_f = vmap(jnp.trace)(f)
    e_col = 2*jnp.einsum('gik,ik,g->',f,m,tr_f)
    e_exc = -jnp.einsum('gij,gjk,ik->',f,f,m)
    ene2 = e_col+e_exc
    return jnp.real(ene1+ene2)

@partial(jit, static_argnums=(3,))
def _frg_rhf_eorb(rot_h1, rot_chol, walker, trial, wave_data):
    '''hf orbital correlation energy'''
    m = jnp.dot(wave_data["prjlo"].T,wave_data["prjlo"])
    nocc = rot_h1.shape[0]
    _calc_green = wavefunctions.rhf(
        trial.norb,trial.nelec,n_batch=trial.n_batch)._calc_green
    green_walker = _calc_green(walker, wave_data)
    f = jnp.einsum('gij,jk->gik', rot_chol[:,:nocc,nocc:],
                    green_walker.T[nocc:,:nocc], optimize='optimal')
    c = vmap(jnp.trace)(f)
    eneo2Jt = jnp.einsum('Gxk,xk,G->',f,m,c)*2 
    eneo2ext = jnp.einsum('Gxy,Gyk,xk->',f,f,m)
    hf_orb_en = eneo2Jt - eneo2ext
    # olp_ratio = _calc_olp_ratio_restricted(walker,wave_data)
    # hf_orb_cr = jnp.real(olp_ratio*hf_orb_en)
    return jnp.real(hf_orb_en)

@partial(jit, static_argnums=(3,))
def frg_rhf_eorb(walkers,ham_data,wave_data,trial):
    hf_orb_en = vmap(_frg_rhf_eorb, in_axes=(None, None, 0, None, None))(
        ham_data['rot_h1'], ham_data['rot_chol'], walkers, trial, wave_data)
    return hf_orb_en

@jax.jit
def frg_norhf_eorb(walkers,ham_data,wave_data):
    hf_orb_en = vmap(_frg_norhf_eorb, in_axes=(0, None, None))(
        walkers, ham_data, wave_data)
    return hf_orb_en

@partial(jit, static_argnums=(2,3,5,6))
def block_orb(prop_data: dict,
              ham_data: dict,
              prop: propagation.propagator,
              trial: wavefunctions,
              wave_data: dict,
              sampler: sampling.sampler,
              which_rhf):
        """Block scan function. Propagation and orbital_i energy calculation."""
        prop_data["key"], subkey = random.split(prop_data["key"])
        fields = random.normal(
            subkey,
            shape=(
                sampler.n_prop_steps,
                prop.n_walkers,
                ham_data["chol"].shape[0],
            ),
        )
        # propgate n_prop_steps x dt
        _step_scan_wrapper = lambda x, y: sampler._step_scan(
            x, y, ham_data, prop, trial, wave_data
        )
        prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)
        prop_data["n_killed_walkers"] += prop_data["weights"].size - jnp.count_nonzero(
            prop_data["weights"]
        )
        prop_data = prop.orthonormalize_walkers(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        
        if isinstance(trial, wavefunctions.rhf):
            if which_rhf == 1:
                orb_en \
                    = frg_rhf_eorb(
                    prop_data["walkers"],ham_data,wave_data,trial)
            elif which_rhf == 2:
                orb_en \
                    = frg_norhf_eorb(
                    prop_data["walkers"],ham_data,wave_data)
        elif isinstance(trial, wavefunctions.cisd):
            hf_orb_cr,_ \
                = afqmc_maker.frg_hf_cr(
                    prop_data["walkers"],ham_data,wave_data,trial)
            ci_orb_cr \
                = afqmc_maker.frg_ci_cr(
                    prop_data["walkers"], ham_data, wave_data, trial)
            orb_en = hf_orb_cr + ci_orb_cr

        energy_samples = jnp.real(
            trial.calc_energy(prop_data["walkers"], ham_data, wave_data)
        )
        energy_samples = jnp.where(
            jnp.abs(energy_samples - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt),
            prop_data["e_estimate"],
            energy_samples,
        )

        blk_wt = jnp.sum(prop_data["weights"])
        blk_en = jnp.sum(energy_samples * prop_data["weights"]) / blk_wt
        # blk_hf_orb_en = jnp.sum(hf_orb_en * prop_data["weights"]) / blk_wt
        blk_eorb= jnp.sum(orb_en * prop_data["weights"]) / blk_wt
        # blk_cc_orb_cr = jnp.sum(cc_orb_cr * prop_data["weights"]) / blk_wt
        # blk_cc_orb_en = blk_hf_orb_cr+blk_cc_orb_cr
        
        prop_data["pop_control_ene_shift"] = (
            0.9 * prop_data["pop_control_ene_shift"] + 0.1 * blk_en
        )

        return prop_data,(blk_wt,blk_en,blk_eorb)

@partial(jit, static_argnums=(2,3,5,6))
def _sr_block_scan_orb(
    prop_data: dict,
    ham_data: dict,
    prop: propagation.propagator,
    trial: wavefunctions,
    wave_data: dict,
    sampler: sampling.sampler,
    which_rhf,
) -> Tuple[dict, Tuple[jax.Array, jax.Array]]:
    def _block_scan_wrapper(x,_):
        return block_orb(x,ham_data,prop,trial,wave_data,sampler,which_rhf)
    
    # propagate n_ene_blocks then do sr
    prop_data, (blk_wt,blk_en,blk_eorb) \
        = lax.scan(
        _block_scan_wrapper, prop_data, xs=None, length=sampler.n_ene_blocks
    )
    prop_data = prop.stochastic_reconfiguration_local(prop_data)
    prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)

    return prop_data, (blk_wt,blk_en,blk_eorb)

@partial(jit, static_argnums=(1, 3, 5,6))
def propagate_phaseless_orb(
    ham_data: dict,
    prop: propagation.propagator,
    prop_data: dict,
    trial: wavefunctions,
    wave_data: dict,
    sampler: sampling.sampler,
    which_rhf,
) -> Tuple[jax.Array, dict]:
    def _sr_block_scan_wrapper(x,_):
        return _sr_block_scan_orb(x,ham_data,prop,trial,wave_data,sampler,which_rhf)

    prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
    prop_data["n_killed_walkers"] = 0
    prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
    prop_data,(blk_wt,blk_en,blk_eorb) \
        = lax.scan(
        _sr_block_scan_wrapper, prop_data, xs=None, length=sampler.n_sr_blocks
    )
    prop_data["n_killed_walkers"] /= (
        sampler.n_sr_blocks * sampler.n_ene_blocks * prop.n_walkers
    )
    wt = jnp.sum(blk_wt)
    en = jnp.sum(blk_en * blk_wt) / wt
    orb_en = jnp.sum(blk_eorb * blk_wt) / wt

    return prop_data, (wt,en,orb_en)

from pyscf.ci.cisd import CISD
from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD
from ad_afqmc.lno.cc import LNOCCSD
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

    s1e = mf.get_ovlp()
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
        frzfrag, orbfrag, can_orbfrag \
            = lno_maker.make_lno(
                lno_cc,orbfragloc,THRESH_INTERNAL,thresh_pno
                )
        
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

        if options["trial"] == "rhf":
            print('# Using RHF trial wavefunction')
            ci1 = []
            ci2 = []
            ecorr_cc = '  None  '
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
                print(f'# lno-ccsd fragment correlation energy: {ecorr_cc}')
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
            print(f'# lno-mp2 fragment correlation energy: {ecorr_p2}')
        else:
            ecorr_p2 = '  None  '

        nelec_act = nactocc*2
        norb_act = nactocc+nactvir
        
        print(f'# number of active electrons: {nelec_act}')
        print(f'# number of active orbitals: {norb_act}')
        print(f'# number of frozen orbitals: {len(frzfrag)}')

        options["seed"] = seeds[ifrag]
        lnoafqmc_runner.prep_lnoafqmc_file(
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
        # if debug:
        #     script = f"{dir_path}/run_lnocc_frg_dbg.py"
        # else:
        #     script = f"{dir_path}/run_lnocc_frg.py"
        script = f"{dir_path}/run_lnocc_frg_test.py"
        os.system(
            f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1;"
            f"{mpi_prefix} python {script} {gpu_flag} |tee frg_{ifrag+1}.out"
        )

        with open(f'frg_{ifrag+1}.out', 'a') as out_file:
            print(f"lno-mp2 orb_corr: {ecorr_p2}",file=out_file)
            print(f"lno-ccsd orb_corr: {ecorr_cc}",file=out_file)
            print(f"number of active electrons: {nelec_act}",file=out_file)
            print(f"number of active orbitals: {norb_act}",file=out_file)

    from pyscf import mp
    mmp = mp.MP2(mf, frozen=frozen)
    e_mp2 = mmp.kernel()[0]

    # if debug:
    #     data_maker.frg2result_dbg(thresh_pno,len(run_frg_list),mf.e_tot,e_mp2=0)
    # else:
    frg2result(thresh_pno,len(run_frg_list),mf.e_tot,e_mp2)

    return None


def frg2result(lno_thresh,nfrag,e_mf,e_mp2):
    with open('results.out', 'w') as out_file:
        print('# frag  mp2_orb_corr  ccsd_orb_corr' \
              '  afqmc_orb_en  err' \
              '  nelec  norb  time',file=out_file)
        for ifrag in range(nfrag):
            with open(f"frg_{ifrag+1}.out","r") as read_file:
                for line in read_file:
                    if "use rhf trial" in line:
                        trial = 'rhf'
                    if "use norhf trial" in line:
                        trial = 'norhf'
                    if "use cisd trial" in line:
                        trial = 'cisd'
                    if "lno-mp2 orb_corr" in line:
                        e_mp2_orb_en = line.split()[2]
                    if "lno-ccsd orb_corr" in line:
                        e_ccsd_orb_en = line.split()[2]
                    if "lno-afqmc orbital energy" in line:
                        afqmc_orb_en = line.split()[-3]
                        afqmc_orb_en_err = line.split()[-1]
                    if "number of active electrons" in line:
                        nelec = line.split()[-1]
                    if "number of active orbitals" in line:
                        norb = line.split()[-1]
                    if "total run time" in line:
                        tot_time = line.split()[3]
                print(f'{ifrag+1:3d}  '
                      f'{e_mp2_orb_en}  {e_ccsd_orb_en}  '
                      f'{afqmc_orb_en}  {afqmc_orb_en_err}  '
                      f'{nelec}  {norb}  {tot_time}', file=out_file)

    data = []
    with open('results.out', 'r') as read_file:
        for line in read_file:
            if not line.startswith("#"):
                data = np.hstack((data,line.split()))
                data[data == 'None'] = '0.000000' 

    data = np.array(data.reshape(nfrag,8))
    e_mp2_orb_en = np.array(data[:,1],dtype='float32')
    e_ccsd_orb_en = np.array(data[:,2],dtype='float32')
    afqmc_orb_en = np.array(data[:,3],dtype='float32')
    afqmc_orb_en_err = np.array(data[:,4],dtype='float32')
    nelec = np.array(data[:,5],dtype='int32')
    norb = np.array(data[:,6],dtype='int32')
    tot_time = np.array(data[:,7],dtype='float32')

    e_mp2_corr = sum(e_mp2_orb_en)
    mp2_cr = e_mp2 - e_mp2_corr
    e_ccsd_corr = sum(e_ccsd_orb_en)
    afqmc_corr = sum(afqmc_orb_en)
    afqmc_corr_err = np.sqrt(sum(afqmc_orb_en_err**2))
    nelec_avg = np.mean(nelec)
    norb_avg = np.mean(norb)
    nelec_max = max(nelec)
    norb_max = max(norb)
    tot_time = sum(tot_time)

    e_mp2_corr = f'{e_mp2_corr:.6f}'
    e_ccsd_corr = f'{e_ccsd_corr:.6f}'
    afqmc_corr = f'{afqmc_corr:.6f}'
    afqmc_corr_err = f'{afqmc_corr_err:.6f}'

    with open('results.out', 'a') as out_file:
        out_file.write(f'# final results \n')
        out_file.write(f'# use {trial} trial wavefunction\n')
        out_file.write(f'# mean-field energy: {e_mf:.8f}\n')
        out_file.write(f'# lno-thresh {lno_thresh}\n')
        out_file.write(f'# e_mp2_corr: {e_mp2_corr}\n')
        out_file.write(f'# e_ccsd_corr: {e_ccsd_corr}\n')
        out_file.write(f'# afqmc_corr: {afqmc_corr} +/- {afqmc_corr_err}\n')
        out_file.write(f'# mp2_correction: {mp2_cr:.8f}\n')
        out_file.write(f'# number of electrons: average {nelec_avg:.2f} maxium {nelec_max}\n')
        out_file.write(f'# number of orbitals: average {norb_avg:.2f} maxium {norb_max}\n')
        out_file.write(f'# total run time: {tot_time:.2f}\n')
    
    return None

def sum_results(n_results):
    with open('sum_results.out', 'w') as out_file:
        print("# lno-thresh(occ,vir) "
              "  mp2_corr  ccsd_corr"
              "  afqmc_corr   err "
              "  mp2_cr nelec_avg   nelec_max  "
              "  norb_avg   norb_max  "
              "  run_time",file=out_file)
        for i in range(n_results):
            with open(f"results.out{i+1}","r") as read_file:
                for line in read_file:
                    if 'wavefunction' in line:
                        trial = line.split()[-3]
                    if "lno-thresh" in line:
                        thresh_occ = line.split()[-2]
                        thresh_vir = line.split()[-1]
                        thresh_occ = float(thresh_occ.strip('()[],'))
                        thresh_vir = float(thresh_vir.strip('()[],'))
                    if "e_mp2_corr:" in line:
                        mp2_corr = line.split()[-1]
                    if "e_ccsd_corr:" in line:
                        ccsd_corr = line.split()[-1]
                    if "afqmc_corr:" in line:
                        afqmc_corr = line.split()[-3]
                        afqmc_corr_err = line.split()[-1]
                    if "mp2_correction:" in line:
                        mp2_cr = line.split()[-1]
                    if "electrons:" in line:
                        nelec_avg = line.split()[-3]
                        nelec_max = line.split()[-1]
                    if "orbitals:" in line:
                        norb_avg = line.split()[-3]
                        norb_max = line.split()[-1]
                    if "time:" in line:
                        run_time = line.split()[-1]
            print(f" ({thresh_occ:.2e},{thresh_vir:.2e}) \t"
                  f" {mp2_corr} \t {ccsd_corr} \t"
                  f" {afqmc_corr} \t {afqmc_corr_err} \t"
                  f" {mp2_cr}  {nelec_avg} \t {nelec_max} \t"
                  f" {norb_avg}  \t {norb_max} \t {run_time}",file=out_file)
        print(f'# use {trial} trial wavefunction',file=out_file)
    return None