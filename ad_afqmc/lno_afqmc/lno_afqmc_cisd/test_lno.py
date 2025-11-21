from functools import partial
from pyscf import gto, scf, cc
import numpy as np

from jax import config
config.update("jax_enable_x64", True)

print = partial(print, flush=True)

a = 2.
nH = 10
atoms = ""
for i in range(nH):
    atoms += f"H {i*a} 0 0 \n"

mol = gto.M(atom=atoms, basis="ccpvdz", unit='b', verbose=4)
mf = scf.RHF(mol).density_fit()
mf.kernel()

nfrozen = 0
mycc = cc.CCSD(mf,frozen=nfrozen)
mycc.kernel()
print(mycc.e_corr)

options = {'n_eql': 4,
           'n_prop_steps': 10,
            'n_ene_blocks': 1,
            'n_sr_blocks': 5,
            'n_blocks': 50,
            'n_walkers': 200,
            'seed': 2,
            'walker_type': 'rhf',
            'trial': 'cisd_ad',
            'dt':0.005,
            'free_projection':False,
            'ad_mode':None,
            'use_gpu': True,
            }

from pyscf.ci.cisd import CISD
from pyscf.cc.ccsd import CCSD
from pyscf import lib
from ad_afqmc.lno.cc import LNOCCSD
from ad_afqmc.lno_afqmc import lno_maker, lno_afqmc
from ad_afqmc.lno.base import lno

frozen = nfrozen
thresh = 1e-4
chol_cut = 1e-7
eris = None
run_frg_list = [0]

mfcc = mf

if isinstance(mfcc, (CCSD, CISD)):
    full_cisd = True
    lnomf = mfcc._scf
else:
    full_cisd = False
    lnomf = mfcc

if isinstance(thresh, list):
    thresh_occ, thresh_vir = thresh
else:
    thresh_occ = thresh*10
    thresh_vir = thresh

lno_cc = LNOCCSD(lnomf, thresh=thresh, frozen=frozen)
lno_cc.thresh_occ = thresh_occ
lno_cc.thresh_vir = thresh_vir
lno_cc.lo_type = 'boys'
lno_cc.no_type = 'ie'
no_type = 'ie'
lno_cc.frag_lolist = '1o'
lno_cc.force_outcore_ao2mo = True

s1e = lnomf.get_ovlp()
lococc = lno_cc.get_lo(lo_type='boys') # localized active occ orbitals
# lococc,locvir = lno_maker.get_lo(lno_cc,lo_type) ### fix this for DF
if eris is None: eris = lno_cc.ao2mo()

frag_lolist = [[i] for i in range(lococc.shape[1])]
print(frag_lolist)
nfrag = len(frag_lolist)

frozen_mask = lno_cc.get_frozen_mask()
thresh_pno = [thresh_occ,thresh_vir]
print(f'# lno thresh {thresh_pno}')

if run_frg_list is None:
    run_frg_list = range(nfrag)

frag_nonvlist = None
if frag_nonvlist is None: frag_nonvlist = lno_cc.frag_nonvlist
if frag_nonvlist is None: frag_nonvlist = [[None,None]] * nfrag

eorb_cc = np.empty(nfrag,dtype='float64')
    
from jax import random
seeds = random.randint(random.PRNGKey(options["seed"]),
                    shape=(len(run_frg_list),), minval=0, maxval=100*nfrag)

for ifrag in run_frg_list:
    print(f'\n########### running fragment {ifrag+1} ##########')

    fraglo = frag_lolist[ifrag]
    orbfragloc = lococc[:,fraglo]
    THRESH_INTERNAL = 1e-10
    frag_target_nocc, frag_target_nvir = frag_nonvlist[ifrag]
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

    # mp2 is not invariant to lno transformation
    # needs to be done in canoical HF orbitals
    # which the globel mp2 is calculated in
    print('# running fragment MP2')
    ecorr_p2 = \
        lno_maker.lno_mp2_frg_e(lnomf,frzfrag,orbfragloc,can_orbfrag)
    ecorr_p2 = f'{ecorr_p2:.8f}'
    print(f'# LNO-MP2 Orbital Energy: {ecorr_p2}')
    
    print('# running fragment CCSD')
    mcc, ecorr_cc = \
        lno_maker.lno_cc_solver(lnomf,orbfrag,orbfragloc,frozen=frzfrag)
    ecorr_cc = f'{ecorr_cc:.8f}'
    print(f'# LNO-CCSD Energy: {mcc.e_tot}')
    print(f'# LNO-CCSD Orbital Energy: {ecorr_cc}')

    ci1 = np.array(mcc.t1)
    ci2 = mcc.t2 + lib.einsum("ia,jb->ijab",ci1,ci1)

    options["seed"] = seeds[ifrag]
    lno_afqmc.prep_lnoafqmc(
        mf,orbfrag,options,
        norb_act=norb_act,nelec_act=nelec_act,
        prjlo=prjlo,norb_frozen=frzfrag,
        ci1=ci1,ci2=ci2,chol_cut=chol_cut,
        )
    lno_afqmc.run_afqmc(options)
