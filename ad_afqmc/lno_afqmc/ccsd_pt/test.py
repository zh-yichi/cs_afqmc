from pyscf import gto, scf, mp, cc

a = 2 # bond length in a cluster
d = 2 # distance between each cluster
unit = 'b' # unit of length
na = 3  # size of a cluster (monomer)
nc = 7 # set as integer multiple of monomers
spin = 0 # spin per monomer
frozen = 0 # frozen orbital per monomer
elmt = 'H'
basis = 'sto6g'
atoms = ""
for n in range(nc*na):
    shift = ((n - n % na) // na) * (d-a)
    atoms += f"{elmt} {n*a+shift:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis=basis, spin=1, unit=unit, verbose=4, max_memory=16000)
mf = scf.UHF(mol).density_fit()
mf.kernel()

mo, _ = mf.stability()
dm = mf.make_rdm1(mo,mf.mo_occ)
mf.kernel(dm0=dm)
mo, _ = mf.stability()
dm = mf.make_rdm1(mo,mf.mo_occ)
mf.kernel(dm0=dm)
mf.stability()

nfrozen = 0
# mmp = mp.UMP2(mf, frozen=nfrozen)
# mmp.kernel()

mycc = cc.UCCSD(mf, frozen=nfrozen)
eris = mycc.ao2mo()
mycc.kernel(eris=eris)
et = mycc.ccsd_t(eris=eris)

from pyscf import lo
import numpy as np
lo_method = lo.Boys
orbocca = mf.mo_coeff[0][:,nfrozen:np.count_nonzero(mf.mo_occ[0])]
orbloca = lo_method(mol, orbocca).kernel()
orboccb = mf.mo_coeff[1][:,nfrozen:np.count_nonzero(mf.mo_occ[1])]
orblocb = lo_method(mol, orboccb).kernel()
lo_coeff = [orbloca, orblocb]

oa = [[[i],[]] for i in range(orbloca.shape[1])]
ob = [[[],[i]] for i in range(orblocb.shape[1])]
frag_lolist = oa + ob

options = {'n_eql': 2,
        'n_prop_steps': 50,
        'n_ene_blocks': 1,
        'n_sr_blocks': 5,
        'n_blocks': 10,
        'n_walkers': 50,
        'seed': 98,
        'walker_type': 'uhf',
        'trial': 'uccsd_pt',
        'dt':0.005,
        'free_projection':False,
        'ad_mode':None,
        'use_gpu': True,
        'max_error': 5e-4
        }

from ad_afqmc.lno_afqmc import ulno_afqmc
ulno_afqmc.run_afqmc(mf,options,lo_coeff,frag_lolist,thresh=1e-4,run_frg_list=[0])
