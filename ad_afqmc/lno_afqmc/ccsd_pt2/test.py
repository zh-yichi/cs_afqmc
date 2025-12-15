from pyscf import gto, scf, cc

a = 2 # bond length in a cluster
d = 10 # distance between each cluster
unit = 'b' # unit of length
na = 2  # size of a cluster (monomer)
nc = 2 # set as integer multiple of monomers
spin = 0 # spin per monomer
frozen = 0 # frozen orbital per monomer
elmt = 'H'
basis = 'sto6g'
atoms = ""
for n in range(nc*na):
    shift = ((n - n % na) // na) * (d-a)
    atoms += f"{elmt} {n*a+shift:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms,basis=basis,spin=spin*nc,unit=unit,verbose=4)
mol.build()

mf = scf.RHF(mol).density_fit()
mf.kernel()

nfrozen = nc*frozen

mycc = cc.CCSD(mf,frozen=nfrozen)
mycc.kernel()

from pyscf import lo
import numpy as np
lo_method = lo.PM
nocc = np.count_nonzero(mf.mo_occ)
orbocc = mf.mo_coeff[:, nfrozen:nocc]
lo_coeff = lo_method(mol, orbocc).kernel()

frag_lolist = [[i] for i in range(lo_coeff.shape[1])]

from pyscf.lno import lnoccsd
mcc = lnoccsd.LNOCCSD_T(mf, lo_coeff, frag_lolist, frozen=frozen).set(verbose=5)
mcc.lno_thresh = [1e-4,1e-5]
mcc.kernel()
# options = {'n_eql': 3,
#            'n_prop_steps': 50,
#             'n_ene_blocks': 1,
#             'n_sr_blocks': 1,
#             'n_blocks': 10,
#             'n_walkers': 3,
#             'seed': 2,
#             'walker_type': 'rhf',
#             'trial': 'ccsd_pt2_ad',
#             'dt':0.005,
#             'free_projection':False,
#             'ad_mode':None,
#             'use_gpu': False,
#             "max_error":1e-4
#             }

# from ad_afqmc.lno_afqmc.ccsd_pt2 import lno_afqmc
# # lno_afqmc.prep_afqmc(mf,mf.mo_coeff,mycc.t1,mycc.t2,[],prjlo,options,chol_cut=1e-5)
# lno_afqmc.run_afqmc(mf,options,lo_coeff,frag_lolist,nproc=1)

# import numpy as np
# nocc = mol.nelectron // 2
# prjlo = np.eye(nocc)

# lno_afqmc.prep_afqmc(mf,mf.mo_coeff,mycc.t1,mycc.t2,[],prjlo,options,chol_cut=1e-5)
# lno_afqmc.run_lnoafqmc(options,nproc=1)

# from ad_afqmc.prop_unrestricted import prop_unrestricted
# prop_unrestricted.prep_afqmc(mycc,options,chol_cut=1e-5)
# prop_unrestricted.run_afqmc(options,nproc=1)