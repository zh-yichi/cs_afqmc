from pyscf import gto, scf, cc

a = 2 # bond length in a cluster
d = 10 # distance between each cluster
unit = 'b' # unit of length
na = 2  # size of a cluster (monomer)
nc = 1 # set as integer multiple of monomers
spin = 0 # spin per monomer
frozen = 0 # frozen orbital per monomer
elmt = 'O'
atoms = ""
for n in range(nc*na):
    shift = ((n - n % na) // na) * (d-a)
    atoms += f"{elmt} {n*a+shift:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis="ccpvdz", unit='b', verbose=4)
mf = scf.RHF(mol).density_fit()
mf.kernel()

nfrozen = 0
mycc = cc.CCSD(mf,frozen=nfrozen)
mycc.kernel()
print(mycc.e_corr)

options = {'n_eql': 5,
           'n_prop_steps': 50,
            'n_ene_blocks': 1,
            'n_sr_blocks': 10,
            'n_blocks': 10,
            'n_walkers': 100,
            'seed': 98,
            'walker_type': 'rhf',
            'trial': 'cisd_ad',
            'dt':0.005,
            'free_projection':False,
            'ad_mode':None,
            'use_gpu': False,
            }

from ad_afqmc.lno_afqmc import lno_afqmc
lno_afqmc.run_lnoafqmc(mf,options,lno_thresh=1e-7,run_frg_list=[0])