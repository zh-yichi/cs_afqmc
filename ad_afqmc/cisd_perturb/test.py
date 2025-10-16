from pyscf import gto, scf, cc

a = 1.05835 # bond length in a cluster
d = 10 # distance between each cluster
na = 2  # size of a cluster (monomer)
nc = 1 # set as integer multiple of monomers
atoms = ""
for n in range(nc*na):
    shift = ((n - n % na) // na) * (d-a)
    atoms += f"H {n*a+shift:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis="sto6g", verbose=4)
mol.build()

mf = scf.RHF(mol)#.density_fit()
e = mf.kernel()

mycc = cc.CCSD(mf)
mycc.kernel()

options = {'n_eql': 2,
           'n_prop_steps': 50,
            'n_ene_blocks': 5,
            'n_sr_blocks': 5,
            'n_blocks': 20,
            'n_walkers': 30,
            'seed': 2,
            'walker_type': 'rhf',
            'trial': 'cisd',
            'dt':0.005,
            'free_projection':False,
            'ad_mode':None,
            'use_gpu': False,
            }

# from ad_afqmc import pyscf_interface, run_afqmc
from ad_afqmc.cisd_perturb import sample_ccsd_pt, sample_ccsd_pt2, sample_uccsd_pt, ccsd_pt
ccsd_pt.prep_afqmc(mycc,chol_cut=1e-7)

sample_ccsd_pt.run_afqmc_ccsd_pt(options,nproc=5)
#sample_uccsd_pt.run_afqmc_uccsd_pt(options,nproc=5)
# run_afqmc.run_afqmc(options,nproc=5)
