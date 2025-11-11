from pyscf import gto, scf, cc

a = 1.20577 # bond length in a cluster
d = 10 # distance between each cluster
unit = 'A' # unit of length
na = 2  # size of a cluster (monomer)
nc = 1 # set as integer multiple of monomers
spin = 2 # spin per monomer
frozen = 2 # frozen orbital per monomer
elmt = 'O'
atoms = ""
for n in range(nc*na):
    shift = ((n - n % na) // na) * (d-a)
    atoms += f"{elmt} {n*a+shift:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis="sto6g",spin=spin*nc, unit=unit, verbose=4)
mol.build()

mf = scf.UHF(mol)
e = mf.kernel()

nfrozen = nc*frozen

mycc = cc.CCSD(mf,frozen=nfrozen)
mycc.kernel()

from mpi4py import MPI
if not MPI.Is_finalized():
    MPI.Finalize()

options = {'n_eql': 3,
           'n_prop_steps': 50,
            'n_ene_blocks': 5,
            'n_sr_blocks': 10,
            'n_blocks': 40,
            'n_walkers': 50,
            'seed': 2,
            'walker_type': 'uhf',
            'trial': 'uccsd_pt2',
            'dt':0.005,
            'free_projection':False,
            'ad_mode':None,
            'use_gpu': True,
            }

from ad_afqmc import config
config.afqmc_config["use_gpu"] = True
config.setup_jax()

from ad_afqmc.prop_unrestricted import prop_unrestricted
prop_unrestricted.prep_afqmc(mycc,options,chol_cut=1e-5)
prop_unrestricted.run_afqmc(options,nproc=8)
