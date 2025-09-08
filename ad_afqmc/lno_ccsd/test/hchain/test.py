import sys
from pyscf.lib import logger
from ad_afqmc.lno_ccsd import lno_ccsd
from pyscf import gto, scf, cc

log = logger.Logger(sys.stdout, 6)

a = 1.
nH = 6
atoms = ""
for i in range(nH):
    atoms += f"H {i*a} 0 0 \n"

mol = gto.M(atom=atoms, basis="sto6g", verbose=4)
mf = scf.RHF(mol)#.density_fit()
mf.kernel()

options = {'n_eql': 4,
           'n_prop_steps': 50,
            'n_ene_blocks': 1,
            'n_sr_blocks': 20,
            'n_blocks': 10,
            'n_walkers': 40,
            'seed': 2,
            'walker_type': 'rhf',
            'trial': 'cisd',
            'dt':0.005,
            'ad_mode':None,
            'use_gpu': False,
            }

threshs = [1e-4,1e-5]
for i,thresh in enumerate(threshs):
    lno_ccsd.run_lno_ccsd_afqmc(mf,thresh,[],options,nproc=5)

lno_ccsd.sum_results(len(threshs))
