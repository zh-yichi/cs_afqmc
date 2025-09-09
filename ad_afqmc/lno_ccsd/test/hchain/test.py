from ad_afqmc.lno_ccsd import lno_ccsd
from pyscf import gto, scf, cc
import os

a = 1.
nH = 6
atoms = ""
for i in range(nH):
    atoms += f"H {i*a} 0 0 \n"

mol = gto.M(atom=atoms, basis="sto6g", verbose=4)
mf = scf.RHF(mol).density_fit()
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

threshs = [1e-4]
for i,thresh in enumerate(threshs):
    lno_ccsd.run_lno_ccsd_afqmc(mf,thresh,[],options,nproc=5,debug=True)
    os.system(f"mv results.out results.out{i+1}")

lno_ccsd.sum_results_dbg(len(threshs))
