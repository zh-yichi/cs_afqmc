from ad_afqmc.lno_afqmc import lnoafqmc_runner, data_maker ,code_tester
from pyscf import gto, scf
import os

a = 1.
nH = 6
atoms = ""
for i in range(nH):
    atoms += f"H {i*a} 0 0 \n"

mol = gto.M(atom=atoms, basis="sto6g", verbose=4)
mf = scf.RHF(mol).density_fit()
mf.kernel()

# mycc = cc.CCSD(mf)
# mycc.kernel()

# myci = ci.CISD(mf)
# myci.kernel()

options = {'n_eql': 4,
           'n_prop_steps': 50,
            'n_ene_blocks': 1,
            'n_sr_blocks': 1,
            'n_blocks': 10,
            'n_walkers': 5,
            'seed': 2,
            'walker_type': 'rhf',
            'trial': 'rhf',
            'dt':0.005,
            'ad_mode':None,
            'use_gpu': False,
            'which_rhf': 1,
            }

threshs = [1e-8]
for i,thresh in enumerate(threshs):
    code_tester.run_lno_afqmc(mf,thresh,[],options,nproc=5)
    os.system(f"mv results.out results.out1")

code_tester.sum_results(1)
