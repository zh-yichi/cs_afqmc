import sys
from pyscf.lib import logger
from ad_afqmc.lno_ccsd import lno_ccsd
from pyscf import gto, scf

log = logger.Logger(sys.stdout, 6)

a = 0.9
nH = 6
atoms = ""
for i in range(nH):
    atoms += f"H {i*a} 0 0 \n"

mol = gto.M(atom=atoms, basis="ccpvdz", verbose=4)
mf = scf.RHF(mol).density_fit()
mf.kernel()

# frozen = 0
# mmp = mp.MP2(mf,frozen=frozen)
# mmp.kernel()[0]

# # cc
# mycc = cc.CCSD(mf)
# mycc.kernel()
# et = mycc.ccsd_t()

# fci
#cisolver = fci.FCI(mf)
#fci_ene, fci_vec = cisolver.kernel()

# print(f'rhf energy is {mf.e_tot}')
# print(f"ccsd energy is {mycc.e_tot}")
# print(f"ccsd_t energy is {mycc.e_tot+et}")
# print(f"ccsd correlation energy is {mycc.e_corr}")

options = {'n_eql': 6,
           'n_prop_steps': 20,
            'n_ene_blocks': 1,
            'n_sr_blocks': 10,
            'n_blocks': 20,
            'n_walkers': 50,
            'seed': 2,
            'walker_type': 'rhf',
            'trial': 'cisd',
            'dt':0.01,
            'ad_mode':None,
            }


thresh = 1.00e-04
lno_ccsd.run_lno_ccsd_afqmc(mf,thresh,[],options,1e-6,8)