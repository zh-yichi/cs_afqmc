from functools import partial

import numpy as np
from pyscf import fci, gto, scf, cc

from ad_afqmc import pyscf_interface, run_afqmc

print = partial(print, flush=True)

a = 0.9
nH = 6
atoms = ""
for i in range(nH):
    atoms += f"H {i*a} 0 0 \n"

mol = gto.M(atom=atoms, basis="ccpvdz", verbose=4)
mf = scf.RHF(mol).density_fit()
mf.kernel()

# cc
mycc = cc.CCSD(mf)
mycc.kernel()
et = mycc.ccsd_t()
print(f"ccsd energy is {mycc.e_tot}")
print(f"ccsd_t energy is {mycc.e_tot+et}")

# fci
# cisolver = fci.FCI(mf)
# fci_ene, fci_vec = cisolver.kernel()
# print(f"fci_ene: {fci_ene}", flush=True)

# ad afqmc
pyscf_interface.prep_afqmc(mycc)
options = {
    "n_eql": 3,
    "n_ene_blocks": 1,
    "n_sr_blocks": 50,
    "n_blocks": 10,
    "n_walkers": 50,
    "seed": 98,
    "trial": "cisd",
    "walker_type": "rhf",
}

# serial run
# run_afqmc.run_afqmc(options=options, mpi_prefix='')

# mpi run
from mpi4py import MPI

MPI.Finalize()
run_afqmc.run_afqmc(options=options, nproc=8)
