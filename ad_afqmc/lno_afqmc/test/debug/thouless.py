from pyscf import gto, scf, cc
from ad_afqmc.lno_ccsd import lno_maker
import numpy as np

a = 1. # 2aB
nH = 10
atoms = ""
for i in range(nH):
    atoms += f"H {i*a:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis="sto6g", verbose=4)
mf = scf.RHF(mol)
mf.kernel()

mycc = cc.CCSD(mf)
mycc.kernel()

t2_0 = np.zeros(mycc.t2.shape)
eris = mycc.ao2mo(mycc.mo_coeff)
eccsd = mycc.energy(mycc.t1, t2_0, eris)
ecc_t2_0 = eccsd+mf.e_tot
print(f'ccsd energy with T2=0 is:{ecc_t2_0}')

# from mpi4py import MPI
# MPI.Finalize()

options = {'n_eql': 5,
           'n_prop_steps': 30,
            'n_ene_blocks': 1,
            'n_sr_blocks': 10,
            'n_blocks': 5,
            'n_walkers': 5,
            'seed': 98,
            'walker_type': 'rhf',
            'trial': 'rhf',
            'dt':0.005,
            'ad_mode':None,
            'use_gpu': False,
            }

u = lno_maker.thouless_trans(mycc.t1)
mo_t = mf.mo_coeff @ u
mf.mo_coeff = mo_t

from ad_afqmc import pyscf_interface, mpi_jax, driver
from mpi4py import MPI
#if not MPI.Is_finalized():
#    MPI.Finalize()

pyscf_interface.prep_afqmc(mf,chol_cut=1e-7)
ham_data, ham, prop, trial, wave_data, sampler, observable, options, _ \
    = (mpi_jax._prep_afqmc(options))

init_walkers = trial.get_init_walkers(wave_data, prop.n_walkers, restricted=True)
init_walkers = u.T @ init_walkers

driver.afqmc(ham_data,ham,prop,trial,wave_data,sampler,observable,options,MPI,init_walkers)
