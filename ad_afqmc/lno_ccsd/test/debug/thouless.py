from pyscf import gto, scf, cc
from ad_afqmc.lno_ccsd import lno_maker

a = 1. # 2aB
nH = 6
atoms = ""
for i in range(nH):
    atoms += f"H {i*a:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis="sto6g", verbose=4)
mf = scf.RHF(mol)
mf.kernel()

mycc = cc.CCSD(mf)
mycc.kernel()

# from mpi4py import MPI
# MPI.Finalize()

options = {'n_eql': 5,
           'n_prop_steps': 30,
            'n_ene_blocks': 1,
            'n_sr_blocks': 10,
            'n_blocks': 5,
            'n_walkers': 1,
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

from ad_afqmc import pyscf_interface, run_afqmc
from mpi4py import MPI
if not MPI.Is_finalized():
    MPI.Finalize()

pyscf_interface.prep_afqmc(mf,chol_cut=1e-7)
run_afqmc.run_afqmc(options=options, nproc=5)
# ham_data, ham, prop, trial, wave_data, sampler, observable, options, _ \
#     = (mpi_jax._prep_afqmc(options))
# nwalkers = 1
# init_walkers = np.stack([wave_data["mo_coeff"]] * nwalkers, axis=0)
# wave_data["mo_coeff"] = wave_data["mo_coeff"] @ u
# driver.afqmc(ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI,init_walkers)