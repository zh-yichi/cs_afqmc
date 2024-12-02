from pyscf import gto, scf
import numpy as np

basis = 'sto6g'


atom1 = f'''
K 0 0 0
'''

mol1 = gto.Mole(
verbose=3,
charge=1,
spin=0,
atom=atom1,
basis=basis,
)

mol1.build()
mf1 = scf.RHF(mol1).density_fit()
mf1 = mf1.newton()
mf1.kernel()


atom2 = f'''
K 0 0 0 
'''

mol2 = gto.Mole(
verbose=3,
atom=atom2,
charge=+1,
spin=2,
basis=basis,
)

mol2.build()
mf2 = scf.RHF(mol2).density_fit()
mf2 = mf2.newton()
mf2.kernel()

print('the rhf energy difference is: ',mf1.e_tot-mf2.e_tot)

from ad_afqmc import pyscf_interface, run_afqmc, mpi_jax, driver

mo_file1="mo1.npz"
chol_file1="chol1"
pyscf_interface.prep_afqmc(mf1,mo_file=mo_file1,chol_file=chol_file1)
mo_file2="mo2.npz"
chol_file2="chol2"
pyscf_interface.prep_afqmc(mf2,mo_file=mo_file2,chol_file=chol_file2)

# options1 = {
#     "dt": 0.005,
#     "n_eql": 4,
#     "n_ene_blocks": 1,
#     "n_sr_blocks": 10,
#     "n_blocks": 200,
#     "n_walkers": 300,
#     "seed": 2,
#     "walker_type": "rhf",
#     "trial": "rhf",
# }

# options2 = {
#     "dt": 0.005,
#     "n_eql": 4,
#     "n_ene_blocks": 1,
#     "n_sr_blocks": 10,
#     "n_blocks": 200,
#     "n_walkers": 300,
#     "seed": 98,
#     "walker_type": "rhf",
#     "trial": "rhf",
# }

# afqmc1 = (mpi_jax._prep_afqmc(options1,mo_file=mo_file1,chol_file=chol_file1))
# e_afqmc1, err_afqmc1 = driver.afqmc(*afqmc1)

# afqmc2 = (mpi_jax._prep_afqmc(options2,mo_file=mo_file2,chol_file=chol_file2))
# e_afqmc2, err_afqmc2 = driver.afqmc(*afqmc2)

# afqmc_en_diff = e_afqmc1 - e_afqmc2
# afqmc_en_diff_err = np.sqrt(err_afqmc1**2+err_afqmc2**2)
# print(f'the afqmc energy difference is {afqmc_en_diff}, error is {afqmc_en_diff_err}')