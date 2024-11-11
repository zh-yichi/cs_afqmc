from pyscf import gto, scf
basis = 'ccpvdz'

d1 = 5
atom1 = f'''
H 0 0 0
H {d1} 0 0
'''

mol1 = gto.Mole(
verbose=3,
atom=atom1,
basis=basis,
)

mol1.build()
mf1 = scf.RHF(mol1).density_fit()
mf1.kernel()

d2 = 6
atom2 = f'''
H 0 0 0
H {d2} 0 0
'''

mol2 = gto.Mole(
verbose=3,
atom=atom2,
basis=basis,
)

mol2.build()
mf2 = scf.RHF(mol2).density_fit()
mf2.kernel()

print('the rhf energy difference is: ',mf1.e_tot-mf2.e_tot)

from ad_afqmc import pyscf_interface, driver, mpi_jax

mo_file1="h2_mo1.npz"
chol_file1="h2_chol1"
pyscf_interface.prep_afqmc(mf1,mo_file=mo_file1,chol_file=chol_file1)
mo_file2="h2_mo2.npz"
chol_file2="h2_chol2"
pyscf_interface.prep_afqmc(mf2,mo_file=mo_file2,chol_file=chol_file2)
