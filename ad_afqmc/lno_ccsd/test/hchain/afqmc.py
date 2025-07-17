from functools import partial
from pyscf import gto, scf, cc
from ad_afqmc import pyscf_interface, run_afqmc

from jax import config
config.update("jax_enable_x64", True)

print = partial(print, flush=True)

a = 1.
nH = 4
atoms = ""
for i in range(nH):
    atoms += f"H {i*a} 0 0 \n"

mol = gto.M(atom=atoms, basis="sto6g", verbose=4)
mf = scf.RHF(mol).density_fit()
mf.kernel()

# cc
mycc = cc.CCSD(mf)
mycc.kernel()
et = mycc.ccsd_t()
print(f"ccsd energy is {mycc.e_tot}")
print(f"ccsd_t energy is {mycc.e_tot+et}")


options = {
    "n_eql": 4,
    "n_ene_blocks": 1,
    "n_sr_blocks": 20,
    "n_blocks": 40,
    "n_walkers": 50,
    "seed": 98,
    "trial": "cisd",
    "walker_type": "rhf",
    "dt":0.005,
}


mo_coeff = mf.mo_coeff
nfrozen = 0
norb_act = mol.nao - nfrozen
nelec_act = mol.nelectron - 2 * nfrozen
t1 = mycc.t1
t2 = mycc.t2


from ad_afqmc.lno_ccsd import lno_ccsd
lno_ccsd.prep_lno_amp_chol_file(mf,mo_coeff,options,norb_act=norb_act,nelec_act=nelec_act,
                                prjlo=[],norb_frozen=nfrozen,t1=t1,t2=t2,
                                df=True,chol_cut=1e-6,
                                option_file='options.bin',
                                mo_file="mo_coeff.npz",
                                amp_file="amplitudes.npz",
                                chol_file="FCIDUMP_chol"
                                )

from mpi4py import MPI

MPI.Finalize()
run_afqmc.run_afqmc(options=options, nproc=5)
