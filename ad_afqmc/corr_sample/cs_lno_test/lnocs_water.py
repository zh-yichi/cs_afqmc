from pyscf import gto, scf

d = 100
basis = 'cc-pvdz'
atom1 = '''
O  -1.551007  -0.114520   0.000000
H  -1.934259   0.762503   0.000000
H  -0.599677   0.040712   0.000000
O   1.350625   0.111469   0.000000
H   1.680398  -0.373741  -0.758561
H   1.680398  -0.373741   0.758561
'''
atom2 = f'''
O  -1.551007  -0.114520   0.000000
H  -1.934259   0.762503   0.000000
H  -0.599677   0.040712   0.000000
O   {1.350625+d}   0.111469   0.000000
H   {1.680398+d}  -0.373741  -0.758561
H   {1.680398+d}  -0.373741   0.758561
'''

mol1 = gto.M(atom=atom1,basis=basis,verbose=4)
mol2 = gto.M(atom=atom2,basis=basis,verbose=4)

mf1 = scf.RHF(mol1).density_fit()
mf1.newton()
mf1.kernel()

mf2 = scf.RHF(mol2).density_fit()
mf2.newton()
mf2.kernel()

frozen = 2

options = {
    "dt": 0.01,
    "n_exp_terms": 6,
    "n_walkers": 40,
    "n_runs": 200,
    "rlx_steps": 0,
    "prop_steps": 10,
    "seed": 23,
    "walker_type": "rhf",
    "trial": "rhf",
    "use_gpu": False,
    "free_proj": False,
}
from ad_afqmc.corr_sample import lnocs
lnocs.run_cs_frags(mf1,mf2,2,options,nproc=10, lno_thresh=1e-4)
