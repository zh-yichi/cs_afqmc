from pyscf import gto, scf, cc, lib

nn2 = 1
s = 20

def n2(d):
    n2 = f'''
    N 0 {d} 0
    N 1.120268 {d} 0
    '''
    return n2

atoms = ''
for i in range(nn2):
    atoms += n2(i*s)

    mol = gto.M(atom=atoms, basis="sto6g", verbose=4)
    mf = scf.RHF(mol).density_fit()
    mf.kernel()

    nfrozen = 2*(i+1)
    mycc = cc.CCSD(mf, frozen=nfrozen)
    t1,t2 = mycc.kernel()[1:]
    t2 += lib.einsum('ia,jb->ijab',t1,t1)

from ad_afqmc.lno_ccsd import lno_ccsd
from mpi4py import MPI
from ad_afqmc import run_afqmc

options = {'n_eql': 5,
           'n_prop_steps': 50,
            'n_ene_blocks': 1,
            'n_sr_blocks': 20,
            'n_blocks': 20,
            'n_walkers': 30,
            'seed': 2,
            'walker_type': 'rhf',
            'trial': 'cisd',
            'dt':0.005,
            'ad_mode':None,
            }

norb_act = 8
nelec_act = 10
norb_frozen = 2
mo_coeff = mf.mo_coeff


lno_ccsd.prep_lno_amp_chol_file(mf,mf.mo_coeff,options,norb_act=8,nelec_act=10,
                                prjlo=[],norb_frozen=[0,1],t1=t1,t2=t2,
                                chol_cut=1e-5,
                                option_file='options.bin',
                                mo_file="mo_coeff.npz",
                                amp_file="amplitudes.npz",
                                chol_file="FCIDUMP_chol"
                                )

# MPI.Finalize()
# run_afqmc.run_afqmc(options,nproc=4)