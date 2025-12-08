from pyscf import gto, scf, cc

a = 2 # bond length in a cluster
d = 2 # distance between each cluster
unit = 'b' # unit of length
na = 2  # size of a cluster (monomer)
nc = 5 # set as integer multiple of monomers
spin = 0 # spin per monomer
frozen = 0 # frozen orbital per monomer
elmt = 'H'
atoms = ""
for n in range(nc*na):
    shift = ((n - n % na) // na) * (d-a)
    atoms += f"{elmt} {n*a+shift:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis="sto6g", unit='b', verbose=4)
mf = scf.RHF(mol).density_fit()
mf.kernel()

nfrozen = 0
mycc = cc.CCSD(mf,frozen=nfrozen)
mycc.kernel()
print(mycc.e_corr)

options = {'n_eql': 5,
           'n_prop_steps': 50,
            'n_ene_blocks': 1,
            'n_sr_blocks': 10,
            'n_blocks': 10,
            'n_walkers': 1,
            'seed': 98,
            'walker_type': 'rhf',
            'trial': 'cisd_ad',
            'dt':0.005,
            'free_projection':False,
            'ad_mode':None,
            'use_gpu': False,
            }

from ad_afqmc.lno.afqmc import LNOAFQMC
filename = 'fragmentenergies.txt'
frozen = 0
for thresh in [1e-4]:
    f = open(filename,'w')
    f.close()
    mfcc = LNOAFQMC(mf, thresh=thresh, frozen=frozen).set(verbose=5)
    mfcc.thresh_occ = 10*thresh
    mfcc.thresh_vir = thresh
    mfcc.nblocks = 10
    mfcc.seed = 98
    mfcc.lo_type = 'boys'
    mfcc.no_type = 'cim'
    mfcc.frag_lolist = '1o'
    mfcc.nwalk_per_proc = 1
    mfcc.nproc = 1
    mfcc.force_outcore_ao2mo = True
    mfcc.kernel()#canonicalize=False,chol_vecs=chol_vecs)
    ecc = mfcc.e_corr

    print("LNO-AFQMC/HF Energy: ", ecc)