import os
import pickle
from functools import partial

import numpy as np

from ad_afqmc import config
from ad_afqmc import pyscf_interface
import struct
import time
from typing import Any, Optional, Sequence, Tuple, Union

import h5py
import jax.numpy as jnp
import numpy as np
import scipy
from pyscf import __config__, ao2mo, df, dft, lib, mcscf, scf

print = partial(print, flush=True)

mo_file = "mo_coeff.npz"
amp_file = "amplitudes.npz"
chol_file = "FCIDUMP_chol"

def run_afqmc(options=None, script=None, mpi_prefix=None, nproc=None):
    from mpi4py import MPI
    if not MPI.Is_finalized():
        MPI.Finalize()
    if options is None:
        options = {}
    with open("options.bin", "wb") as f:
        pickle.dump(options, f)
    if script is None:
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        script = f"{dir_path}/mpi_jax.py"
    use_gpu = options["use_gpu"]
    if use_gpu:
        print(f'# running AFQMC on GPU')
        gpu_flag = "--use_gpu"
        mpi_prefix = ""
        nproc = None
        config.afqmc_config["use_gpu"] = True
        config.setup_jax()
        MPI = config.setup_comm()
    else:
        print(f'# running AFQMC on CPU')
        gpu_flag = ""
        mpi_prefix = "mpirun "
        if nproc is not None:
            mpi_prefix += f"-np {nproc} "
    os.system(
        f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; {mpi_prefix} python {script} {gpu_flag}"
    )
    try:
        ene_err = np.loadtxt("ene_err.txt")
    except:
        print("AFQMC did not execute correctly.")
        ene_err = 0.0, 0.0
    return ene_err[0], ene_err[1]


def run_afqmc_fp(options=None, script=None, mpi_prefix=None, nproc=None):
    if options is None:
        options = {}
    with open("options.bin", "wb") as f:
        pickle.dump(options, f)
    if script is None:
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        script = f"{dir_path}/mpi_jax.py"
    if mpi_prefix is None:
        mpi_prefix = "mpirun "
    if nproc is not None:
        mpi_prefix += f"-np {nproc} "
    os.system(
        f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; {mpi_prefix} python {script}"
    )
    # ene_err = np.loadtxt('ene_err.txt')
    # return ene_err[0], ene_err[1]

def run_afqmc_lno_mf(mf,
                     mo_coeff = None,
                     norb_act = None,
                     nelec_act=None,
                     norb_frozen = [],
                     orbitalE=-2,
                     script=None,
                     integrals=None,
                     dt = 0.005,
                     nwalk_per_proc = 5,
                     nblocks = 100,
                     mpi_prefix = None,                     
                     nproc = None,
                     chol_cut = 1e-5,
                     maxError=1e-4,
                     prjlo=None):

    print("#\n# Preparing LNO-AFQMC calculation")
    options = {'n_eql': 1,
             'n_ene_blocks': 1,
             'n_sr_blocks': 10,
             'n_blocks': nblocks,
             'n_walkers': nwalk_per_proc,
             'seed': 98,
             'walker_type': 'rhf',
             'trial': 'rhf',
             'dt':dt,
             'ad_mode':None,
             'orbE':orbitalE,
             'prjlo':prjlo,
             'maxError':maxError,
             'LNO':True,
             }
    import pickle
    with open('options.bin', 'wb') as f:
        pickle.dump(options, f)

    mol = mf.mol
    # choose the orbital basis
    if mo_coeff is None:
        if isinstance(mf, scf.uhf.UHF):
            mo_coeff = mf.mo_coeff[0]
        elif isinstance(mf, scf.rhf.RHF):
            mo_coeff = mf.mo_coeff
        else:
            raise Exception("# Invalid mean field object!")

    # calculate cholesky integrals
    print("# Calculating Cholesky integrals")
    h1e, chol, nelec, enuc, nbasis, nchol = [ None ] * 6
    if integrals is not None:
      enuc = integrals['h0']
      h1e = integrals['h1']
      eri = integrals['h2']
      nelec = mol.nelec
      nbasis = h1e.shape[-1]
      norb = nbasis
      eri = ao2mo.restore(4, eri, norb)
      chol0 = pyscf_interface.modified_cholesky(eri, chol_cut)
      nchol = chol0.shape[0]
      chol = np.zeros((nchol, norb, norb))
      for i in range(nchol):
          for m in range(norb):
              for n in range(m + 1):
                  triind = m * (m + 1) // 2 + n
                  chol[i, m, n] = chol0[i, triind]
                  chol[i, n, m] = chol0[i, triind]

    else:
        print('# Generating Cholesky Integrals')
        #_, chol, _, _ = pyscf_interface.generate_integrals(mol, mf.get_hcore(), mo_coeff, chol_cut)
        # nbasis = h1e.shape[-1]

        mc = mcscf.CASSCF(mf, norb_act, nelec_act) 
        mc.frozen = norb_frozen
        nelec = mc.nelecas
        mc.mo_coeff = mo_coeff
        h1e, enuc = mc.get_h1eff()

        nbasis = h1e.shape[-1]
        print(f'# frozen orbitals are {norb_frozen}')
        if isinstance(norb_frozen, (int, float)) and norb_frozen == 0:
            norb_frozen = []
        act = np.array([i for i in range(mol.nao) if i not in norb_frozen])
        print(f'# local active orbitals are {act}') #yichi
        print(f'# local active space size {len(act)}') #yichi
        e = ao2mo.kernel(mf.mol,mo_coeff[:,act],compact=False)
        print(f'# loc_eris shape: {e.shape}') #yichi
        # add e = pyscf_interface.df(mol_mf,e) for selected loc_mos
        chol = pyscf_interface.modified_cholesky(e,max_error = chol_cut)
        print(f'# chol shape: {chol.shape}') #yichi
    
    print("# Finished calculating Cholesky integrals\n")
    print('# Size of the correlation space:')
    print(f'# Number of electrons: {nelec}')
    print(f'# Number of basis functions: {nbasis}')
    print(f'# Number of Cholesky vectors: {chol.shape[0]}\n')
    chol = chol.reshape((-1, nbasis, nbasis))
    v0 = 0.5 * np.einsum('nik,njk->ij', chol, chol, optimize='optimal')
    h1e_mod = h1e - v0
    chol = chol.reshape((chol.shape[0], -1))

    # write mo coefficients
    trial_coeffs = np.empty((2, nbasis, nbasis))
    overlap = mf.get_ovlp(mol)
    if isinstance(mf, (scf.uhf.UHF, scf.rohf.ROHF)):
        uhfCoeffs = np.empty((nbasis, 2 * nbasis))
        if isinstance(mf, scf.uhf.UHF):
            q, r = np.linalg.qr(mo_coeff[:, norb_frozen:].T.dot(overlap).dot(mf.mo_coeff[0][:, norb_frozen:]))
            uhfCoeffs[:, :nbasis] = q
            q, r = np.linalg.qr(mo_coeff[:, norb_frozen:].T.dot(overlap).dot(mf.mo_coeff[1][:, norb_frozen:]))
            uhfCoeffs[:, nbasis:] = q
        else:
            q, r = np.linalg.qr(mo_coeff[:, norb_frozen:].T.dot(overlap).dot(mf.mo_coeff[:, norb_frozen:]))
            uhfCoeffs[:, :nbasis] = q
            uhfCoeffs[:, nbasis:] = q

        trial_coeffs[0] = uhfCoeffs[:, :nbasis]
        trial_coeffs[1] = uhfCoeffs[:, nbasis:]
        #np.savetxt("uhf.txt", uhfCoeffs)
        np.savez('mo_coeff.npz', mo_coeff=trial_coeffs)

    elif isinstance(mf, scf.rhf.RHF):
        #q, r = np.linalg.qr(mo_coeff[:, norb_frozen:].T.dot(overlap).dot(mf.mo_coeff[:, norb_frozen:]))
        q = np.eye(mol.nao- len(norb_frozen))
        trial_coeffs[0] = q
        trial_coeffs[1] = q
        np.savez("mo_coeff.npz",mo_coeff=trial_coeffs)

    pyscf_interface.write_dqmc(h1e,h1e_mod,chol,sum(nelec),nbasis,enuc,ms=mol.spin,filename='FCIDUMP_chol',mo_coeffs=trial_coeffs)
    #if script is None:
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    script = f"{dir_path}/mpi_jax.py"
    use_gpu = config.afqmc_config["use_gpu"]
    gpu_flag = "--use_gpu" if use_gpu else ""
    if mpi_prefix is None:
        if use_gpu:
            mpi_prefix = ""
        else:
            mpi_prefix = "mpirun "
    if nproc is not None:
        mpi_prefix += f"-np {nproc} "
    os.system(
        f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; {mpi_prefix} python {script} {gpu_flag} > afqmc.out"
    )
    
    target_line_prefix = "orbE energy: "
    with open('afqmc.out', "r") as file:
        for line in file:
    	    if line.startswith(target_line_prefix):
                line = line[len(target_line_prefix):].strip()
                values = line.split()
                value1 = float(values[0])
                value2 = values[2]
                try:
                    value2 = float(value2)
                except ValueError:
                    pass  # Leave it as string if conversion fails
                break
    
    input_file = "afqmc.out"
    with open(input_file, "r") as infile:
        found_header = False
        for line in infile:
            if found_header:
                print(line.strip())
            elif line.strip().startswith("# Number of large deviations:"):
                found_header = True

    return value1,value2
