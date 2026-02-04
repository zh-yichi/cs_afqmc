import os
import pickle
from ad_afqmc import config

def run_afqmc(options,nproc=None,dbg=False,
              option_file='options.bin'):

    with open(option_file, 'wb') as f:
        pickle.dump(options, f)

    use_gpu = options["use_gpu"]
    if use_gpu:
        config.afqmc_config["use_gpu"] = True
        config.setup_jax()
        print(f'# running AFQMC on GPU')
        gpu_flag = "--use_gpu"
        mpi_prefix = ""
    else:
        print(f'# running AFQMC on CPU')
        gpu_flag = ""
        mpi_prefix = "mpirun "
        if nproc is not None:
            mpi_prefix += f"-np {nproc} "
    if  'pt' in options['trial'] and 'cc' in options['trial']:
        if 'pt2' in options['trial']:
            if dbg:
                script='run_afqmc_ccsd_pt2_dbg.py'
            else:
                script='run_afqmc_ccsd_pt2.py'
        else:
            script='run_afqmc_ccsd_pt.py'

    else:
        script='run_afqmc_sampling.py'

    if options["free_projection"]:
        script = 'run_fp_afqmc_sampling.py'
        if options['fp_abs']:
            script = 'run_fpabs_afqmc_sampling.py'

    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    script = f"{dir_path}/{script}"
    print(f'# AFQMC script: {script}')
    
    os.system(
        f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1;"
        f"{mpi_prefix} python {script} {gpu_flag} |tee afqmc.out"
    )
