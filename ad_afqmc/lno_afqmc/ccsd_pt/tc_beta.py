options = {'n_eql': 4,
        'n_prop_steps': 50,
        'n_ene_blocks': 1,
        'n_sr_blocks': 5,
        'n_blocks': 50,
        'n_walkers': 50,
        'seed': 24,
        'walker_type': 'uhf',
        'trial': 'uccsd_pt2_beta',
        'dt':0.005,
        'free_projection':False,
        'ad_mode':None,
        'use_gpu': False,
        'max_error': 5e-4
        }

from ad_afqmc.lno_afqmc import ulno_afqmc
ulno_afqmc.run_lnoafqmc(options,nproc=1)

