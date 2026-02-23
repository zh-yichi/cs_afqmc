#options =  {'n_eql': 3,
#            'n_prop_steps': 50,
#            'n_ene_blocks': 1,
#            'n_sr_blocks': 5,
#            'n_blocks': 100,
#            'dt': 0.005,
#            'n_walkers': 100,
#            'seed': 2,
#            'walker_type': 'rhf',
#            'trial': 'stoccsd3',
#            'nslater': 100, # number of slater determinants to sample the ccsd
#            'use_gpu': False,
#            }
#
#from ad_afqmc.prop_unrestricted.mixed_wave import prep, launch_afqmc
#import jax
#jax.config.update("jax_enable_x64", True)
## prep.prep_afqmc(mycc,chol_cut=1e-5)
#launch_afqmc.run_afqmc(options)


options =  {'n_eql': 3,
            'n_prop_steps': 50,
            'n_ene_blocks': 1,
            'n_sr_blocks': 5,
            'n_blocks': 100,
            'dt': 0.005,
            'n_walkers': 100,
            'seed': 2,
            'walker_type': 'rhf',
            'trial': 'stoccsd4',
            'nslater': 100, # number of slater determinants to sample the ccsd
            'use_gpu': False,
            }

from ad_afqmc.prop_unrestricted.mixed_wave import prep, launch_afqmc
import jax
jax.config.update("jax_enable_x64", True)
# prep.prep_afqmc(mycc,chol_cut=1e-5)
launch_afqmc.run_afqmc(options)
