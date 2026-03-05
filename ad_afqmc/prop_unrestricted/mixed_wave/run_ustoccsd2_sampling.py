import time
import argparse
import numpy as np
from jax import random
from jax import numpy as jnp
from ad_afqmc import config, stat_utils
from ad_afqmc.prop_unrestricted.mixed_wave import prep, sampling
from functools import partial
print = partial(print, flush=True)

init_time = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    if args.use_gpu:
        config.afqmc_config["use_gpu"] = True

config.setup_jax()

ham_data, ham, prop, trial, wave_data, sampler, options = (prep._prep_afqmc())

print(f'# the sampler is: {sampler}')
print(f'# the trial is: {trial}')

init_walkers = None
trial_rdm1 = trial.get_rdm1(wave_data)
if "rdm1" not in wave_data:
    wave_data["rdm1"] = trial_rdm1
ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
ham_data = ham.build_propagation_intermediates(ham_data, prop, trial, wave_data)

prop_data = prop.init_prop_data(trial, wave_data, ham_data, init_walkers)

if jnp.abs(jnp.sum(prop_data["overlaps"])) < 1.0e-6:
    raise ValueError(
        "Initial overlaps are zero. Pass walkers with non-zero overlap."
    )

prop_data["key"] = random.PRNGKey(options["seed"])

xtaus = trial.get_xtaus(prop_data, wave_data, prop)

prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
prop_data["n_killed_walkers"] = 0
prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
e_init = prop_data["e_estimate"]

oci, eci = trial.calc_energy_cid(prop_data["walkers"], ham_data, wave_data)
num_cr, den_cr = trial.calc_correction(prop_data["walkers"], xtaus, ham_data, wave_data)
# den_cr = trial.calc_denominator(prop_data["walkers"], xtaus, wave_data)

eci_init = jnp.real(eci)[0]
ecc_init = jnp.real((oci*eci + num_cr) / (oci + den_cr))[0]

print(f'# Propagating with {options["n_walkers"]} walkers')
print("# Equilibration sweeps:")
print("# atom_time  energy_ci  numer_cr  denom_cr  e_stocc  Walltime")
print(f"  {0.:.2f}  {eci_init.real:.6f}  {num_cr[0].real:.6f}  {den_cr[0].real:.6f}  {ecc_init.real:.6f}  {time.time() - init_time:.2f}")

sampler_eq = sampling.sampler_stoccsd2(
    n_prop_steps=50, 
    # n_ene_blocks=1, 
    n_sr_blocks=50, 
    n_chol = sampler.n_chol
    )

block_time = prop.dt * sampler_eq.n_prop_steps * sampler_eq.n_sr_blocks

for n in range(1,options["n_eql"]+1):
    prop_data, (whf, num_ci, den_ci, num_cr, den_cr) \
        = sampler_eq.propagate_phaseless(prop_data, ham_data, prop, trial, wave_data)

    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_local(prop_data)

    eci = num_ci / den_ci
    ecc = (num_ci + num_cr) / (den_ci + den_cr)

    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * eci

    print(f" {n*block_time:.2f}  {eci.real:.6f}  {num_cr.real:.6f}  {den_cr.real:.6f}  {ecc.real:.6f}  {time.time() - init_time:.2f} ")

print("# Sampling sweeps:")
print("# nBlocks  energy_ci  error  energy_cc  error  Walltime")

whf_sp = np.zeros(sampler.n_blocks,dtype="float64")
numci_sp = np.zeros(sampler.n_blocks,dtype="complex128")
denci_sp = np.zeros(sampler.n_blocks,dtype="complex128")#float64")
numcr_sp = np.zeros(sampler.n_blocks,dtype="complex128")
dencr_sp = np.zeros(sampler.n_blocks,dtype="complex128")
eci_sp = np.zeros(sampler.n_blocks,dtype="complex128")
# ecc_sp = np.zeros(sampler.n_blocks,dtype="complex128")

for n in range(sampler.n_blocks):
    prop_data, (whf, numci, denci, numcr, dencr) \
        = sampler.propagate_phaseless(prop_data, ham_data, prop, trial, wave_data)

    whf_sp[n] = whf
    numci_sp[n] = numci
    denci_sp[n] = denci
    numcr_sp[n] = numcr
    dencr_sp[n] = dencr
    eci_sp[n] = jnp.real(numci / denci)
    # ecc_sp[n] = jnp.real((numci + numcr) / (denci + dencr))

    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_local(prop_data)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * eci_sp[n]

    if (n+1) % (max(sampler.n_blocks // 10, 1)) == 0 and n > 0:
        # calculate variance by covariance
        # E(CISD) = <CISD|H|AF> / <CISD|AF>

        whf = np.sum(whf_sp[:n+1])
        whf_numci = np.sum(whf_sp[:n+1] * numci_sp[:n+1])
        whf_denci = np.sum(whf_sp[:n+1] * denci_sp[:n+1])
        whf_numcr = np.sum(whf_sp[:n+1] * numcr_sp[:n+1])
        whf_dencr = np.sum(whf_sp[:n+1] * dencr_sp[:n+1])

        numci = whf_numci / whf
        denci = whf_denci / whf
        eci = numci / denci

        # partial_eci/partial_num, partial_eci/partial_don
        deci = [1/denci, -numci/denci**2]
        covci = np.cov([numci_sp[:n+1], denci_sp[:n+1]])
        eci_err = np.sqrt(deci @ covci @ deci) / np.sqrt((n))
        
        # E(CCSD) = (<CISD|H|AF> + <stoCC-stoCI|H|AF>) / (<CISD|AF> + <stoCC-stoCI|AF>)
        numcr = whf_numcr / whf
        dencr = whf_dencr / whf
        ecc = (numci + numcr) / (denci + dencr)

        # partial_ecc...
        decc = [1/(denci+dencr), 
                1/(denci+dencr), 
                -(numci+numcr)/(denci+dencr)**2, 
                -(numci+numcr)/(denci+dencr)**2,
                ]
        covcc = np.cov([numci_sp[:n+1], numcr_sp[:n+1], denci_sp[:n+1], dencr_sp[:n+1]])
        ecc_err = np.sqrt(decc @ covcc @ decc) / np.sqrt((n))

        ecc_err_jk = sampler.blocking(whf_sp[:n+1], numci_sp[:n+1], numcr_sp[:n+1], denci_sp[:n+1], dencr_sp[:n+1])

        print(f"  {n+1:4d}  {eci.real:.6f}  {eci_err.real:.6f}  {ecc.real:.6f}  {ecc_err.real:.6f}  {ecc_err_jk.real:.6f}  {time.time() - init_time:.2f}")


############################ done sampling ###########################
nsample = len(whf_sp)
whf = np.sum(whf_sp)
whf_numci = np.sum(whf_sp * numci_sp)
whf_denci = np.sum(whf_sp * denci_sp)
whf_numcr = np.sum(whf_sp * numcr_sp)
whf_dencr = np.sum(whf_sp * dencr_sp)

numci = whf_numci / whf
denci = whf_denci / whf
numcr = whf_numcr / whf
dencr = whf_dencr / whf

# CISD
eci = numci / denci
deci = [1/denci, -numci/denci**2]
covci = np.cov([numci_sp, denci_sp])
eci_err_cov = np.sqrt(deci @ covci @ deci) / np.sqrt((nsample))
eci_err_jk = sampler.blocking(whf_sp, numci_sp, numcr_sp*0, denci_sp, dencr_sp*0)

#CCSD
decc = [1/(denci+dencr), 
        1/(denci+dencr), 
        -(numci+numcr)/(denci+dencr)**2, 
        -(numci+numcr)/(denci+dencr)**2,
        ]
covcc = np.cov([numci_sp, numcr_sp, denci_sp, dencr_sp])
ecc_err_cov = np.sqrt(decc @ covcc @ decc) / np.sqrt((nsample))

ecc = (numci + numcr) / (denci + dencr)
ecc_err_jk = sampler.blocking(whf_sp, numci_sp, numcr_sp, denci_sp, dencr_sp)

print(f"Final Results:")
print(f"AFQMC/CISD energy (covariance): {eci.real:.6f} +/- {eci_err_cov.real:.6f}")
print(f"AFQMC/CISD energy (Jackknife): {eci.real:.6f} +/- {eci_err_jk.real:.6f}")
print(f"AFQMC/sto-CCSD energy (covariance): {ecc.real:.6f} +/- {ecc_err_cov.real:.6f}")
print(f"AFQMC/sto-CCSD energy (Jackknife): {ecc.real:.6f} +/- {ecc_err_jk.real:.6f}")
print(f"total run time: {time.time() - init_time:.2f}")
