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
wave_data['xtau'] = trial.get_xtau(wave_data, prop_data)

prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
prop_data["n_killed_walkers"] = 0
prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
e_init = prop_data["e_estimate"]

oci, eci = trial.calc_energy_ci(prop_data["walkers"], ham_data, wave_data)
ocr, ecr = trial.calc_energy_cr(prop_data["walkers"], ham_data, wave_data)
eci_init = jnp.real(eci)[0]
ecc_init = jnp.real((oci*eci + ecr) / (oci + ocr))[0]

print(f'# Propagating with {options["n_walkers"]} walkers')
print("# Equilibration sweeps:")
print("# atom_time \t <CI|H|walker> \t <stoCC|H|walker> \t Walltime")
print(f"    {0.:.2f} \t {eci_init:.6f} \t {ecc_init:.6f} \t {time.time() - init_time:.2f}")

sampler_eq = sampling.sampler_stoccsd2(
    n_prop_steps=50, 
    n_ene_blocks=5, 
    n_sr_blocks=10, 
    n_chol = sampler.n_chol
    )

block_time = prop.dt * sampler_eq.n_prop_steps * sampler_eq.n_ene_blocks * sampler_eq.n_sr_blocks

for n in range(1,options["n_eql"]+1):
    prop_data, (wci, wcc, eci, ecc) \
        = sampler_eq.propagate_phaseless(prop_data, ham_data, prop, trial, wave_data)

    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_local(prop_data)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * eci

    print(f"   {n*block_time:.2f} \t {eci:.6f} \t {ecc:.6f} \t {time.time() - init_time:.2f} ")

print("# Sampling sweeps:")
print("#  nblock \t <CI|H|walker> \t error \t \t <stoCC|H|walker> \t error \t \t Walltime")

wci_sp = np.zeros(sampler.n_blocks,dtype="float64")
wcc_sp = np.zeros(sampler.n_blocks,dtype="float64")
eci_sp = np.zeros(sampler.n_blocks,dtype="float64")
ecc_sp = np.zeros(sampler.n_blocks,dtype="float64")

for n in range(sampler.n_blocks):
    prop_data, (wci, wcc, eci, ecc) \
        = sampler.propagate_phaseless(prop_data, ham_data, prop, trial, wave_data)
    
    wci_sp[n] = wci
    wcc_sp[n] = wcc
    eci_sp[n] = eci
    ecc_sp[n] = ecc

    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_local(prop_data)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * eci

    if (n+1) % (max(sampler.n_blocks // 10, 1)) == 0 and n > 0:
        eci, eci_err = \
            stat_utils.blocking_analysis(wci_sp[: n + 1], eci_sp[: n + 1])
        ecc, ecc_err = \
            stat_utils.blocking_analysis(wcc_sp[: n + 1], ecc_sp[: n + 1])
        if eci_err is not None and ecc_err is not None:
            print(f"  {n+1:4d} \t \t {eci:.6f} \t {eci_err:.6f} \t {ecc:.6f} \t {ecc_err:.6f} \t {time.time() - init_time:.2f}")
            if eci_err < options["max_error"] and ecc_err < options["max_error"]:
                break
        else:
            print(f"  {n+1:4d} \t \t {eci:.6f} \t -------- \t {ecc:.6f} \t -------- \t {time.time() - init_time:.2f}")


nsamples = n + 1
print(f'# total number of samples {nsamples}')
wci_sp = wci_sp[:nsamples]
wcc_sp = wcc_sp[:nsamples]
eci_sp = eci_sp[:nsamples]
ecc_sp = ecc_sp[:nsamples]

####### CISD ##########
samples_clean, idx = stat_utils.reject_outliers(np.stack((wci_sp, eci_sp)).T, 1)
print(f"# Number of outliers in CISD Trial post: {nsamples - samples_clean.shape[0]} ")

wci_sp = samples_clean[:, 0]
eci_sp = samples_clean[:, 1]

eci, eci_err = stat_utils.blocking_analysis(wci_sp, eci_sp, neql=0, printQ=True)

if eci_err is not None:
    eci_err = f"{eci_err:.6f}"
else:
    eci_err = f"  {eci_err}  "
eci = f"{eci:.6f}"

####### sto-CCSD ##########
samples_clean, idx = stat_utils.reject_outliers(np.stack((wcc_sp, ecc_sp)).T, 1)
print(f"# Number of outliers in sto-CCSD Trial post: {nsamples - samples_clean.shape[0]} ")

wcc_sp = samples_clean[:, 0]
ecc_sp = samples_clean[:, 1]

ecc, ecc_err = stat_utils.blocking_analysis(wcc_sp, ecc_sp, neql=0, printQ=True)

if ecc_err is not None:
    ecc_err = f"{ecc_err:.6f}"
else:
    ecc_err = f"  {ecc_err}  "
ecc = f"{ecc:.6f}"

print(f"Final Results:")
print(f"AFQMC/CISD energy: {eci} +/- {eci_err}")
print(f"AFQMC/sto-CCSD energy: {ecc} +/- {ecc_err}")
print(f"total run time: {time.time() - init_time:.2f}")
