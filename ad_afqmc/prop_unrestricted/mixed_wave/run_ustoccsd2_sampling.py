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

sampler_eq = sampling.sampler_ustoccsd2(
    n_prop_steps=50, 
    n_ene_blocks=5, 
    n_sr_blocks=10, 
    n_chol = sampler.n_chol
    )

block_time = prop.dt * sampler_eq.n_prop_steps * sampler_eq.n_ene_blocks * sampler_eq.n_sr_blocks

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

        print(f"  {n+1:4d}  {eci.real:.6f}  {eci_err.real:.6f}  {ecc.real:.6f}  {ecc_err.real:.6f}  {time.time() - init_time:.2f}")

# calculate variance by Jackknife
whf_jk = whf - whf_sp[:n+1]
numci_jk = (whf_numci - whf_sp[:n+1] * numci_sp[:n+1]) / whf_jk
denci_jk = (whf_denci - whf_sp[:n+1] * denci_sp[:n+1]) / whf_jk
numcr_jk = (whf_numcr - whf_sp[:n+1] * numcr_sp[:n+1]) / whf_jk
dencr_jk = (whf_dencr - whf_sp[:n+1] * dencr_sp[:n+1]) / whf_jk

# CISD
eci_jk = numci_jk / denci_jk
eci_jk_mean = np.sum(eci_jk) / (n+1)
eci_jk_err = np.sqrt(np.sum((eci_jk-eci_jk_mean)**2) * n/(n+1))

#CCSD
ecc_jk = (numci_jk + numcr_jk) / (denci_jk + dencr_jk)
# for i in range(len(whf_jk)):
#     print(f'  {i+1}  {ecc_jk.real[i]:.6f}')
ecc_jk_mean = np.sum(ecc_jk) / (n+1)
ecc_jk_err = np.sqrt(np.sum((ecc_jk-ecc_jk_mean)**2) * n/(n+1))
#print(f"  {n+1:4d}  {eci_jk_mean.real:.6f}  {eci_jk_err.real:.6f}  {ecc_jk_mean.real:.6f}  {ecc_jk_err.real:.6f}  {time.time() - init_time:.2f}")


#         # eci, eci_err = \
#         #     stat_utils.blocking_analysis(wci_sp[: n + 1], eci_sp[: n + 1])
#         # ecc, ecc_err = \
#         #     stat_utils.blocking_analysis(wcc_sp[: n + 1], ecc_sp[: n + 1])
#         # if eci_err is not None and ecc_err is not None:
#         #     print(f"  {n+1:4d} \t \t {eci:.6f} \t {eci_err:.6f} \t {ecc:.6f} \t {ecc_err:.6f} \t {time.time() - init_time:.2f}")
#         #     if eci_err < options["max_error"] and ecc_err < options["max_error"]:
#         #         break
#         # else:
#         #     print(f"  {n+1:4d} \t \t {eci:.6f} \t -------- \t {ecc:.6f} \t -------- \t {time.time() - init_time:.2f}")


# nsamples = n + 1
# print(f'# total number of samples {nsamples}')
# whf_sp = whf_sp[:nsamples]
# wci_sp = wci_sp[:nsamples]
# eci_sp = eci_sp[:nsamples]
# num_sp = num_sp[:nsamples]
# don_sp = don_sp[:nsamples]
# ecc_sp = ecc_sp[:nsamples]

# ####### CISD ##########
# samples_clean, idx = stat_utils.reject_outliers(np.stack((wci_sp, eci_sp)).T, 1)
# print(f"# Number of outliers in CISD Trial post: {nsamples - samples_clean.shape[0]} ")

# wci_sp = samples_clean[:, 0]
# eci_sp = samples_clean[:, 1]

# eci, eci_err = stat_utils.blocking_analysis(wci_sp, eci_sp, neql=0, printQ=True)

# if eci_err is not None:
#     eci_err = f"{eci_err:.6f}"
# else:
#     eci_err = f"  {eci_err}  "
# eci = f"{eci:.6f}"

# ####### sto-CCSD ##########
# # samples_clean, idx = stat_utils.reject_outliers(np.stack((wcc_sp, ecc_sp)).T, 1)
# # print(f"# Number of outliers in sto-CCSD Trial post: {nsamples - samples_clean.shape[0]} ")

# # wcc_sp = samples_clean[:, 0]
# # ecc_sp = samples_clean[:, 1]

# # ecc, ecc_err = stat_utils.blocking_analysis(wcc_sp, ecc_sp, neql=0, printQ=True)

# # if ecc_err is not None:
# #     ecc_err = f"{ecc_err:.6f}"
# # else:
# #     ecc_err = f"  {ecc_err}  "
# # ecc = f"{ecc:.6f}"

print(f"Final Results:")
print(f"AFQMC/CISD energy (covariance): {eci.real:.6f} +/- {eci_err.real:.6f}")
print(f"AFQMC/CISD energy (Jackknife): {eci_jk_mean.real:.6f} +/- {eci_jk_err.real:.6f}")
print(f"AFQMC/sto-CCSD energy (covariance): {ecc.real:.6f} +/- {ecc_err.real:.6f}")
print(f"AFQMC/sto-CCSD energy (Jackknife): {ecc_jk_mean.real:.6f} +/- {ecc_jk_err.real:.6f}")
print(f"total run time: {time.time() - init_time:.2f}")
