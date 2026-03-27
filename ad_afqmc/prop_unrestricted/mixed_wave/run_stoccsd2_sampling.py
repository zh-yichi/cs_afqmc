import time
import argparse
import numpy as np
from jax import random
from jax import numpy as jnp
from ad_afqmc import config
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
xtaus, prop_data = trial.get_xtaus(prop_data, wave_data, prop)

prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
prop_data["n_killed_walkers"] = 0
prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
e_init = prop_data["e_estimate"]

oci, eci = trial.calc_energy_cid(prop_data["walkers"], ham_data, wave_data)
numcr, dencr = trial.calc_correction(prop_data["walkers"], xtaus, ham_data, wave_data)
# den_cr = trial.calc_denominator(prop_data["walkers"], xtaus, wave_data)

eci_init = jnp.real(eci)[0]
ecc_init = jnp.real((oci*eci + numcr) / (oci + dencr))[0]
print(f'# initial AFQMC/CISD Energy is {eci_init:.6f}')
print(f'# initial AFQMC/stoCCSD Energy is {ecc_init:.6f}')

# print(f'# Propagating with {options["n_walkers"]} walkers')
# print("# Equilibration sweeps:")
# print("# atom_time  energy_ci  numer_cr  denom_cr  e_stocc  Walltime")
# print(f"  {0.:.2f}  {eci_init.real:.6f}  {numcr[0].real:.6f}  {dencr[0].real:.6f}  {ecc_init.real:.6f}  {time.time() - init_time:.2f}")

# sampler_eq = sampling.sampler_stoccsd2(
#     n_prop_steps=50, 
#     # n_ene_blocks=1, 
#     n_sr_blocks=50, 
#     n_chol = sampler.n_chol
#     )

# block_time = prop.dt * sampler_eq.n_prop_steps * sampler_eq.n_sr_blocks

# for n in range(1,options["n_eql"]+1):
#     prop_data, (whf, numci, denci, numcr, dencr) \
#         = sampler_eq.propagate_phaseless(prop_data, ham_data, prop, trial, wave_data)

#     prop_data = prop.orthonormalize_walkers(prop_data)
#     prop_data = prop.stochastic_reconfiguration_local(prop_data)

#     eci = numci / denci
#     ecc = (numci + numcr) / (denci + dencr)

#     prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * eci

#     print(f" {n*block_time:.2f}  {eci.real:.6f}  {numcr.real:.6f}  {dencr.real:.6f}  {ecc.real:.6f}  {time.time() - init_time:.2f} ")
 
ehf = jnp.real(trial.calc_energy(prop_data["walkers"], ham_data, wave_data))[0]

print(f'# Propagating with {options["n_walkers"]} walkers')
print("# Equilibration sweeps with AFQMC/HF: ")
print("# atom_time \t energy \t Walltime")
print(f"  {0.:.2f} \t \t {ehf:.6f} \t {time.time() - init_time:.2f}")

sampler_eq = sampling.sampler(
    n_prop_steps = 50, 
    n_ene_blocks = 1, 
    n_sr_blocks = 50, 
    n_chol = sampler.n_chol
    )

block_time = prop.dt * sampler_eq.n_prop_steps * sampler_eq.n_ene_blocks * sampler_eq.n_sr_blocks

for n in range(1,options["n_eql"]+1):
    prop_data, (wt, e) = \
        sampler_eq.propagate_phaseless(
            prop_data, ham_data, prop, trial, wave_data)
    
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * e

    print(f"  {n * block_time:.2f} \t {e:.6f} \t {time.time() - init_time:.2f}")

 
print("# Sampling sweeps:")
print("# nBlocks  energy_hf  error  energy_ci  error  energy_cc  error  Walltime")

whf_sp = np.zeros(sampler.n_blocks, dtype="float64")
ehf_sp = np.zeros(sampler.n_blocks,dtype="float64")
numci_sp = np.zeros(sampler.n_blocks, dtype="complex128")
denci_sp = np.zeros(sampler.n_blocks, dtype="complex128")
numcr_sp = np.zeros(sampler.n_blocks, dtype="complex128")
dencr_sp = np.zeros(sampler.n_blocks, dtype="complex128")

for n in range(sampler.n_blocks):
    prop_data, (whf, ehf, numci, denci, numcr, dencr) \
        = sampler.propagate_phaseless(prop_data, ham_data, prop, trial, wave_data)

    whf_sp[n] = whf
    ehf_sp[n] = ehf
    numci_sp[n] = numci
    denci_sp[n] = denci
    numcr_sp[n] = numcr
    dencr_sp[n] = dencr
    eci_estimate = numci / denci
    ecc_estimate = jnp.real((numci + numcr) / (denci + dencr))

    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * ecc_estimate

    if (n+1) % (max(sampler.n_blocks // 10, 1)) == 0 and n > 0:

        weight = np.sum(whf_sp[:n+1])

        whf_ehf = whf_sp[:n+1] * ehf_sp[:n+1]
        whf_numci = whf_sp[:n+1] * numci_sp[:n+1]
        whf_denci = whf_sp[:n+1] * denci_sp[:n+1]
        whf_numcr = whf_sp[:n+1] * numcr_sp[:n+1]
        whf_dencr = whf_sp[:n+1] * dencr_sp[:n+1]

        numci_avg = np.sum(whf_numci) / weight
        denci_avg = np.sum(whf_denci) / weight
        numcr_avg = np.sum(whf_numcr) / weight
        dencr_avg = np.sum(whf_dencr) / weight

        ehf_avg = np.sum(whf_ehf) / weight        
        ehf_err = np.sqrt(np.sum(whf_sp[:n+1] * (ehf_sp[:n+1]-ehf_avg)**2) / weight) / np.sqrt(n)

        eci = (numci_avg / denci_avg).real
        eci_sp = (whf_numci / whf_denci).real
        eci_sp_err = np.std(eci_sp) / np.sqrt(n)

        ecc = ((numci_avg + numcr_avg) / (denci_avg + dencr_avg)).real
        ecc_sp = ((whf_numci + whf_numcr) / (whf_denci + whf_dencr)).real
        ecc_sp_err = np.std(ecc_sp) / np.sqrt(n)

        print(f" {n+1:4d}  {ehf_avg:.6f}  {ehf_err:.6f}  {eci:.6f} {eci_sp_err:.6f} {ecc:.6f} {ecc_sp_err:.6f}  {time.time() - init_time:.2f}")

############################ post sampling ###########################
whf_clean, numci_clean, denci_clean = sampler.filter_outliers(whf_sp, numci_sp, denci_sp, zeta=10)
whf_clean, numcr_clean, dencr_clean = sampler.filter_outliers(whf_sp, numcr_sp, dencr_sp, zeta=10)
 
nsample = len(whf_clean)
whf = np.sum(whf_clean)
numci_avg = np.sum(whf_clean * numci_clean) / whf
denci_avg = np.sum(whf_clean * denci_clean) / whf
numcr_avg = np.sum(whf_clean * numcr_clean) / whf
dencr_avg = np.sum(whf_clean * dencr_clean) / whf

# numci_avg = whf_numci / whf
# denci_avg = whf_denci / whf
# numcr_avg = whf_numcr / whf
# dencr_avg = whf_dencr / whf

# CISD
# whf_clean, numci_clean, denci_clean = sampler.filter_outliers(whf_sp, numci_sp, denci_sp, zeta=5)

# whf_numci = whf_clean * numci_clean
# whf_denci = whf_clean * denci_clean
print(f'# Post Processing AFQMC/CISD')
eci = (numci_avg / denci_avg).real
eci_blk_err = sampler.blk_average(whf_clean, numci_clean, denci_clean, max_size=10)

for i, err in enumerate(eci_blk_err):
    if np.abs((err - eci_blk_err[i-1]) / err) < 0.04:
        break
print(f'# autocorrelation eliminated at blocking {i+1} with estimate error {err:.6f}')
eci_sp_err = err

deci = [1/denci_avg, -numci_avg/denci_avg**2]
covci = np.cov([numci_clean, denci_clean])
eci_cov_err = (np.sqrt(deci @ covci @ deci) / np.sqrt((nsample))).real

# eci_err_jk = sampler.blocking(whf_sp, numci_sp, numcr_sp*0, denci_sp, dencr_sp*0)

#CCSD
# whf_clean, numcc_clean, dencc_clean = sampler.filter_outliers(whf_sp, (numci_sp+numcr_sp), (denci_sp+dencr_sp), zeta=5)

# numcc = (numci_clean + numcr_clean)
# dencc = (denci_clean + dencr_clean)
print(f'# Post Processing AFQMC/stoCCSD')
ecc = ((numci_avg + numcr_avg) / (denci_avg + dencr_avg)).real
ecc_blk_err = sampler.blk_average(whf_clean, (numci_clean + numcr_clean), (denci_clean + dencr_clean), max_size=20)

for i, err in enumerate(ecc_blk_err):
    if np.abs((err - ecc_blk_err[i-1]) / err) < 0.04:
        break
print(f'# autocorrelation eliminated at blocking {i+1} with estimate error {err:.6f}')
ecc_sp_err = err

decc = [1/(denci_avg+dencr_avg), 
        1/(denci_avg+dencr_avg), 
        -(numci_avg+numcr_avg)/(denci_avg+dencr_avg)**2, 
        -(numci_avg+numcr_avg)/(denci_avg+dencr_avg)**2,]

covcc = np.cov([numci_clean, numcr_clean, denci_clean, dencr_clean])
ecc_cov_err = (np.sqrt(decc @ covcc @ decc) / np.sqrt((nsample))).real

# ecc = (numci + numcr) / (denci + dencr)
# ecc_err_jk = sampler.blocking(whf_sp, numci_sp, numcr_sp, denci_sp, dencr_sp)

print(f"Final Results:")
print(f"AFQMC/CISD energy (covariance): {eci:.6f} +/- {eci_cov_err:.6f}")
print(f"AFQMC/CISD energy (dir sample): {eci:.6f} +/- {eci_sp_err:.6f}")
# print(f"AFQMC/CISD energy (Jackknife): {eci.real:.6f} +/- {eci_err_jk.real:.6f}")
print(f"AFQMC/sto-CCSD energy (covariance): {ecc:.6f} +/- {ecc_cov_err:.6f}")
print(f"AFQMC/sto-CCSD energy (dir sample): {ecc:.6f} +/- {ecc_sp_err:.6f}")
# print(f"AFQMC/sto-CCSD energy (Jackknife): {ecc.real:.6f} +/- {ecc_err_jk.real:.6f}")
print(f"total run time: {time.time() - init_time:.2f}")
