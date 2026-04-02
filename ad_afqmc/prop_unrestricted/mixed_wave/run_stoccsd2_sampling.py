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

print(f'Sampler is {sampler}')
print(f'Trial is {trial}')

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
print(f'Initial AFQMC/CISD Energy is {eci_init:.6f}')
print(f'Initial AFQMC/stoCCSD Energy is {ecc_init:.6f}')
 
sampler_eq = sampling.sampler(
    n_prop_steps = 50, 
    n_ene_blocks = 1, 
    n_sr_blocks = 50, 
    n_chol = sampler.n_chol
    )

block_time = prop.dt * sampler_eq.n_prop_steps * sampler_eq.n_ene_blocks * sampler_eq.n_sr_blocks
print(f'Propagating with {options["n_walkers"]} walkers')
print(f"---------------- Equilibration sweeps -----------------")
print(f"{'inv_T':>5s}  {'energy':>10s}  {'runTime':>8s}")
print(f"{0.:5.2f}  {e_init:10.6f}  {time.time() - init_time:8.2f}")

for n in range(1,options["n_eql"]+1):
    prop_data, (wt, e) = \
        sampler_eq.propagate_phaseless(
            prop_data, ham_data, prop, trial, wave_data)
    
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * e
    print(f"{n*block_time:5.2f}  {e:10.6f}  {time.time() - init_time:8.2f}")

print("------------------- Sampling sweeps ---------------------")
print(f"{'NB':>5s}  "
      f"{'energy/HF':>10s}  {'error':>8s}  "
      f"{'energy/CI':>10s}  {'error':>8s}  "
      f"{'energy/CC':>10s}  {'error':>8s}  "
      f"{'o[CI/HF]':>10s}  {'error':>8s}  "
      f"{'o[CC/HF]':>10s}  {'error':>8s}  "
      f"{'runTime':>8s}")

# print("# nBlocks  energy/hf  error  energy/ci  error  energy/cc  error  Walltime")

whf_sp = np.zeros(sampler.n_blocks, dtype="float64")
ehf_sp = np.zeros(sampler.n_blocks,dtype="float64")
numci_sp = np.zeros(sampler.n_blocks, dtype="complex128")
denci_sp = np.zeros(sampler.n_blocks, dtype="complex128")
numcr_sp = np.zeros(sampler.n_blocks, dtype="complex128")
dencr_sp = np.zeros(sampler.n_blocks, dtype="complex128")
ecc_sp = np.zeros(sampler.n_blocks,dtype="float64")

for n in range(sampler.n_blocks):
    prop_data, (whf, ehf, numci, denci, numcr, dencr) \
        = sampler.propagate_phaseless(prop_data, ham_data, prop, trial, wave_data)

    whf_sp[n] = whf
    ehf_sp[n] = ehf
    numci_sp[n] = numci
    denci_sp[n] = denci
    numcr_sp[n] = numcr
    dencr_sp[n] = dencr

    ecc_estimate = jnp.real((numci + numcr) / (denci + dencr))
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * ecc_estimate
    ecc_sp[n] = ecc_estimate

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

        # denci_avg = denci_avg.real
        denci_err = np.sqrt(np.sum(whf_sp[:n+1] * (denci_sp[:n+1] - denci_avg)**2) / weight / n).real

        dencc_avg = (denci_avg + dencr_avg).real
        dencc_err = np.sqrt(np.sum(whf_sp[:n+1] * (denci_sp[:n+1] + dencr_sp[:n+1] - dencc_avg)**2) / weight / n).real

        ehf_avg = np.sum(whf_ehf) / weight        
        ehf_err = np.sqrt(np.sum(whf_sp[:n+1] * (ehf_sp[:n+1]-ehf_avg)**2) / weight) / np.sqrt(n)

        eci_avg = (numci_avg / denci_avg).real
        ecis = (whf_numci / whf_denci).real
        eci_err = np.std(ecis) / np.sqrt(n)

        ecc_avg = ((numci_avg + numcr_avg) / (denci_avg + dencr_avg)).real
        eccs = ((whf_numci + whf_numcr) / (whf_denci + whf_dencr)).real
        ecc_err = np.std(eccs) / np.sqrt(n)

        print(f"{n+1:5d}  "
              f"{ehf_avg:10.6f}  {ehf_err:8.6f}  "
              f"{eci_avg:10.6f}  {eci_err:8.6f}  "
              f"{ecc_avg:10.6f}  {ecc_err:8.6f}  "
              f"{denci_avg.real:10.6f}  {denci_err:8.6f}  "
              f"{dencc_avg:10.6f}  {dencc_err:8.6f}  "
              f"{time.time() - init_time:8.2f}")
        # print(f" {n+1:4d}  {ehf_avg:.6f}  {ehf_err:.6f}  {eci:.6f} {eci_sp_err:.6f} {ecc:.6f} {ecc_sp_err:.6f}  {time.time() - init_time:.2f}")

############################ post sampling ###########################
print('------------- Post Propagation -----------------')
from scipy.optimize import curve_fit
# block_sizes = np.arange(1, len(block_errs) + 1)

# Model: error(x) = A - B * exp(-x / tau)
def model(x, a, b, tau):
    return a - b * np.exp(-x / tau)

def filter_outliers(refs, zeta=10):

    median = np.median(refs)
    mad = 1.4826 * np.median(np.abs(refs - median))
    bound = zeta * mad
    mask = np.abs(refs - median) < bound
    
    print(f"Outlier energy bound [{median-bound:.6f}, {median+bound:.6f}]")
    
    return mask

mask = filter_outliers(ecc_sp, zeta=20)
# whf_clean, numci_clean, denci_clean = sampler.filter_outliers(whf_sp, numci_sp, denci_sp, zeta=10)
# whf_clean, numcr_clean, dencr_clean = sampler.filter_outliers(whf_sp, numcr_sp, dencr_sp, zeta=10)

whf_clean = whf_sp[mask]
numci_clean = numci_sp[mask]
denci_clean = denci_sp[mask]
numcr_clean = numcr_sp[mask]
dencr_clean = dencr_sp[mask]

nclean = len(whf_clean)
nsample = len(whf_sp)
print(f"Removed {nsample-nclean} outliers with energies {ecc_sp[~mask]}")

nsample = len(whf_clean)
whf = np.sum(whf_clean)
numci_avg = np.sum(whf_clean * numci_clean) / whf
denci_avg = np.sum(whf_clean * denci_clean) / whf
numcr_avg = np.sum(whf_clean * numcr_clean) / whf
dencr_avg = np.sum(whf_clean * dencr_clean) / whf

# CISD
print('------------- Processing AFQMC/CISD(HF) -----------------')
eci_avg = (numci_avg / denci_avg).real
block_errs = sampler.blk_average(whf_clean, numci_clean, denci_clean, max_size=20)

block_sizes = np.arange(1, len(block_errs) + 1)
p0 = [block_errs[-1], block_errs[-1] - block_errs[0], 5.0]
popt, pcov = curve_fit(model, block_sizes, block_errs, p0=p0, maxfev=10000)
plateau_value = popt[0]
perr = np.sqrt(np.diag(pcov))

print(f"Plateau error estimate: {plateau_value:.6f} ± {perr[0]:.6f}")
print(f"Decay constant (tau):   {popt[2]:.2f} blocks")
convergence_block = -popt[2] * np.log(0.05)
print(f"~95% of plateau reached at block_size ≈ {convergence_block:.0f}")
if convergence_block > nclean or convergence_block < 0:
    print(f"Plateau not reached within sampled blocks, use max error")
    plateau_value = block_errs.max()
print(f"Blocked clean AFQMC/CISD(HF) energy: {eci_avg:.6f} ± {plateau_value:.6f}")

deci = [1/denci_avg, -numci_avg/denci_avg**2]
covci = np.cov([numci_clean, denci_clean])
eci_cov_err = (np.sqrt(deci @ covci @ deci) / np.sqrt((nsample))).real


#CCSD
print('------------- Processing AFQMC/stoDiffCCSD -----------------')
ecc_avg = ((numci_avg + numcr_avg) / (denci_avg + dencr_avg)).real
block_errs = sampler.blk_average(whf_clean, (numci_clean + numcr_clean), (denci_clean + dencr_clean), max_size=20)

block_sizes = np.arange(1, len(block_errs) + 1)
p0 = [block_errs[-1], block_errs[-1] - block_errs[0], 5.0]
popt, pcov = curve_fit(model, block_sizes, block_errs, p0=p0, maxfev=10000)
plateau_value = popt[0]
perr = np.sqrt(np.diag(pcov))

print(f"Plateau error estimate: {plateau_value:.6f} ± {perr[0]:.6f}")
print(f"Decay constant (tau):   {popt[2]:.2f} blocks")
convergence_block = -popt[2] * np.log(0.05)
print(f"~95% of plateau reached at block_size ≈ {convergence_block:.0f}")
if convergence_block > nclean or convergence_block < 0:
    print(f"Plateau not reached within sampled blocks, use max error")
    plateau_value = block_errs.max()
print(f"Blocked clean AFQMC/stoDiffCCSD energy: {ecc_avg:.6f} ± {plateau_value:.6f}")

# decc = [1/(denci_avg+dencr_avg), 
#         1/(denci_avg+dencr_avg), 
#         -(numci_avg+numcr_avg)/(denci_avg+dencr_avg)**2, 
#         -(numci_avg+numcr_avg)/(denci_avg+dencr_avg)**2,]

# covcc = np.cov([numci_clean, numcr_clean, denci_clean, dencr_clean])
# ecc_cov_err = (np.sqrt(decc @ covcc @ decc) / np.sqrt((nsample))).real

# ecc = (numci + numcr) / (denci + dencr)
# ecc_err_jk = sampler.blocking(whf_sp, numci_sp, numcr_sp, denci_sp, dencr_sp)

# print(f"Final Results:")
# print(f"AFQMC/CISD energy (covariance): {eci:.6f} +/- {eci_cov_err:.6f}")
# print(f"AFQMC/CISD energy (dir sample): {eci:.6f} +/- {eci_sp_err:.6f}")
# print(f"AFQMC/CISD energy (Jackknife): {eci.real:.6f} +/- {eci_err_jk.real:.6f}")
# print(f"AFQMC/sto-CCSD energy (covariance): {ecc:.6f} +/- {ecc_cov_err:.6f}")
# print(f"AFQMC/sto-CCSD energy (dir sample): {ecc:.6f} +/- {ecc_sp_err:.6f}")
# print(f"AFQMC/sto-CCSD energy (Jackknife): {ecc.real:.6f} +/- {ecc_err_jk.real:.6f}")

print(f"total run time: {time.time() - init_time:.2f}")
