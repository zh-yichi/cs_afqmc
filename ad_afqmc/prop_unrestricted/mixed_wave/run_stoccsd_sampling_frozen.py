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
xtaus, prop_data = trial.get_xtaus(prop_data, wave_data, prop)

prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
prop_data["n_killed_walkers"] = 0
prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]

olp_cc, ene_cc = trial.calc_energy_stoccsd(prop_data["walkers"], xtaus, ham_data, wave_data)
olp_hf = prop_data["overlaps"]
wt_hf = prop_data["weights"]
ecc_init = jnp.real(jnp.sum(wt_hf*olp_cc/olp_hf*ene_cc)/jnp.sum(wt_hf*olp_cc/olp_hf))
ehf_init = jnp.real(prop_data["e_estimate"])

print(f'# initial AFQMC/stoCCSD Energy is {ecc_init:.6f}')
print(f'# Propagating with {options["n_walkers"]} walkers')

print("# Equilibration sweeps with AFQMC/HF: ")
print("# atom_time \t energy \t Walltime")
print(f"  {0.:.2f} \t \t {ehf_init:.6f} \t {time.time() - init_time:.2f}")

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

print("# Sampling sweeps With Forzen Walkers from Equilibration:")
print("# nBlocks   energy/hf   error   energy/cc   error   energy/cc(abs)   error   Sign_Real  Sign_Imag  cc/hf_olp  Walltime")

whf_sp = np.zeros(sampler.n_blocks, dtype="float64")
ehf_sp = np.zeros(sampler.n_blocks, dtype="float64")
ecc_avg_sp = np.zeros(sampler.n_blocks, dtype="complex128")
ecc_abs_sp = np.zeros(sampler.n_blocks, dtype="complex128")
occ_avg_sp = np.zeros(sampler.n_blocks, dtype="complex128")
occ_abs_sp = np.zeros(sampler.n_blocks, dtype="float64")

for n in range(sampler.n_blocks):
    prop_data, (whf, ehf, ecc_avg, ecc_abs, occ_avg, occ_abs) \
        = sampler._block_froze(prop_data, ham_data, prop, trial, wave_data)

    whf_sp[n] = whf
    ehf_sp[n] = ehf
    ecc_avg_sp[n] = ecc_avg
    ecc_abs_sp[n] = ecc_abs
    occ_avg_sp[n] = occ_avg
    occ_abs_sp[n] = occ_abs
    ecc_estimate = jnp.real(ecc_avg / occ_avg)
    # ecc_estimate_abs = jnp.real(ecc_abs / occ_abs)

    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * ecc_estimate

    if (n+1) % (max(sampler.n_blocks // 10, 1)) == 0 and n > 0:

        weight = np.sum(whf_sp[:n+1])
        whf_ehf = np.sum(whf_sp[:n+1] * ehf_sp[:n+1])
        
        whf_ecc_avg = np.sum(whf_sp[:n+1] * ecc_avg_sp[:n+1])
        whf_occ_avg = np.sum(whf_sp[:n+1] * occ_avg_sp[:n+1])
        whf_ecc_abs = np.sum(whf_sp[:n+1] * ecc_abs_sp[:n+1])
        whf_occ_abs = np.sum(whf_sp[:n+1] * occ_abs_sp[:n+1])

        blk_ehf_avg = np.real(whf_ehf / weight)
        blk_ehf_err = np.sqrt(np.sum(whf_sp[:n+1] * (ehf_sp[:n+1]-blk_ehf_avg)**2) / weight) / np.sqrt(n)

        blk_ecc_avg = np.real(whf_ecc_avg / whf_occ_avg)
        blk_ecc_abs = np.real(whf_ecc_abs / whf_occ_abs)
        blk_ecc_avg_sp = np.real((whf_sp[:n+1] * ecc_avg_sp[:n+1]) / (whf_sp[:n+1] * occ_avg_sp[:n+1]))
        blk_ecc_avg_err = np.real(np.std(blk_ecc_avg_sp) / np.sqrt(n))
        blk_ecc_abs_sp = np.real((whf_sp[:n+1] * ecc_abs_sp[:n+1]) / (whf_sp[:n+1] * occ_abs_sp[:n+1]))
        blk_ecc_abs_err = np.real(np.std(blk_ecc_abs_sp) / np.sqrt(n))

        blk_occ_avg = np.sum(whf_sp[:n+1] * occ_avg_sp[:n+1]) / weight
        blk_occ_abs = np.sum(whf_sp[:n+1] * occ_abs_sp[:n+1]) / weight
        sign = blk_occ_avg / blk_occ_abs
        sign_real = jnp.real(sign)
        sign_imag = jnp.imag(sign)

        print(f" {n+1:4d}  {blk_ehf_avg:.6f}  {blk_ehf_err:.6f}  {blk_ecc_avg:.6f}  {blk_ecc_avg_err:.6f}  {blk_ecc_abs:.6f}  {blk_ecc_abs_err:.6f}  {sign_real:.6f}  {sign_imag:.6f}  {blk_occ_avg.real:.6f}  {time.time() - init_time:.2f}")

print('# Post Propagation')

ecc_avg_raw = np.real(whf_ecc_avg / whf_occ_avg)
ecc_avg_err_raw = sampler.blk_average(whf_sp, ecc_avg_sp, occ_avg_sp, max_size=None)

find_err = False
for i, avg_err in enumerate(ecc_avg_err_raw):
    if np.abs((avg_err - ecc_avg_err_raw[i-1]) / avg_err) < 0.03:
        find_err = True
        print(f'# autocorrelation eliminated at blocking {i+1} with err {avg_err:.6f}')
        break

if not find_err: 
    avg_err = ecc_avg_err_raw.max()
    print(f'# not find error plateu during blocking (more samples recommended), use maxium error')


ecc_abs_raw = np.real(whf_ecc_abs / whf_occ_abs)
ecc_abs_err_raw = sampler.blk_average(whf_sp, ecc_abs_sp, occ_abs_sp, max_size=None)

find_err = False
for i, abs_err in enumerate(ecc_abs_err_raw):
    if np.abs((abs_err - ecc_abs_err_raw[i-1]) / abs_err) < 0.03:
        find_err = True
        print(f'# autocorrelation eliminated at blocking {i+1} with err {abs_err:.6f}')
        break

if not find_err:
    abs_err = ecc_abs_err_raw.max()
    print(f'# not find error plateu during blocking (more samples recommended), use maxium error')

occ_avg = np.sum(whf_sp * occ_avg_sp) / weight
occ_abs = np.sum(whf_sp * occ_abs_sp) / weight

occ_err = np.real(np.sqrt(np.sum(whf_sp * (occ_avg_sp - occ_avg)**2) / weight) / np.sqrt(len(whf_sp)))
sign = occ_avg / occ_abs
sign_real = jnp.real(sign)
sign_imag = jnp.imag(sign)

print(f'# Raw AFQMC/stoCCSD      Energy: {ecc_avg_raw:.6f} +/- {avg_err:.6f}')
print(f'# Raw AFQMC/stoCCSD(abs) Energy: {ecc_abs_raw:.6f} +/- {abs_err:.6f}')
print(f'# Raw AFQMC/stoCCSD Sign: {sign_real:.6f} + i{sign_imag:.6f}')
print(f'# Raw AFQMC/stoCCSD overlap: {occ_avg.real:.6f} +/- {occ_err:.6f}')

# whf_clean, num_clean, den_clean = sampler.filter_outliers(whf_sp, num_sp, den_sp, zeta=10)
# print(f'removed outliers {len(whf_sp) - len(whf_clean)}')

# whf = np.sum(whf_clean)
# whf_num = np.sum(whf_clean * num_clean)
# whf_den = np.sum(whf_clean * den_clean)

# ecc_clean = np.real(whf_num / whf_den)
# ecc_err_clean = sampler.blk_average(whf_clean, num_clean, den_clean, max_size=10)

# find_err = False
# for i, err in enumerate(ecc_err_clean):
#     if np.abs((err - ecc_err_clean[i-1]) / err) < 0.04:
#         find_err = True
#         print(f'# autocorrelation eliminated at blocking {i+1} with err {err:.6f}')
#         break

# if not find_err: 
#     err = ecc_err_clean.max()
#     print(f'# not find error plateu during blocking (more samples recommended), use maxium error')

# print(f'# Clean AFQMC/stoCCSD Energy: {ecc_clean:.6f} +/- {err:.6f}')
print(f"total run time: {time.time() - init_time:.2f}")
