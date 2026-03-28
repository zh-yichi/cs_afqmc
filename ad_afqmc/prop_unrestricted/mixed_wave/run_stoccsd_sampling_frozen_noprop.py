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

# print("# Equilibration sweeps with AFQMC/HF: ")
# print("# atom_time \t energy \t Walltime")
# print(f"  {0.:.2f} \t \t {ehf_init:.6f} \t {time.time() - init_time:.2f}")

# sampler_eq = sampling.sampler(
#     n_prop_steps = 10, 
#     n_ene_blocks = 1, 
#     n_sr_blocks = 1, 
#     n_chol = sampler.n_chol
#     )

# block_time = prop.dt * sampler_eq.n_prop_steps * sampler_eq.n_ene_blocks * sampler_eq.n_sr_blocks

# for n in range(1,options["n_eql"]+1):
#     prop_data, (wt, e) = \
#         sampler_eq.propagate_phaseless(
#             prop_data, ham_data, prop, trial, wave_data)
    
#     prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * e 

#     print(f"  {n * block_time:.2f} \t {e:.6f} \t {time.time() - init_time:.2f}")

print("# Sampling sweeps With Forzen Walkers from Equilibration:")
print("# nBlocks  energy/hf  error  energy/cc  error  Sign_Real  Sign_Imag  Walltime")

whf_sp = np.zeros(sampler.n_blocks, dtype="float64")
ehf_sp = np.zeros(sampler.n_blocks, dtype="float64")
ecc_num_sp = np.zeros(sampler.n_blocks, dtype="complex128")
occ_den_sp = np.zeros(sampler.n_blocks, dtype="complex128")
occ_abs_sp = np.zeros(sampler.n_blocks, dtype="float64")

for n in range(sampler.n_blocks):
    prop_data, (whf, ehf, ecc_num, occ_den, occ_abs) \
        = sampler._block_froze(prop_data, ham_data, prop, trial, wave_data)

    whf_sp[n] = whf
    ehf_sp[n] = ehf
    ecc_num_sp[n] = ecc_num
    occ_den_sp[n] = occ_den
    occ_abs_sp[n] = occ_abs
    ecc_estimate = jnp.real(ecc_num / occ_den)

    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * ecc_estimate

    if (n+1) % (max(sampler.n_blocks // 10, 1)) == 0 and n > 0:

        weight = np.sum(whf_sp[:n+1])
        whf_ehf = np.sum(whf_sp[:n+1] * ehf_sp[:n+1])
        whf_ecc_num = np.sum(whf_sp[:n+1] * ecc_num_sp[:n+1])
        whf_occ_den = np.sum(whf_sp[:n+1] * occ_den_sp[:n+1])

        ehf_avg = np.real(whf_ehf / weight)
        ehf_err = np.sqrt(np.sum(whf_sp[:n+1] * (ehf_sp[:n+1]-ehf_avg)**2) / weight) / np.sqrt(n)

        ecc_avg = np.real(whf_ecc_num / whf_occ_den)
        ecc_sp = np.real((whf_sp[:n+1] * ecc_num_sp[:n+1]) / (whf_sp[:n+1] * occ_den_sp[:n+1]))
        ecc_err = np.real(np.std(ecc_sp) / np.sqrt(n))

        occ_avg = np.sum(whf_sp[:n+1] * occ_den_sp[:n+1]) / weight
        occ_abs_avg = np.sum(whf_sp[:n+1] * occ_abs_sp[:n+1]) / weight
        sign = occ_avg / occ_abs_avg
        sign_real = jnp.real(sign)
        sign_imag = jnp.imag(sign)

        print(f" {n+1:4d}  {ehf_avg:.6f}  {ehf_err:.6f}  {ecc_avg:.6f}  {ecc_err:.6f}  {sign_real:.6f}  {sign_imag:.6f}  {time.time() - init_time:.2f}")

print('# Post Propagation')

ecc_raw = np.real(whf_ecc_num / whf_occ_den)
ecc_err_raw = sampler.blk_average(whf_sp, ecc_num_sp, occ_den_sp, max_size=None)

find_err = False
for i, err in enumerate(ecc_err_raw):
    if np.abs((err - ecc_err_raw[i-1]) / err) < 0.03:
        find_err = True
        print(f'# autocorrelation eliminated at blocking {i+1} with err {err:.6f}')
        break

if not find_err: 
    err = ecc_err_raw.max()
    print(f'# not find error plateu during blocking (more samples recommended), use maxium error')

occ_avg = np.sum(whf_sp * occ_den_sp) / weight
occ_abs_avg = np.sum(whf_sp * occ_abs_sp) / weight
sign = occ_avg / occ_abs_avg
sign_real = jnp.real(sign)
sign_imag = jnp.imag(sign)

print(f'# Raw AFQMC/stoCCSD Energy: {ecc_raw:.6f} +/- {err:.6f}')
print(f'# Raw AFQMC/stoCCSD Sign: {sign_real:.6f} + i{sign_imag:.6f}')

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
