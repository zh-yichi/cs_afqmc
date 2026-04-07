import time
import argparse
import numpy as np
from jax import random
from jax import numpy as jnp
from ad_afqmc import config, stat_utils
from ad_afqmc.prop_unrestricted import prep, sampling
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

prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
prop_data["n_killed_walkers"] = 0
prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
e_init = prop_data["e_estimate"]

print(f'# Propagating with {options["n_walkers"]} walkers')
print("# Equilibration sweeps:")
print("# atom_time \t energy \t Walltime")
print(f"    {0.:.2f} \t {e_init:.6f} \t {time.time() - init_time:.2f}")

sampler_eq = sampling.sampler(
    n_prop_steps=50, 
    n_ene_blocks=5, 
    n_sr_blocks=10, 
    n_chol = sampler.n_chol
    )
block_time = prop.dt * sampler_eq.n_prop_steps * sampler_eq.n_ene_blocks * sampler_eq.n_sr_blocks

for n in range(1,options["n_eql"]+1):
    prop_data, (wt, e) \
        = sampler_eq.propagate_phaseless(prop_data, ham_data, prop, trial, wave_data)

    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_local(prop_data)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * e

    print(f"   {n*block_time:.2f} \t {e:.6f} \t {time.time() - init_time:.2f} ")

print("# Sampling sweeps:")
print("#  nblock \t energy \t error \t \t Walltime")

wt_sp = np.zeros(sampler.n_blocks,dtype="float64")
e_sp = np.zeros(sampler.n_blocks,dtype="float64")

for n in range(sampler.n_blocks):
    prop_data, (wt, e) \
        = sampler.propagate_phaseless(prop_data, ham_data, prop, trial, wave_data)
    
    wt_sp[n] = wt
    e_sp[n] = e

    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_local(prop_data)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * e

    if n % (max(sampler.n_blocks // 10, 1)) == 0 and n > 0:
        e, e_err = \
            stat_utils.blocking_analysis(
                wt_sp[: n + 1],
                e_sp[: n + 1],
                )
        if e_err is not None:
            print(f"  {n:4d} \t \t {e:.6f} \t {e_err:.6f} \t {time.time() - init_time:.2f}")
            if e_err < options["max_error"] and n > 20:
                break
        else:
            print(f"  {n:4d} \t \t {e:.6g} \t ------ \t {time.time() - init_time:.2f}")


nsamples = n + 1
print(f'# total number of samples {nsamples}')
wt_sp = wt_sp[:nsamples]
e_sp = e_sp[:nsamples]

samples_clean, idx \
    = stat_utils.reject_outliers(
        np.stack((wt_sp, e_sp)).T, 1)

print(f"# Number of outliers in post: {nsamples - samples_clean.shape[0]}")

wt_sp = samples_clean[:, 0]
e_sp = samples_clean[:, 1]

e, e_err = stat_utils.blocking_analysis(wt_sp, e_sp, neql=0, printQ=True)

if e_err is not None:
    e_err = f"{e_err:.6f}"
else:
    e_err = f"  {e_err}  "
e = f"{e:.6f}"

print(f"Final Results:")
print(f"AFQMC energy: {e} +/- {e_err}")

def blocking_analysis(
        wt_sp, 
        en_sp, 
        min_nblocks=20
        ):
    import numpy as np
    from scipy.optimize import curve_fit
    
    nsample = len(wt_sp)
    max_size = nsample // min_nblocks
    if max_size < 10:
        min_nblocks = max(nsample // 10, 3)
        max_size = nsample // min_nblocks
        print(f"Warning: small dataset, relaxed min_nblocks to {min_nblocks}")
    block_sizes = np.arange(1, max_size + 1)
    block_vars = np.zeros(max_size)
    block_var_errs = np.zeros(max_size)
    block_means = np.zeros(max_size)
    print(f"nsample = {nsample}, max_block_size = {max_size}, min_nblocks = {min_nblocks}")
    print(f"{'B':>4s}  {'NB':>4s}  {'NS':>4s}  {'Energy':>10s}  {'Error':>8s}  {'dError':>8s}")
    for i, block_size in enumerate(block_sizes):
        n_blocks = nsample // block_size
        sl = slice(0, n_blocks * block_size)
        wt = (wt_sp[sl]).reshape(n_blocks, block_size)
        wt_en = (wt_sp[sl] * en_sp[sl]).reshape(n_blocks, block_size)
        block_weight = np.sum(wt, axis=1)
        block_energy = np.sum(wt_en, axis=1) / block_weight
        block_mean = np.mean(block_energy)
        block_var = np.var(block_energy, ddof=1) / n_blocks  # variance of the mean
        block_error = np.sqrt(block_var)
        # Uncertainty on variance: var / sqrt((n_blocks - 1) / 2)
        var_of_var = block_var * np.sqrt(2.0 / (n_blocks - 1))
        err_of_err = block_error / np.sqrt(2.0 * (n_blocks - 1))
        block_means[i] = block_mean
        block_vars[i] = block_var
        block_var_errs[i] = var_of_var
        print(f'{block_size:4d}  {n_blocks:4d}  {block_size*n_blocks:4d}  '
            f'{block_mean:10.6f}  {block_error:8.6f}  {err_of_err:8.6f}')

    def model(x, a, b, tau):
        return a - b * np.exp(-x / tau)
    p0 = [block_vars.max(), block_vars.max() - block_vars[0], 5.0]
    try:
        popt, pcov = curve_fit(model, block_sizes, block_vars,
                            sigma=block_var_errs, absolute_sigma=True,
                            p0=p0, maxfev=10000)
        plateau_var = popt[0]
        plateau_var_unc = np.sqrt(pcov[0, 0])
        plateau_value = np.sqrt(plateau_var)
        # Error propagation: d(sqrt(v)) = dv / (2 sqrt(v))
        plateau_uncertainty = plateau_var_unc / (2.0 * plateau_value)
        tau = popt[2]
        ratio = 0.01 * popt[0] / popt[1]
        if ratio > 0:
            plateau_block_size = int(np.ceil(-popt[2] * np.log(ratio)))
        else:
            plateau_block_size = 1
        plateau_block_size = min(plateau_block_size, max_size)
        print(f"Fit (variance): plateau_var = {plateau_var:.3e} ± {plateau_var_unc:.3e}")
        print(f"Fit (error):    plateau = {plateau_value:.6f} ± {plateau_uncertainty:.6f}")
        print(f"     autocorrelation length ~ {tau:.1f} blocks")
        print(f"     plateau reached at block size ~ {plateau_block_size}")
    except RuntimeError as e:
        print(f"\nFit failed: {e}")
        idx_max = np.argmax(block_vars)
        plateau_value = np.sqrt(block_vars[idx_max])
        plateau_uncertainty = block_var_errs[idx_max] / (2.0 * plateau_value)
        plateau_block_size = block_sizes[idx_max]
        popt, pcov = None, None
        print(f"Fallback max error: {plateau_value:.6f} +/- {plateau_uncertainty:.6f}")
        print(f"     plateau at block size ~ {plateau_block_size}")
    return plateau_value

plateau_value = blocking_analysis(wt_sp, e_sp, min_nblocks=20)
print(f"My blocking AFQMC energy: {e} +/- {plateau_value:.6f}")

print(f"total run time: {time.time() - init_time:.2f}")
