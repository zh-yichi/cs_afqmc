import numpy as np
from jax import random
from jax import numpy as jnp
from functools import partial
from ad_afqmc import config
from ad_afqmc.prop_unrestricted import prep, sampling
import time
import argparse

import jax
jax.config.update("jax_enable_x64", True)

init_time = time.time()
print = partial(print, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    if args.use_gpu:
        config.afqmc_config["use_gpu"] = True
        
config.setup_jax()

ham_data, ham, prop, trial, wave_data, sampler, options = (prep._prep_afqmc())

### initialize propagation
seed = options["seed"]
init_walkers = None
trial_rdm1 = trial.get_rdm1(wave_data)
if "rdm1" not in wave_data:
    wave_data["rdm1"] = trial_rdm1
ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
ham_data = ham.build_propagation_intermediates(ham_data, prop, trial, wave_data)
h0 = ham_data['h0']

prop_data = prop.init_prop_data(trial, wave_data, ham_data, init_walkers)
if jnp.abs(jnp.sum(prop_data["overlaps"])) < 1.0e-6:
    raise ValueError(
        "Initial overlaps are zero. Pass walkers with non-zero overlap."
    )
prop_data["key"] = random.PRNGKey(options["seed"])

prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
prop_data["n_killed_walkers"] = 0 
t, e0, et = trial.calc_energy_pt(
    prop_data['walkers'], ham_data, wave_data)

ept = jnp.real(e0 + et - t*(e0-h0))
# ept = jnp.array(jnp.sum(ept_sp) / prop.n_walkers)
prop_data["e_estimate"] = ept[0]
prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]

print(f'# Propagating with {options["n_walkers"]} walkers')
print(f'# initial AFQMC/ptCCSD Energy is {ept[0]:.6f}')
print("# Equilibration sweeps with AFQMC/HF: ")
print("#   Iter \t Energy_HF \t Walltime")
print(f"  {0:5d} \t {e0[0].real:.6f} \t {time.time() - init_time:.2f}")

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
    
    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_local(prop_data)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * e.real

    print(f"  {n * block_time:.2f} \t {e.real:.6f} \t {time.time() - init_time:.2f}")


print("#\n# Sampling sweeps:")
print("#  Iter \t Energy_PT \t error \t \t Walltime")

wt_sp = np.zeros(sampler.n_blocks, dtype="float64")
t_sp = np.zeros(sampler.n_blocks, dtype="complex128") # dtype="float64") # dtype="complex128")
e0_sp = np.zeros(sampler.n_blocks, dtype="complex128") #dtype="float64") # dtype="complex128")
et_sp = np.zeros(sampler.n_blocks, dtype="complex128") #dtype="float64") # dtype="complex128")
ept_sp = np.zeros(sampler.n_blocks, dtype="float64")
    
for n in range(sampler.n_blocks):
    prop_data, (wt, t, e0, et) =\
        sampler.propagate_phaseless(prop_data, ham_data, prop, trial, wave_data)
    
    wt_sp[n] = wt
    t_sp[n] = t
    e0_sp[n] = e0
    et_sp[n] = et

    ept = (e0 + et - t * (e0 - h0)).real
    ept_sp[n] = ept

    # prop_data = prop.orthonormalize_walkers(prop_data)
    # prop_data = prop.stochastic_reconfiguration_local(prop_data)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * ept

    if (n+1) % (max(sampler.n_blocks // 10, 1)) == 0 and n > 0:
        t = np.sum(wt_sp[:n+1] * t_sp[:n+1]) / np.sum(wt_sp[:n+1])
        e0 = np.sum(wt_sp[:n+1] * e0_sp[:n+1]) / np.sum(wt_sp[:n+1])
        et = np.sum(wt_sp[:n+1] * et_sp[:n+1]) / np.sum(wt_sp[:n+1])

        ept = (e0 + et- t*(e0-h0)).real

        # covariant error (pE/pt,pE/pe0,pE/pet)
        dE = np.array([-e0+h0,1-t,1])
        cov_te0et = np.cov([t_sp[:n+1], e0_sp[:n+1], et_sp[:n+1]])
        ept_cov_err = (np.sqrt(dE @ cov_te0et @ dE) / np.sqrt((n+1))).real
        # print(f"  {n:4d} \t \t {ept:.6f} \t {ept_cov_err:.6f} \t {time.time() - init_time:.2f}")

        # sample error
        ept_sp_err = (np.std(ept_sp[:n+1]) / np.sqrt(len(ept_sp[:n+1]))).real

        print(f"  {n+1:4d} \t \t {ept:.6f} \t {ept_cov_err:.6f} \t {ept_sp_err:.6f} \t {time.time() - init_time:.2f}")
        # if ept_cov_err < options["max_error"] and ept_sp_err < options["max_error"] and n > 20:
        #     break


print(f"# Final Results:")
nsamples = np.count_nonzero(wt_sp)
print(f'# total number of samples {nsamples}')
wt_sp = wt_sp[:nsamples]
t_sp = t_sp[:nsamples]
e0_sp = e0_sp[:nsamples]
et_sp = et_sp[:nsamples]
ept_sp = ept_sp[:nsamples]

wt = np.sum(wt_sp)
t = np.sum(wt_sp * t_sp) / wt
e0 = np.sum(wt_sp * e0_sp) / wt
et = np.sum(wt_sp * et_sp) / wt

ept = (e0 + et - t*(e0-h0)).real

dE = np.array([-e0+h0, 1-t, 1])
cov_te0et = np.cov([t_sp, e0_sp, et_sp])
ept_cov_err = (np.sqrt(dE @ cov_te0et @ dE)/np.sqrt(nsamples)).real

ept_sp_err = np.std(ept_sp) / np.sqrt(nsamples)

print(f"# Raw AFQMC/ptCCSD energy (covariance): {ept:.6f} +/- {ept_cov_err:.6f}")
print(f"# Raw AFQMC/ptCCSD energy (dir sample): {ept:.6f} +/- {ept_sp_err:.6f}")

print(f'# Clean Samples')
d = np.abs(ept_sp-np.median(ept_sp))
d_med = np.median(d) + 1e-7
z = d/d_med
mask = z < 30
print(f'# the outliers zeta {z[~mask]} | energy {ept_sp[~mask]}')

wt_clean = wt_sp[mask]
t_clean = t_sp[mask]
e0_clean = e0_sp[mask]
et_clean = et_sp[mask]
ept_clean = ept_sp[mask]

nclean = len(wt_clean)
print('# remove outliers in direct sampling ', nsamples-nclean)

wt = np.sum(wt_clean)
t = np.sum(wt_clean * t_clean) / wt
e0 = np.sum(wt_clean * e0_clean) / wt
et = np.sum(wt_clean * et_clean) / wt

ept = (e0 + et - t*(e0-h0)).real

dE = np.array([-e0+h0, 1-t, 1])
cov_te0et = np.cov([t_clean, e0_clean, et_clean])
ept_cov_err = (np.sqrt(dE @ cov_te0et @ dE)/np.sqrt(nclean)).real

ept_sp_err = np.std(ept_clean) / np.sqrt(nclean)

print(f"# clean AFQMC/ptCCSD energy (covariance): {ept:.6f} +/- {ept_cov_err:.6f}")
print(f"# clean AFQMC/ptCCSD energy (dir sample): {ept:.6f} +/- {ept_sp_err:.6f}")

print('# performing blocking analysis')
max_size = nclean // 10
block_errs = np.zeros(max_size)
print('# Blk_SZ  NBlk  NSmp  Energy  Error')
for i, block_size in enumerate(range(1,max_size+1)):
    n_blocks = nclean // block_size

    wt_truncated = wt_clean[:n_blocks * block_size]
    t_truncated = t_clean[:n_blocks * block_size]
    e0_truncated = e0_clean[:n_blocks * block_size]
    et_truncated = et_clean[:n_blocks * block_size]

    wt_t = wt_truncated * t_truncated
    wt_e0 = wt_truncated * e0_truncated
    wt_et = wt_truncated * et_truncated

    wt = wt_truncated.reshape(n_blocks, block_size)
    wt_t = wt_t.reshape(n_blocks, block_size)
    wt_e0 = wt_e0.reshape(n_blocks, block_size)
    wt_et = wt_et.reshape(n_blocks, block_size)

    block_wt = np.sum(wt, axis=1)
    block_t = np.sum(wt_t, axis=1) / block_wt
    block_e0 = np.sum(wt_e0, axis=1) / block_wt
    block_et = np.sum(wt_et, axis=1) / block_wt

    block_energy = (block_e0 + block_et - block_t*(block_e0-h0)).real
    block_mean = np.mean(block_energy)
    block_error = np.std(block_energy, ddof=1) / np.sqrt(n_blocks)
    print(f' {block_size:3d}  {n_blocks:3d}  {block_size*n_blocks:4d}  {block_mean:.6f}  {block_error:.6f}')
    block_errs[i] = block_error

for i, err in enumerate(block_errs):
    if np.abs((err - block_errs[i-1]) / err) < 0.04:
        break
print(f'# autocorrelation eliminated at blocking {i+1} with estimate error {err:.6f}')

print(f"# Blocked clean AFQMC/ptCCSD energy (dir sample): {ept:.6f} +/- {err:.6f}")
print(f"# total run time: {time.time() - init_time:.2f}")

