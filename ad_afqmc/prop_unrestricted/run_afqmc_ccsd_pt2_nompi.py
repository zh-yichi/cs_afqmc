import time
import argparse
import numpy as np
from jax import random
from jax import numpy as jnp
from ad_afqmc import config
from ad_afqmc.prop_unrestricted import prep, sampling
from functools import partial
import jax
jax.config.update("jax_enable_x64", True)

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
h0 = ham_data['h0']

prop_data = prop.init_prop_data(trial, wave_data, ham_data, init_walkers)
if jnp.abs(jnp.sum(prop_data["overlaps"])) < 1.0e-6:
    raise ValueError(
        "Initial overlaps are zero. Pass walkers with non-zero overlap."
    )

prop_data["key"] = random.PRNGKey(options["seed"])
prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
prop_data["n_killed_walkers"] = 0

t1, t2, e0, e1 = trial.calc_energy_pt(prop_data["walkers"], ham_data, wave_data)
ept_sp = h0 + e0/t1 + e1/t1 - t2 * e0 / t1**2
ept = jnp.real(jnp.sum(ept_sp) / prop.n_walkers)
prop_data["e_estimate"] = ept
prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
print(f'# initial AFQMC/pt2CCSD Energy is {ept:.6f}')

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
    
    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_local(prop_data)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * e

    print(f"  {n * block_time:.2f} \t {e:.6f} \t {time.time() - init_time:.2f}")

print("# Sampling sweeps:")
print("# blocks \t energy \t error \t \t Walltime")

wt_sp = np.zeros(sampler.n_blocks, dtype="float64")
t1_sp = np.zeros(sampler.n_blocks, dtype="complex128")# dtype="float64")
t2_sp = np.zeros(sampler.n_blocks, dtype="complex128")# dtype="float64")
e0_sp = np.zeros(sampler.n_blocks, dtype="complex128")# dtype="float64")
e1_sp = np.zeros(sampler.n_blocks, dtype="complex128")# dtype="float64")
ept_sp = np.zeros(sampler.n_blocks, dtype="float64")

for n in range(sampler.n_blocks):
    prop_data, (wt, t1, t2, e0, e1) =\
        sampler.propagate_phaseless(prop_data, ham_data, prop, trial, wave_data)
    
    wt_sp[n] = wt
    t1_sp[n] = t1
    t2_sp[n] = t2
    e0_sp[n] = e0
    e1_sp[n] = e1

    ept = (h0 + 1/t1*e0 + 1/t1*e1 - 1/t1**2 * t2 * e0).real
    ept_sp[n] = ept

    # prop_data = prop.orthonormalize_walkers(prop_data)
    # prop_data = prop.stochastic_reconfiguration_local(prop_data)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * ept

    if (n+1) % (max(sampler.n_blocks // 10, 1)) == 0 and n > 0:
        wt = np.sum(wt_sp[:n+1])
        t1 = np.sum(wt_sp[:n+1] * t1_sp[:n+1]) / wt
        t2 = np.sum(wt_sp[:n+1] * t2_sp[:n+1]) / wt
        e0 = np.sum(wt_sp[:n+1] * e0_sp[:n+1]) / wt
        e1 = np.sum(wt_sp[:n+1] * e1_sp[:n+1]) / wt

        ept = (h0 + 1/t1 * e0 + 1/t1 * e1 - 1/t1**2 * t2 * e0).real
        # (pE/pt1,pE/pt2,pE/pe0,pE/pe1)
        dE = np.array([-1/t1**2 * (e0+e1) + 2/t1**3 * t2 * e0,
                       -1/t1**2 * e0,
                        1/t1 - 1/t1**2 * t2,
                        1/t1])
        cov_te0e1 = np.cov([t1_sp[:n+1],
                            t2_sp[:n+1],
                            e0_sp[:n+1],
                            e1_sp[:n+1]]
                            )
        ept_err = (np.sqrt(dE @ cov_te0e1 @ dE)/np.sqrt((n+1))).real
        print(f"  {n+1:4d} \t \t {ept:.6f} \t {ept_err:.6f} \t {time.time() - init_time:.2f}")
        # if ept_err < options["max_error"] and n > 20:
        #     break

nsamples = n + 1
print(f'# total number of samples {nsamples}')
wt_sp = wt_sp[:nsamples]
t1_sp = t1_sp[:nsamples]
t2_sp = t2_sp[:nsamples]
e0_sp = e0_sp[:nsamples]
e1_sp = e1_sp[:nsamples]
ept_sp = ept_sp[:nsamples]

wt = np.sum(wt_sp)
t1 = np.sum(wt_sp * t1_sp) / wt
t2 = np.sum(wt_sp * t2_sp) / wt
e0 = np.sum(wt_sp * e0_sp) / wt
e1 = np.sum(wt_sp * e1_sp) / wt

ept = (h0 + 1/t1 * e0 + 1/t1 * e1 - 1/t1**2 * t2 * e0).real
# dE = (pE/pt1,pE/pt2,pE/pe0,pE/pe1)
dE = np.array([-1/t1**2 * (e0+e1) + 2/t1**3 * t2 * e0,
               -1/t1**2 * e0,
                1/t1 - 1/t1**2 * t2,
                1/t1])
cov_te0e1 = np.cov([t1_sp, t2_sp, e0_sp, e1_sp])
ept_cov_err = (np.sqrt(dE @ cov_te0e1 @ dE) / np.sqrt(len(wt_sp))).real

ept_sp_err = np.std(ept_sp) / np.sqrt(nsamples)

print(f"# Raw AFQMC/pt2CCSD energy (covariance): {ept:.6f} +/- {ept_cov_err:.6f}")
print(f"# Raw AFQMC/pt2CCSD energy (dir sample): {ept:.6f} +/- {ept_sp_err:.6f}")

# print(f'# Calculate Mahalanobis distance ')
# t1_var = np.sum(t1_sp - t1)**2
# if np.sqrt(t1_var) < 1e-12:
#     print(f'# <exp(T1)> is not varying ({t1_var:.2e}) during the sampling, it might cause numerical in the variance')
#     print(f'# <exp(T1)> is removed from the covariant matrix. Recommend using ptCCSD trial')
#     x = np.vstack([t2_sp, e0_sp, e1_sp]).T
#     mu = np.array([t2, e0, e1])
#     d2 = np.zeros(nsamples)
#     for i in range(nsamples):
#         d2[i] = (x[i]-mu).T @ np.linalg.inv(cov_te0e1[1:,1:]) @ (x[i]-mu)
#         # print(d2[i])

# else:
#     x = np.vstack([t1_sp, t2_sp, e0_sp, e1_sp]).T
#     mu = np.array([t1, t2, e0, e1])
#     d2 = np.zeros(nsamples)
#     for i in range(nsamples):
#         d2[i] = (x[i]-mu).T @ np.linalg.inv(cov_te0e1) @ (x[i]-mu)
#         # print(d2[i], x[i])

# mask = d2 < 20
# print(f'# remove outliers {nsamples - np.sum(mask)}')

# wt_sp = wt_sp[mask]
# t1_sp = t1_sp[mask]
# t2_sp = t2_sp[mask]
# e0_sp = e0_sp[mask]
# e1_sp = e1_sp[mask]

# wt = np.sum(wt_sp)
# t1 = np.sum(wt_sp * t1_sp) / wt
# t2 = np.sum(wt_sp * t2_sp) / wt
# e0 = np.sum(wt_sp * e0_sp) / wt
# e1 = np.sum(wt_sp * e1_sp) / wt

# ept = (h0 + 1/t1 * e0 + 1/t1 * e1 - 1/t1**2 * t2 * e0).real
# # dE = (pE/pt1,pE/pt2,pE/pe0,pE/pe1)
# dE = np.array([-1/t1**2 * (e0+e1) + 2/t1**3 * t2 * e0,
#             -1/t1**2 * e0,
#                 1/t1 - 1/t1**2 * t2,
#                 1/t1])
# cov_te0e1 = np.cov([t1_sp, t2_sp, e0_sp, e1_sp])
# ept_err = (np.sqrt(dE @ cov_te0e1 @ dE) / np.sqrt(len(wt_sp))).real

# print(f"# AFQMC/pt2CCSD energy (outliners removed): {ept:.6f} +/- {ept_err:.6f}")

print(f'# Clean Observation')
d = np.abs(ept_sp-np.median(ept_sp))
d_med = np.median(d) + 1e-7
z = d/d_med
mask = z < 30
print(f'# the outliers zeta {z[~mask]} | energy {ept_sp[~mask]}')

wt_clean = wt_sp[mask]
t1_clean = t1_sp[mask]
t2_clean = t2_sp[mask]
e0_clean = e0_sp[mask]
e1_clean = e1_sp[mask]
ept_clean = ept_sp[mask]

nclean = len(wt_clean)
print('# remove outliers in direct sampling ', nsamples-nclean)

wt = np.sum(wt_clean)
t1 = np.sum(wt_clean * t1_clean) / wt
t2 = np.sum(wt_clean * t2_clean) / wt
e0 = np.sum(wt_clean * e0_clean) / wt
e1 = np.sum(wt_clean * e1_clean) / wt

ept = (h0 + 1/t1 * e0 + 1/t1 * e1 - 1/t1**2 * t2 * e0).real
# dE = (pE/pt1,pE/pt2,pE/pe0,pE/pe1)
dE = np.array([-1/t1**2 * (e0+e1) + 2/t1**3 * t2 * e0,
               -1/t1**2 * e0,
                1/t1 - 1/t1**2 * t2,
                1/t1])
cov_te0e1 = np.cov([t1_clean, t2_clean, e0_clean, e1_clean])
ept_cov_err = (np.sqrt(dE @ cov_te0e1 @ dE) / np.sqrt(nclean)).real
ept_sp_err = np.std(ept_clean) / np.sqrt(nclean)

print(f"# clean AFQMC/pt2CCSD energy (covariance): {ept:.6f} +/- {ept_cov_err:.6f}")
print(f"# clean AFQMC/ptCCSD energy (dir sample): {ept:.6f} +/- {ept_sp_err:.6f}")


max_size = nclean // 10
block_errs = np.zeros(max_size)
print('# Blk_SZ  NBlk  NSmp  Energy  Error')
for i, block_size in enumerate(range(1,max_size+1)):
    n_blocks = nclean // block_size

    wt_truncated = wt_clean[:n_blocks * block_size]
    t1_truncated = t1_clean[:n_blocks * block_size]
    t2_truncated = t2_clean[:n_blocks * block_size]
    e0_truncated = e0_clean[:n_blocks * block_size]
    e1_truncated = e1_clean[:n_blocks * block_size]

    wt_t1 = wt_truncated * t1_truncated
    wt_t2 = wt_truncated * t2_truncated
    wt_e0 = wt_truncated * e0_truncated
    wt_e1 = wt_truncated * e1_truncated

    wt = wt_truncated.reshape(n_blocks, block_size)
    wt_t1 = wt_t1.reshape(n_blocks, block_size)
    wt_t2 = wt_t2.reshape(n_blocks, block_size)
    wt_e0 = wt_e0.reshape(n_blocks, block_size)
    wt_e1 = wt_e1.reshape(n_blocks, block_size)

    # block_wt = np.sum(wt, axis=1)
    block_t1 = np.sum(wt_t1, axis=1)# / block_wt
    block_t2 = np.sum(wt_t2, axis=1)# / block_wt
    block_e0 = np.sum(wt_e0, axis=1)# / block_wt
    block_e1 = np.sum(wt_e1, axis=1)# / block_wt

    block_energy = (h0 + block_e0/block_t1 + block_e1/block_t1 - (block_t2*block_e0)/block_t1**2).real #(block_num / block_den).real
    block_mean = np.mean(block_energy)
    block_error = np.std(block_energy, ddof=1) / np.sqrt(n_blocks)
    print(f' {block_size:3d}  {n_blocks:3d}  {block_size*n_blocks:4d}  {block_mean:.6f}  {block_error:.6f}')
    # block_size[i] = b
    # energy[i] = block_mean
    block_errs[i] = block_error

for i, err in enumerate(block_errs):
    if np.abs((err - block_errs[i-1]) / err) < 0.04:
        break
print(f'# autocorrelation eliminated at blocking {i+1} with estimate error {err:.6f}')

# print(ept_sp)

# ept = np.mean(ept_clean)
# ept_err = np.std(ept_clean) / np.sqrt(len(ept_clean))

print(f"# Blocked clean AFQMC/pt2CCSD energy (dir sample): {ept:.6f} +/- {err:.6f}")

print(f"# total run time: {time.time() - init_time:.2f}")
