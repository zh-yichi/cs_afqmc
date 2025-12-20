import numpy as np
nfrag = 62
run_frg_list = range(nfrag)
eorb_pt2 = np.empty(nfrag,dtype='float64')
eorb_pt2_err = np.empty(nfrag,dtype='float64')
run_time = np.empty(nfrag,dtype='float64')
for n, i in enumerate(run_frg_list):
    with open(f"lnoafqmc.out{i+1}", "r") as rf:
        for line in rf:
            if "Ept (direct observation)" in line:
                eorb_pt2[n] = float(line.split()[-3])
                eorb_pt2_err[n] = float(line.split()[-1])
            if "total run time" in line:
                run_time[n] = float(line.split()[-1])

# nelec_list = []
# norb_list = []
eorb_mp2 = []
eorb_ccsd = []
eorb_ccsd_t = []

with open(f"lno_result.out", "r") as rf:
    for line in rf:
        if not line.startswith('#') and len(line)>0 :
            parts = line.split()
            eorb_mp2 = eorb_mp2.append(parts[1])
            eorb_ccsd = eorb_ccsd.append(parts[2])
            eorb_ccsd_t = eorb_ccsd_t.append(parts[3])

# nelec = (np.mean(nelec_list[:,0]),np.mean(nelec_list[:,1]))
# norb = (np.mean(norb_list[:,0]),np.mean(norb_list[:,1]))
eorb_mp2 = np.array(eorb_mp2)
eorb_ccsd = np.array(eorb_ccsd)
eorb_ccsd_t = np.array(eorb_ccsd_t)
e_mp2 = sum(eorb_mp2)
e_ccsd = sum(eorb_ccsd)
e_ccsd_t = sum(eorb_ccsd_t)
e_ccsd_pt = sum(eorb_ccsd)
e_afqmc_pt2 = sum(eorb_pt2)
e_afqmc_pt2_err = np.sqrt(sum(eorb_pt2_err**2))
tot_time = sum(run_time)

with open(f'lno_result_new.out', 'w') as out_file:
    print('# frag  eorb_mp2  eorb_ccsd  eorb_ccsd(t) ' \
            '  eorb_afqmc/hf  eorb_afqmc/ccsd_pt  nelec  norb  time',
            file=out_file)
    for n, i in enumerate(run_frg_list):
        print(f'{i+1:3d}  '
                f'{eorb_mp2[n]:.8f}  {eorb_ccsd[n]:.8f}  {eorb_ccsd_t[n]:.8f}  '
                f'{eorb_pt2[n]:.6f} +/- {eorb_pt2_err[n]:.6f}  {run_time[n]:.2f}', file=out_file)
        

    print(f'# LNO-MP2 Energy: {e_mp2:.8f}',file=out_file)
    print(f'# LNO-CCSD Energy: {e_ccsd:.8f}',file=out_file)
    print(f'# LNO-CCSD(T) Energy: {e_ccsd_pt:.8f}',file=out_file)
    print(f'# LNO-AFQMC/CCSD_PT Energy: {e_afqmc_pt2:.6f} +/- {e_afqmc_pt2_err:.6f}',file=out_file)
    print(f'# MP2 Correction: {-0.41867154:.8f}',file=out_file)
    print(f"# total run time: {tot_time:.2f}",file=out_file)