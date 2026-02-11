import numpy as np
import re

def ulno_data(nfrag, lno_thresh, outfile='lno_result.out'):
    lno_thresh = [lno_thresh*10, lno_thresh]

    # nfrag = len(frg_list)
    frg_list = range(nfrag)
    # nelec = np.zeros((nfrag,2),dtype='int32')
    # norb = np.zeros((nfrag,2),dtype='int32')
    nelec_list = []
    norb_list = []
    # eorb_mp2 = np.zeros(nfrag,dtype='float64')
    # eorb_ccsd = np.zeros(nfrag,dtype='float64')
    eorb_pt2cov = np.zeros(nfrag,dtype='float64')
    eorb_pt2cov_err = np.zeros(nfrag,dtype='float64')
    eorb_pt2dir = np.zeros(nfrag,dtype='float64')
    eorb_pt2dir_err = np.zeros(nfrag,dtype='float64')
    run_time = np.zeros(nfrag,dtype='float64')
    for n, i in enumerate(frg_list):
        with open(f"lnoafqmc.out{i+1}", "r") as rf:
            for line in rf:
                if line.startswith('# nelec:'):
                    nums = re.findall(r'\d+', line)
                    nelec_list.append([int(nums[0]), int(nums[1])])
                if line.startswith('# norb:'):
                    nums = re.findall(r'\d+', line)
                    norb_list.append([int(nums[0]), int(nums[1])])
                if "Ept (covariance)" in line:
                    eorb_pt2cov[n] = float(line.split()[-3])
                    eorb_pt2cov_err[n] = float(line.split()[-1])
                if "Ept (direct observation)" in line:
                    eorb_pt2dir[n] = float(line.split()[-3])
                    eorb_pt2dir_err[n] = float(line.split()[-1])
                if "total run time" in line:
                    run_time[n] = float(line.split()[-1])

    nelec = np.array(nelec_list, dtype=np.int32)
    norb  = np.array(norb_list,  dtype=np.int32)

    eafqmc_pt2cov = np.sum(eorb_pt2cov)
    eafqmc_pt2cov_err = np.sqrt(sum(eorb_pt2cov_err**2))
    eafqmc_pt2dir = np.sum(eorb_pt2dir)
    eafqmc_pt2dir_err = np.sqrt(sum(eorb_pt2dir_err**2))
    tot_time = np.sum(run_time)

    with open(outfile, 'w') as out_file:
        print('# frag \t eorb_afqmc/ccsd_pt2(cov) \t eorb_afqmc/ccsd_pt2(dir) \t nelec \t norb \t \t time', file=out_file)
        for n, i in enumerate(frg_list):
            print(f'{i+1:3d} \t'#  {eorb_mp2[n]:.8f}  {eorb_ccsd[n]:.8f}'
                  f'{eorb_pt2cov[n]:.6f} +/- {eorb_pt2cov_err[n]:.6f} \t \t'
                  f'{eorb_pt2dir[n]:.6f} +/- {eorb_pt2dir_err[n]:.6f} \t \t'
                  f'{nelec[n]} \t {norb[n]} \t {run_time[n]:.2f}', file=out_file)
        print(f'# LNO Thresh: ({lno_thresh[0]:.2e},{lno_thresh[1]:.2e})',file=out_file)
        print(f'# LNO Average Number of Electrons: ({nelec[:,0].mean():.1f},{nelec[:,1].mean():.1f})',file=out_file)
        print(f'# LNO Average Number of Basis: ({norb[:,0].mean():.1f},{norb[:,1].mean():.1f})',file=out_file)
        # print(f'# LNO-MP2 Energy: {e_mp2:.8f}',file=out_file)
        # print(f'# LNO-CCSD Energy: {e_ccsd:.8f}',file=out_file)
        print(f'# LNO-AFQMC/CCSD_PT(cov) Energy: {eafqmc_pt2cov:.6f} +/- {eafqmc_pt2cov_err:.6f}',file=out_file)
        print(f'# LNO-AFQMC/CCSD_PT(dir) Energy: {eafqmc_pt2dir:.6f} +/- {eafqmc_pt2dir_err:.6f}',file=out_file)
        # print(f'# MP2 Correction: {emp2_tot-e_mp2:.8f}',file=out_file)
        print(f"# total run time: {tot_time:.2f}",file=out_file)