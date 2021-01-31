import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

data_id = 'fox'

gammas = [1/4, 1/2, 3/4, 1, 5/4, 6/4, 7/4, 8/4]
alphas = 2**np.array([3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7])
trials = 5

for gamma in gammas:
        for a in alphas:
                for trial in range(trials):
                        command = f'python image_regression.py --config configs/{data_id}.txt '+\
                                f'--logdir=logs/{data_id}-tune-gamma/{gamma}-{a}-try{trial} --kernel gamma_exp --length_scale {a} --gamma_order {gamma}' 
                        print(command)
                        os.system(command)

'''
Make figure
'''
params = {'legend.fontsize': 12,
         'axes.labelsize': 12,
         'axes.titlesize': 13,
         'xtick.labelsize':10,
         'ytick.labelsize':10}
matplotlib.rcParams.update(params)

plt.figure(figsize=(5,4))
ax = plt.gca()
for gamma in gammas:
        psnrs = []
        for a in alphas:
                psnr = 0
                for trial in range(trials):
                        result_path = f'logs/{data_id}-tune-gamma/{gamma}-{a}-try{trial}/gffm-gamma_exp/test_psnr.npy'
                        psnr += np.load(result_path)
                psnrs.append(psnr / trials)
        ax.plot(alphas, psnrs, label=r'$\gamma='+f'{gamma}'+r'$')
        
ax.set_xlabel(r'length scale')
ax.set_xlim((alphas[0], alphas[-1]))
ax.set_title(f'gamma-exponential kernel function', y=-0.4)
ax.grid(True, which='major', alpha=.3)
ax.set_xscale('log', basex=2)
ax.set_ylabel('PSNR')

plt.legend(loc='center left', bbox_to_anchor=(1,.5), handlelength=1)
plt.tight_layout()
plt.savefig('fig_tune_gamma.png')
plt.show()
