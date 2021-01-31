import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

data_id = 'fox'

matern_orders = [1/4, 1/2, 1, 3/2, 2, 5/2]
alphas = 2**np.array([3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7])
trials = 5

for order in matern_orders:
        for a in alphas:
                for trial in range(trials):
                        command = f'python image_regression.py --config configs/{data_id}.txt '+\
                                f'--logdir=logs/{data_id}-tune-matern/{order}-{a}-try{trial} --kernel matern --length_scale {a} --matern_order {order}' 
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
for order in matern_orders:
        psnrs = []
        for a in alphas:
                psnr = 0
                for trial in range(trials):
                        result_path = f'logs/{data_id}-tune-matern/{order}-{a}-try{trial}/gffm-matern/test_psnr.npy'
                        psnr += np.load(result_path)
                psnrs.append(psnr / trials)
        ax.plot(alphas, psnrs, label=r'$\nu='+f'{order}'+r'$')
        
ax.set_xlabel(r'length scale')
ax.set_xlim((alphas[0], alphas[-1]))
ax.set_title(f'matern class kernel function', y=-0.4)
ax.grid(True, which='major', alpha=.3)
ax.set_xscale('log', basex=2)
ax.set_ylabel('PSNR')

plt.legend(loc='center left', bbox_to_anchor=(1,.5), handlelength=1)
plt.tight_layout()
plt.savefig('fig_tune_matern.png')
plt.show()
