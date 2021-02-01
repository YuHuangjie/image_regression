import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

data_id = 'fox'

orders = 2**np.array([4,5,5.5,6,6.5,7,7.5,8])
trials = 5

for order in orders:
        for trial in range(trials):
                command = f'python image_regression.py --config configs/{data_id}.txt '+\
                        f'--logdir=logs/{data_id}-tune-poly/{order}-try{trial} --kernel poly --poly_order {order}' 
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
psnrs = []
for order in orders:
        psnr = 0
        for trial in range(trials):
                result_path = f'logs/{data_id}-tune-poly/{order}-try{trial}/gffm-poly/test_psnr.npy'
                psnr += np.load(result_path)
        psnrs.append(psnr / trials)
ax.plot(orders, psnrs, marker='*')
        
ax.set_xlabel(r'$\alpha$')
ax.set_xlim((orders[0], orders[-1]))
ax.set_title(f'polynomial kernel function', y=-0.4)
ax.grid(True, which='major', alpha=.3)
ax.set_xscale('log', basex=2)
ax.set_ylabel('PSNR')

plt.tight_layout()
plt.savefig('fig_tune_poly.png')
plt.show()
