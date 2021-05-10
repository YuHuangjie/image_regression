import os
import shutil
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import configargparse
import imageio

from model import *
from sample import *
import numpy as np

def get_2d_mgrid(shape):
    pixel_coords = np.stack(np.mgrid[:shape[0], :shape[1]], axis=-1).astype(np.float32)

    # normalize pixel coords onto [-1, 1]
    pixel_coords[..., 0] = pixel_coords[..., 0] / max(shape[0] - 1, 1)
    pixel_coords[..., 1] = pixel_coords[..., 1] / max(shape[1] - 1, 1)
    pixel_coords -= 0.5
    pixel_coords *= 2.
    # flatten 
    pixel_coords = torch.Tensor(pixel_coords).view(-1, 2)

    return pixel_coords

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image, randomize=False, batch_size=0):

        self.image = image

        self.mgrid = get_2d_mgrid(self.image.shape)
        data = torch.from_numpy(self.image)
        self.data = data.view(-1, 3)
        self.N_samples = self.mgrid.shape[0]
        self.batch_size = batch_size
        self.randomize = randomize

        # shuffle data
        if self.randomize:
            perm = np.random.permutation(self.N_samples)
            self.mgrid = self.mgrid[perm]
            self.data = self.data[perm]

    def __len__(self):
        if self.randomize:
            return self.N_samples // self.batch_size
        else:
            return self.N_samples

    def __getitem__(self, idx):
        if self.randomize:
            start = np.random.randint(0, self.N_samples)
            end = start + self.batch_size
            if end < self.N_samples:
                return (self.mgrid[start:end], self.data[start:end])
            else:
                # rotate
                end -= self.N_samples
                return ( np.vstack([self.mgrid[:end], self.mgrid[start:]]), 
                        np.vstack([ self.data[:end],  self.data[start:]]))
        else:
            return (self.mgrid[idx, :], self.data[idx, :])

def train(model, train_dataloader, lr, epochs, logdir, epochs_til_checkpoint=10, 
    steps_til_summary=100, val_dataloader=None, global_step=0, model_params=None):
    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    os.makedirs(logdir, exist_ok=True)

    checkpoints_dir = os.path.join(logdir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    summaries_dir = os.path.join(logdir, 'summaries')
    os.makedirs(summaries_dir, exist_ok=True)

    writer = SummaryWriter(summaries_dir, purge_step=global_step)

    total_steps = global_step
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        pbar.update(total_steps)
        train_losses = []
        for epoch in range(total_steps//len(train_dataloader), epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                torch.save({'model': model.state_dict(),
                            'params': model_params,
                            'global_step': total_steps},
                           os.path.join(checkpoints_dir, f'model_epoch_{epoch:04}.pt'))
                np.savetxt(os.path.join(checkpoints_dir, f'train_losses_epoch_{epoch:04}.txt'),
                           np.array(train_losses))

                if val_dataloader is not None:
                    tqdm.write("Running partial validation set...")
                    model.eval()
                    with torch.no_grad():
                        val_losses = []
                        for (model_input, gt) in val_dataloader:
                            model_input, gt = model_input.cuda(), gt.cuda()
                            val_loss = model_loss2(model, model_input, gt)
                            val_losses.append(val_loss)
                            if len(val_losses) > 10:
                                break

                        writer.add_scalar("val_loss", torch.mean(torch.Tensor(val_losses)), total_steps)
                        tqdm.write(f"val_loss {torch.mean(torch.Tensor(val_losses))}")
                    model.train()

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()

                model_input = model_input.cuda()
                gt = gt.cuda()

                train_loss = model_loss2(model, model_input, gt)
                writer.add_scalar('train_loss', train_loss.item(), total_steps)
                train_losses.append(train_loss.item())

                optim.zero_grad()
                train_loss.backward()
                optim.step()

                pbar.update(1)

                # evaludate
                if not total_steps % steps_til_summary:
                    tqdm.write(f"Epoch {epoch}, Total loss {train_loss:.6}, iteration time {time.time()-start_time:.6}")

                total_steps += 1

        torch.save({'model': model.state_dict(),
                    'params': model_params,
                    'global_step': total_steps},
                   os.path.join(checkpoints_dir, 'model_final.pt'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))

p = configargparse.ArgumentParser()

p.add_argument('--config', is_config_file=True, help='config file path')
p.add_argument('--logdir', type=str, default='./logs/default', help='root for logging')
p.add_argument('--test_only', action='store_true', help='test only')
p.add_argument('--restart', action='store_true', help='do not reload from checkpoints')

p.add_argument('--image', type=str, default='greece.jpg', help='path to image')

# General training options
p.add_argument('--batch_size', type=int, default=50000)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train for.')
p.add_argument('--kernel', type=str, default="exp", help='choose from [exp], [exp2], [matern], [gamma_exp], [rq], [poly]')
p.add_argument('--siren', action='store_true', help='substitute relu activation function with sin')

p.add_argument('--epochs_til_ckpt', type=int, default=50,
               help='Epoch interval until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Step interval until loss is printed.')
p.add_argument('--model_type', type=str, default='gffm',
               help='Options currently are "relu" (all relu activations), "ffm" (fourier feature mapping),'
                    '"gffm" (generalized ffm)')
p.add_argument('--ffm_map_size', type=int, default=4096,
               help='mapping dimension of ffm')
p.add_argument('--ffm_map_scale', type=float, default=16,
               help='Gaussian mapping scale of positional input')
p.add_argument('--gffm_map_size', type=int, default=4096,
               help='mapping dimension of gffm')

p.add_argument('--length_scale', type=float, default=64, help='(inverse) length scale of [exp,matern,gamma] kernel')
p.add_argument('--matern_order', type=float, default=0.5, help='\nu in Matern class kernel function')
p.add_argument('--gamma_order', type=float, default=1, help='gamma in gamma-exp kernel')
p.add_argument('--rq_order', type=float, default=4, help='order in rational-quadratic kernel')
p.add_argument('--poly_order', type=float, default=4, help='order in polynomial kernel')
args = p.parse_args()

# prepare data loader
image = imageio.imread(args.image).astype(np.float32) / 255.
train_dataset = ImageDataset(image, randomize=True, batch_size=args.batch_size)
val_dataset = ImageDataset(image, randomize=False)
train_dataloader = DataLoader(train_dataset, pin_memory=True, batch_size=1, shuffle=True)

logdir = os.path.join(args.logdir, f'{args.model_type}')
logdir += f'-{args.kernel}' if args.model_type=='gffm' else ''
if args.restart:
    shutil.rmtree(logdir, ignore_errors=True)

# load checkpoints
global_step = 0
model_params = None
state_dict = None
if os.path.exists(os.path.join(logdir, 'checkpoints')):
    ckpts = [os.path.join(logdir, 'checkpoints', f) for f in sorted(os.listdir(os.path.join(logdir, 'checkpoints'))) if 'pt' in f]
    if len(ckpts) > 0 and not args.restart:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        global_step = ckpt['global_step']
        model_params = ckpt['params']
        state_dict = ckpt['model']

# network architecture
network_size = (4,256)

if args.model_type == 'relu':
    model = make_relu_network(*network_size, use_siren=args.siren)
elif args.model_type == 'ffm':
    if model_params is None:
        B = torch.normal(0., args.ffm_map_scale, size=(args.ffm_map_size//2, 2))
    else:
        B = model_params
    model = make_ffm_network(*network_size, B, use_siren=args.siren)
    model_params = (B)
elif args.model_type == 'gffm':
    if model_params is None:
        if args.kernel == 'exp1':
            W = exp_sample(args.length_scale, args.gffm_map_size)
        elif args.kernel == 'exp2':
            W = exp2_sample(args.length_scale, args.gffm_map_size)
        elif args.kernel == 'matern':
            W = matern_sample(args.length_scale, args.matern_order, args.gffm_map_size)
        elif args.kernel == 'gamma_exp':
            W = gamma_exp2_sample(args.length_scale, args.gamma_order, args.gffm_map_size)
        elif args.kernel == 'rq':
            W = rq_sample(args.length_scale, args.rq_order, args.gffm_map_size)
        elif args.kernel == 'poly':
            W = poly_sample(args.poly_order, args.gffm_map_size)
        else:
            raise NotImplementedError
        b = np.random.uniform(0, np.pi * 2, args.gffm_map_size)
    else:
        W, b = model_params
    model = make_rff_network(*network_size, W, b, use_siren=args.siren)
    model_params = (W, b)
else:
    raise NotImplementedError

if state_dict is not None:
        model.load_state_dict(state_dict)
model.cuda()

# training
if not args.test_only:
    train(model, train_dataloader, args.lr, epochs=args.num_epochs, 
        logdir=logdir, epochs_til_checkpoint=args.epochs_til_ckpt, 
        steps_til_summary=args.steps_til_summary, val_dataloader=None,
        global_step=global_step, model_params=model_params)

# make full testing
print("Running full validation set...")
val_dataloader = DataLoader(val_dataset, pin_memory=False, batch_size=args.batch_size, shuffle=False)

model.eval()
with torch.no_grad():
    preds = []
    for (model_input, gt) in tqdm(val_dataloader):
        model_input, gt = model_input.cuda(), gt.cuda()
        model_out = model_pred(model, model_input)
        preds.append(model_out.cpu().numpy())

pred_image = np.vstack(preds).reshape(image.shape)
imageio.imwrite(os.path.join(logdir, "test.png"), np.clip(pred_image * 255, 0, 255).astype(np.uint8))

mse = np.mean((image - pred_image)**2)
psnr = -10 * np.log10(mse)
print(f"PSNR: {psnr}")
np.save(os.path.join(logdir, 'test_psnr.npy'), np.array([psnr]))
