import os
import time

import imageio
import torch
import wandb
from cleanfid import fid
from torch.utils.data import DataLoader, TensorDataset

from data import set_up_data
from helpers.imle_helpers import backtrack, reconstruct
from helpers.train_helpers import (load_imle, load_opt, save_latents,
                                   save_latents_latest, save_model,
                                   save_snoise, set_up_hyperparams, update_ema)
from helpers.utils import ZippedDataset, get_cpu_stats_over_ranks
from metrics.ppl import calc_ppl
from metrics.ppl_uniform import calc_ppl_uniform
from sampler import Sampler
from visual.generate_rnd import generate_rnd
from visual.generate_rnd_nn import generate_rnd_nn
from visual.generate_sample_nn import generate_sample_nn
from visual.interpolate import random_interp
from visual.nn_interplate import nn_interp
from visual.spatial_visual import spatial_vissual
from visual.utils import (generate_and_save, generate_for_NN,
                          generate_images_initial,
                          get_sample_for_visualization)
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def training_step_imle(H, n, targets, latents, snoise, imle, ema_imle, optimizer, loss_fn):
    t0 = time.time()
    imle.zero_grad()
    px_z = imle(latents, snoise)
    loss = loss_fn(px_z, targets.permute(0, 3, 1, 2))
    loss.backward()
    optimizer.step()
    if ema_imle is not None:
        update_ema(imle, ema_imle, H.ema_rate)

    stats = get_cpu_stats_over_ranks(dict(loss_nans=0, loss=loss))
    stats.update(skipped_updates=0, iter_time=time.time() - t0, grad_norm=0)
    return stats

class DecayLR:
    def __init__(self, tmax=100000, staleness=10):
        self.tmax = int(tmax)
        self.staleness = staleness
        assert self.tmax > 0
        self.lr_step = (0 - 1) / self.tmax

    def step(self, step):
        per = step % self.staleness
        lr = 1 + self.lr_step * step
        lr = lr + ((0 - 1) / self.staleness) * per
        lr = max(1e-6, min(1.0, lr))
        return lr

def get_lrschedule(args, optimizer):
    # if args.lr_schedule:
    #     scheduler = DecayLR(tmax=args.num_steps)
    #     lr_scheduler = LambdaLR(optimizer, lambda x: scheduler.step(x))
    # else:
    #     lr_scheduler = LambdaLR(optimizer, lambda x: 1.0)
    # return lr_scheduler
    scheduler = DecayLR(tmax=args.num_steps, staleness=args.imle_staleness)
    return LambdaLR(optimizer, lambda x: scheduler.step(x))
    # return LambdaLR(optimizer, lambda x: 1.0)
    



def train_loop_imle(H, data_train, data_valid, preprocess_fn, imle, ema_imle, logprint):
    subset_len = len(data_train)
    if H.subset_len != -1:
        subset_len = H.subset_len
    for data_train in DataLoader(data_train, batch_size=subset_len):
        data_train = TensorDataset(data_train[0])
        break

    optimizer, scheduler, _, iterate, _ = load_opt(H, imle, logprint)
    # lr_scheduler = get_lrschedule(H, optimizer)
    lr_scheduler = scheduler

    stats = []
    H.ema_rate = torch.as_tensor(H.ema_rate)

    subset_len = H.subset_len
    if subset_len == -1:
        subset_len = len(data_train)

    sampler = Sampler(H, subset_len, preprocess_fn)

    last_updated = torch.zeros(subset_len, dtype=torch.int16).cuda()
    times_updated = torch.zeros(subset_len, dtype=torch.int8).cuda()
    change_thresholds = torch.empty(subset_len).cuda()
    change_thresholds[:] = H.change_threshold
    best_fid = 100000

    loss_fn = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').cuda()


    epoch = -1
    for outer in range(H.num_epochs):
        for split_ind, split_x_tensor in enumerate(DataLoader(data_train, batch_size=subset_len, pin_memory=True)):
            split_x_tensor = split_x_tensor[0].contiguous()
            split_x = TensorDataset(split_x_tensor)
            sampler.init_projection(split_x_tensor)
            viz_batch_original, _ = get_sample_for_visualization(split_x, preprocess_fn, H.num_images_visualize, H.dataset)

            print('Outer batch - {}'.format(split_ind, len(split_x)))

            while True:
                epoch += 1
                last_updated[:] = last_updated + 1

                sampler.selected_dists[:] = sampler.calc_dists_existing(split_x_tensor, imle, dists=sampler.selected_dists)
                dists_in_threshold = sampler.selected_dists < change_thresholds
                updated_enough = last_updated >= H.imle_staleness
                updated_too_much = last_updated >= H.imle_force_resample
                in_threshold = torch.logical_and(dists_in_threshold, updated_enough)
                all_conditions = torch.logical_or(in_threshold, updated_too_much)
                to_update = torch.nonzero(all_conditions, as_tuple=False).squeeze(1)
                change_thresholds[to_update] = sampler.selected_dists[to_update].clone() * (1 - H.change_coef)

                if epoch == 0:
                    if os.path.isfile(str(H.restore_latent_path)):
                        latents = torch.load(H.restore_latent_path)
                        sampler.selected_latents[:] = latents[:]
                        for x in DataLoader(split_x, batch_size=H.num_images_visualize, pin_memory=True):
                            break
                        batch_slice = slice(0, x[0].size()[0])
                        latents = sampler.selected_latents[batch_slice]
                        with torch.no_grad():
                            snoise = [s[batch_slice] for s in sampler.selected_snoise]
                            generate_for_NN(sampler, x[0], latents, snoise, viz_batch_original.shape, imle,
                                f'{H.save_dir}/NN-samples_{outer}-{split_ind}-imle.png', logprint)
                        print('loaded latest latents')

                    if os.path.isfile(str(H.restore_latent_path)):
                        threshold = torch.load(H.restore_threshold_path)
                        change_thresholds[:] = threshold[:]
                        print('loaded thresholds', torch.mean(change_thresholds))
                    else:
                        to_update = None



                print(to_update)
                sampler.imle_sample_force(split_x_tensor, imle, to_update)

                if to_update is not None:
                    last_updated[to_update] = 0
                    times_updated[to_update] = times_updated[to_update] + 1

                save_latents_latest(H, split_ind, sampler.selected_latents)
                save_latents_latest(H, split_ind, change_thresholds, name='threshold_latest')

                if to_update is not None and to_update.shape[0] >= H.num_images_visualize:
                    latents = sampler.selected_latents[to_update[:H.num_images_visualize]]
                    with torch.no_grad():
                        generate_for_NN(sampler, split_x_tensor[to_update[:H.num_images_visualize]], latents,
                                        [s[to_update[:H.num_images_visualize]] for s in sampler.selected_snoise],
                                        viz_batch_original.shape, imle,
                                        f'{H.save_dir}/NN-samples_{epoch}-imle.png', logprint)

                

                comb_dataset = ZippedDataset(split_x, TensorDataset(sampler.selected_latents))
                data_loader = DataLoader(comb_dataset, batch_size=H.n_batch, pin_memory=True, shuffle=True)
                for cur, indices in data_loader:
                    x = cur[0]
                    latents = cur[1][0]
                    _, target = preprocess_fn(x)
                    cur_snoise = [s[indices] for s in sampler.selected_snoise]
                    # stat = training_step_imle(H, target.shape[0], target, latents, cur_snoise, imle, ema_imle, optimizer, sampler.calc_loss)
                    stat = training_step_imle(H, target.shape[0], target, latents, cur_snoise, imle, ema_imle, optimizer, loss_fn)
                    stats.append(stat)
                    scheduler.step()

                    if iterate % H.iters_per_images == 0:
                        with torch.no_grad():
                            generate_images_initial(H, sampler, viz_batch_original,
                                                    sampler.selected_latents[0: H.num_images_visualize],
                                                    [s[0: H.num_images_visualize] for s in sampler.selected_snoise],
                                                    viz_batch_original.shape, imle, ema_imle,
                                                    f'{H.save_dir}/samples-{iterate}.png', logprint)

                    iterate += 1
                    if iterate % H.iters_per_save == 0:
                        fp = os.path.join(H.save_dir, 'latest')
                        logprint(f'Saving model@ {iterate} to {fp}')
                        save_model(fp, imle, ema_imle, optimizer, H)
                        save_latents_latest(H, split_ind, sampler.selected_latents)
                        save_latents_latest(H, split_ind, change_thresholds, name='threshold_latest')

                    if iterate % H.iters_per_ckpt == 0:
                        save_model(os.path.join(H.save_dir, f'iter-{iterate}'), imle, ema_imle, optimizer, H)
                        save_latents(H, iterate, split_ind, sampler.selected_latents)
                        save_latents(H, iterate, split_ind, change_thresholds, name='threshold')
                        save_snoise(H, iterate, sampler.selected_snoise)

                # lr_scheduler.step()

                cur_dists = torch.empty([subset_len], dtype=torch.float32).cuda()
                cur_dists[:] = sampler.calc_dists_existing(split_x_tensor, imle, dists=cur_dists)
                torch.save(cur_dists, f'{H.save_dir}/latent/dists-{epoch}.npy')
                        
                metrics = {
                    'mean_loss': torch.mean(cur_dists).item(),
                    'std_loss': torch.std(cur_dists).item(),
                    'max_loss': torch.max(cur_dists).item(),
                    'min_loss': torch.min(cur_dists).item(),
                    'epoch': epoch,
                }

                if epoch % H.fid_freq == 0:
                    generate_and_save(H, imle, sampler, subset_len * H.fid_factor)
                    print(f'{H.data_root}/img', f'{H.save_dir}/fid/')
                    cur_fid = fid.compute_fid(f'{H.data_root}/img', f'{H.save_dir}/fid/', verbose=False)
                    if cur_fid < best_fid:
                        best_fid = cur_fid
                        # save models
                        fp = os.path.join(H.save_dir, 'best_fid')
                        logprint(f'Saving model best fid {best_fid} @ {iterate} to {fp}')
                        save_model(fp, imle, ema_imle, optimizer, H)

                    metrics['fid'] = cur_fid
                    metrics['best_fid'] = best_fid
                    

                logprint(model=H.desc, type='train_loss', step=iterate, **metrics)

                if H.use_wandb:
                    wandb.log(metrics, step=iterate)



def main(H=None):
    H_cur, logprint = set_up_hyperparams()
    if not H:
        H = H_cur
    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)
    imle, ema_imle = load_imle(H, logprint)

    if H.use_wandb:
        wandb.init(
            name=H.wandb_name,
            project=H.wandb_project,
            config=H,
            mode=H.wandb_mode,
        )

    os.makedirs(f'{H.save_dir}/fid', exist_ok=True)

    if H.mode == 'eval':
        with torch.no_grad():
            # Generating
            sampler = Sampler(H, len(data_train), preprocess_fn)
            n_samp = H.n_batch
            temp_latent_rnds = torch.randn([n_samp, H.latent_dim], dtype=torch.float32).cuda()
            for i in range(0, H.num_images_to_generate // n_samp):
                if (i % 10 == 0):
                    print(i * n_samp)
                temp_latent_rnds.normal_()
                tmp_snoise = [s[:n_samp].normal_() for s in sampler.snoise_tmp]
                samp = sampler.sample(temp_latent_rnds, imle, tmp_snoise)
                for j in range(n_samp):
                    imageio.imwrite(f'{H.save_dir}/{i * n_samp + j}.png', samp[j])


    elif H.mode == 'reconstruct':

        subset_len = H.subset_len
        if subset_len == -1:
            subset_len = len(data_train)
        ind = 0
        for split_ind, split_x_tensor in enumerate(DataLoader(data_train, batch_size=H.subset_len, pin_memory=True)):
            if (ind == 14):
                break
            split_x = TensorDataset(split_x_tensor[0])
            ind += 1
            
        for param in imle.parameters():
            param.requires_grad = False
        viz_batch_original, _ = get_sample_for_visualization(split_x, preprocess_fn,
                                                                H.num_images_visualize, H.dataset)
        if os.path.isfile(str(H.restore_latent_path)):
            latents = torch.tensor(torch.load(H.restore_latent_path), requires_grad=True)
        else:
            latents = torch.randn([viz_batch_original.shape[0], H.latent_dim], requires_grad=True)
        sampler = Sampler(H, subset_len, preprocess_fn)
        reconstruct(H, sampler, imle, preprocess_fn, viz_batch_original, latents, 'reconstruct', logprint, training_step_imle)

    elif H.mode == 'backtrack':
        for param in imle.parameters():
            param.requires_grad = False
        for split_x in DataLoader(data_train, batch_size=H.subset_len):
            split_x = split_x[0]
            pass
        print(f'split shape is {split_x.shape}')
        sampler = Sampler(H, H.subset_len, preprocess_fn)
        backtrack(H, sampler, imle, preprocess_fn, split_x, logprint, training_step_imle)


    elif H.mode == 'train':
        train_loop_imle(H, data_train, data_valid_or_test, preprocess_fn, imle, ema_imle, logprint)

    elif H.mode == 'ppl':
        sampler = Sampler(H, H.subset_len, preprocess_fn)
        calc_ppl(H, imle, sampler)

    elif H.mode == 'ppl_uniform':
        sampler = Sampler(H, H.subset_len, preprocess_fn)
        calc_ppl_uniform(H, imle, sampler)
    
    elif H.mode == 'interpolate':
        with torch.no_grad():
            for split_x in DataLoader(data_train, batch_size=H.subset_len):
                split_x = split_x[0]
            viz_batch_original, _ = get_sample_for_visualization(split_x, preprocess_fn,
                                                                    H.num_images_visualize, H.dataset)
            sampler = Sampler(H, H.subset_len, preprocess_fn)
            for i in range(H.num_images_to_generate):
                random_interp(H, sampler, (0, 256, 256, 3), imle, f'{H.save_dir}/interp-{i}.png', logprint)

    elif H.mode == 'spatial_visual':
        with torch.no_grad():
            for split_x in DataLoader(data_train, batch_size=H.subset_len):
                split_x = split_x[0]
            viz_batch_original, _ = get_sample_for_visualization(split_x, preprocess_fn,
                                                                    H.num_images_visualize, H.dataset)
            sampler = Sampler(H, H.subset_len, preprocess_fn)
            for i in range(H.num_images_to_generate):
                print(H.num_images_to_generate, i)
                spatial_vissual(H, sampler, (0, 256, 256, 3), imle, f'{H.save_dir}/interp-{i}.png', logprint)

    elif H.mode == 'generate_rnd':
        with torch.no_grad():
            for split_x in DataLoader(data_train, batch_size=H.subset_len):
                split_x = split_x[0]
            viz_batch_original, _ = get_sample_for_visualization(split_x, preprocess_fn,
                                                                    H.num_images_visualize, H.dataset)
            sampler = Sampler(H, H.subset_len, preprocess_fn)
            generate_rnd(H, sampler, (0, 256, 256, 3), imle, f'{H.save_dir}/rnd.png', logprint)

    elif H.mode == 'generate_rnd_nn':
        with torch.no_grad():
            for split_x in DataLoader(data_train, batch_size=len(data_train)):
                split_x = split_x[0]
            viz_batch_original, _ = get_sample_for_visualization(split_x, preprocess_fn,
                                                                    H.num_images_visualize, H.dataset)
            sampler = Sampler(H, H.subset_len, preprocess_fn)
            generate_rnd_nn(H, split_x,  sampler, (0, 256, 256, 3), imle, f'{H.save_dir}', logprint, preprocess_fn)

    elif H.mode == 'nn_interp':
        with torch.no_grad():
            for split_x in DataLoader(data_train, batch_size=len(data_train)):
                split_x = split_x[0]
            viz_batch_original, _ = get_sample_for_visualization(split_x, preprocess_fn,
                                                                    H.num_images_visualize, H.dataset)
            sampler = Sampler(H, H.subset_len, preprocess_fn)
            nn_interp(H, split_x,  sampler, (0, 256, 256, 3), imle, f'{H.save_dir}', logprint, preprocess_fn)

    elif H.mode == 'generate_sample_nn':
        with torch.no_grad():
            for split_x in DataLoader(data_train, batch_size=len(data_train)):
                split_x = split_x[0]
            viz_batch_original, _ = get_sample_for_visualization(split_x, preprocess_fn,
                                                                    H.num_images_visualize, H.dataset)
            sampler = Sampler(H, H.subset_len, preprocess_fn)
            generate_sample_nn(H, split_x,  sampler, (0, 256, 256, 3), imle, f'{H.save_dir}/rnd2.png', logprint, preprocess_fn)

    elif H.mode == 'backtrack_interpolate':
        with torch.no_grad():
            for split_x in DataLoader(data_train, batch_size=H.subset_len):
                split_x = split_x[0]
            viz_batch_original, _ = get_sample_for_visualization(split_x, preprocess_fn,
                                                                    H.num_images_visualize, H.dataset)
            sampler = Sampler(H, H.subset_len, preprocess_fn)
            latents = torch.tensor(torch.load(f'{H.restore_latent_path}'), requires_grad=True, dtype=torch.float32, device='cuda')
            for i in range(latents.shape[0] - 1):
                lat0 = latents[i:i+1]
                lat1 = latents[i+1:i+2]
                sn1 = None
                sn2 = None
                random_interp(H, sampler, (0, 256, 256, 3), imle, f'test/interp-{i}.png', logprint, lat0, lat1, sn1, sn2)


if __name__ == "__main__":
    main()
