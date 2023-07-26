from curses import update_lines_cols
from math import comb
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from LPNet import LPNet
from dciknn_cuda import DCI, MDCI
from torch.optim import AdamW
from helpers.utils import ZippedDataset
from models import parse_layer_string


class Sampler:
    def __init__(self, H, sz, preprocess_fn):
        self.pool_size = int(H.force_factor * sz)
        self.preprocess_fn = preprocess_fn
        self.l2_loss = torch.nn.MSELoss(reduce=False).cuda()
        self.H = H
        self.latent_lr = H.latent_lr
        self.entire_ds = torch.arange(sz)
        self.selected_latents = torch.empty([sz, H.latent_dim], dtype=torch.float32)
        self.selected_latents_tmp = torch.empty([sz, H.latent_dim], dtype=torch.float32)

        blocks = parse_layer_string(H.dec_blocks)
        self.block_res = [s[0] for s in blocks]
        self.res = sorted(set([s[0] for s in blocks if s[0] <= H.max_hierarchy]))
        self.neutral_snoise = [torch.zeros([self.H.imle_db_size, 1, s, s], dtype=torch.float32) for s in self.res]
        self.snoise_tmp = [torch.randn([self.H.imle_db_size, 1, s, s], dtype=torch.float32) for s in self.res]
        self.selected_snoise = [torch.randn([sz, 1, s, s,], dtype=torch.float32) for s in self.res]
        self.snoise_pool = [torch.randn([self.pool_size, 1, s, s], dtype=torch.float32) for s in self.res]

        self.selected_dists = torch.empty([sz], dtype=torch.float32).cuda()
        self.selected_dists[:] = np.inf
        self.selected_dists_tmp = torch.empty([sz], dtype=torch.float32).cuda()
        self.temp_latent_rnds = torch.empty([self.H.imle_db_size, self.H.latent_dim], dtype=torch.float32)
        self.temp_samples = torch.empty([self.H.imle_db_size, H.image_channels, self.H.image_size, self.H.image_size],
                                        dtype=torch.float32)

        self.pool_latents = torch.randn([self.pool_size, H.latent_dim], dtype=torch.float32)
        self.sample_pool_usage = torch.ones([sz], dtype=torch.bool)

        self.projections = []
        self.lpips_net = LPNet(pnet_type=H.lpips_net, path=H.lpips_path).cuda()

        fake = torch.zeros(1, 3, H.image_size, H.image_size).cuda()
        out, shapes = self.lpips_net(fake)
        dims = [int(H.proj_dim * 1. / len(out)) for _ in range(len(out))]
        if H.proj_proportion:
            sm = sum([dim.shape[1] for dim in out])
            dims = [int(out[feat_ind].shape[1] * (H.proj_dim / sm)) for feat_ind in range(len(out) - 1)]
            dims.append(H.proj_dim - sum(dims))
        print(dims)
        for ind, feat in enumerate(out):
            print(feat.shape)
            self.projections.append(F.normalize(torch.randn(feat.shape[1], dims[ind]), p=2, dim=1).cuda())

        self.temp_samples_proj = torch.empty([self.H.imle_db_size, sum(dims)], dtype=torch.float32).cuda()
        self.dataset_proj = torch.empty([sz, sum(dims)], dtype=torch.float32)
        self.pool_samples_proj = torch.empty([self.pool_size, sum(dims)], dtype=torch.float32)
        self.snoise_pool_samples_proj = torch.empty([sz * H.snoise_factor, sum(dims)], dtype=torch.float32)

    def get_projected(self, inp, permute=True):
        if permute:
            out, _ = self.lpips_net(inp.permute(0, 3, 1, 2).cuda())
        else:
            out, _ = self.lpips_net(inp.cuda())
        gen_feat = []
        for i in range(len(out)):
            gen_feat.append(torch.mm(out[i], self.projections[i]))
            # TODO divide?
        return torch.cat(gen_feat, dim=1)

    def init_projection(self, dataset):
        for proj_mat in self.projections:
            proj_mat[:] = F.normalize(torch.randn(proj_mat.shape), p=2, dim=1)

        for ind, x in enumerate(DataLoader(TensorDataset(dataset), batch_size=self.H.n_batch)):
            batch_slice = slice(ind * self.H.n_batch, ind * self.H.n_batch + x[0].shape[0])
            self.dataset_proj[batch_slice] = self.get_projected(self.preprocess_fn(x)[1])

    def sample(self, latents, gen, snoise=None):
        with torch.no_grad():
            nm = latents.shape[0]
            if snoise is None:
                for i in range(len(self.res)):
                    self.snoise_tmp[i].normal_()
                snoise = [s[:nm] for s in self.snoise_tmp]
            px_z = gen(latents, snoise).permute(0, 2, 3, 1)
            xhat = (px_z + 1.0) * 127.5
            xhat = xhat.detach().cpu().numpy()
            xhat = np.minimum(np.maximum(0.0, xhat), 255.0).astype(np.uint8)
            return xhat

    def sample_from_out(self, px_z):
        with torch.no_grad():
            px_z = px_z.permute(0, 2, 3, 1)
            xhat = (px_z + 1.0) * 127.5
            xhat = xhat.detach().cpu().numpy()
            xhat = np.minimum(np.maximum(0.0, xhat), 255.0).astype(np.uint8)
            return xhat

    def calc_loss(self, inp, tar, use_mean=True):
        inp_feat, inp_shape = self.lpips_net(inp)
        tar_feat, _ = self.lpips_net(tar)
        res = 0
        for i, g_feat in enumerate(inp_feat):
            res += torch.sum((g_feat - tar_feat[i]) ** 2, dim=1) / (inp_shape[i] ** 2)
        if use_mean:
            return self.H.lpips_coef * res.mean() + self.H.l2_coef * self.l2_loss(inp, tar).mean()
        else:
            return self.H.lpips_coef * res + self.H.l2_coef * torch.mean(self.l2_loss(inp, tar), dim=[1, 2, 3])

    def calc_dists_existing(self, dataset_tensor, gen, dists=None, latents=None, to_update=None, snoise=None):
        if dists is None:
            dists = self.selected_dists
        if latents is None:
            latents = self.selected_latents
        if snoise is None:
            snoise = self.selected_snoise

        if to_update is not None:
            latents = latents[to_update]
            dists = dists[to_update]
            dataset_tensor = dataset_tensor[to_update]
            snoise = [s[to_update] for s in snoise]

        for ind, x in enumerate(DataLoader(TensorDataset(dataset_tensor), batch_size=self.H.n_batch)):
            _, target = self.preprocess_fn(x)
            batch_slice = slice(ind * self.H.n_batch, ind * self.H.n_batch + target.shape[0])
            cur_latents = latents[batch_slice]
            cur_snoise = [s[batch_slice] for s in snoise]
            with torch.no_grad():
                out = gen(cur_latents, cur_snoise)
                dist = self.calc_loss(target.permute(0, 3, 1, 2), out, use_mean=False)
                dists[batch_slice] = torch.squeeze(dist)
        return dists

    def imle_sample(self, dataset, gen, factor=None):
        if factor is None:
            factor = self.H.imle_factor
        imle_pool_size = int(len(dataset) * factor)
        t1 = time.time()
        self.selected_dists_tmp[:] = self.selected_dists[:]
        for i in range(imle_pool_size // self.H.imle_db_size):
            self.temp_latent_rnds.normal_()
            for j in range(len(self.res)):
                self.snoise_tmp[j].normal_()
            for j in range(self.H.imle_db_size // self.H.imle_batch):
                batch_slice = slice(j * self.H.imle_batch, (j + 1) * self.H.imle_batch)
                cur_latents = self.temp_latent_rnds[batch_slice]
                cur_snoise = [x[batch_slice] for x in self.snoise_tmp]
                with torch.no_grad():
                    self.temp_samples[batch_slice] = gen(cur_latents, cur_snoise)
                    self.temp_samples_proj[batch_slice] = self.get_projected(self.temp_samples[batch_slice], False)

            if not gen.module.dci_db:
                device_count = torch.cuda.device_count()
                gen.module.dci_db = MDCI(self.temp_samples_proj.shape[1], num_comp_indices=self.H.num_comp_indices,
                                            num_simp_indices=self.H.num_simp_indices, devices=[i for i in range(device_count)], ts=device_count)

                # gen.module.dci_db = DCI(self.temp_samples_proj.shape[1], num_comp_indices=self.H.num_comp_indices,
                                            # num_simp_indices=self.H.num_simp_indices)
            gen.module.dci_db.add(self.temp_samples_proj)

            t0 = time.time()
            for ind, y in enumerate(DataLoader(dataset, batch_size=self.H.imle_batch)):
                # t2 = time.time()
                _, target = self.preprocess_fn(y)
                x = self.dataset_proj[ind * self.H.imle_batch:ind * self.H.imle_batch + target.shape[0]]
                cur_batch_data_flat = x.float()
                nearest_indices, _ = gen.module.dci_db.query(cur_batch_data_flat, num_neighbours=1)
                nearest_indices = nearest_indices.long()[:, 0]

                batch_slice = slice(ind * self.H.imle_batch, ind * self.H.imle_batch + x.size()[0])
                actual_selected_dists = self.calc_loss(target.permute(0, 3, 1, 2),
                                                       self.temp_samples[nearest_indices].cuda(), use_mean=False)
                # actual_selected_dists = torch.squeeze(actual_selected_dists)

                to_update = torch.nonzero(actual_selected_dists < self.selected_dists[batch_slice], as_tuple=False)
                to_update = torch.squeeze(to_update)
                self.selected_dists[ind * self.H.imle_batch + to_update] = actual_selected_dists[to_update].clone()
                self.selected_latents[ind * self.H.imle_batch + to_update] = self.temp_latent_rnds[nearest_indices[to_update]].clone()
                for k in range(len(self.res)):
                    self.selected_snoise[k][ind * self.H.imle_batch + to_update] = self.snoise_tmp[k][nearest_indices[to_update]].clone()

                del cur_batch_data_flat

            gen.module.dci_db.clear()

        # adding perturbation
        changed = torch.sum(self.selected_dists_tmp != self.selected_dists).item()
        print("Samples and NN are calculated, time: {}, mean: {} # changed: {}, {}%".format(time.time() - t1,
                                                                                            self.selected_dists.mean(),
                                                                                            changed, (changed / len(
                dataset)) * 100))

    def resample_pool(self, gen, ds):
        # self.init_projection(ds)
        self.pool_latents.normal_()
        for i in range(len(self.res)):
            self.snoise_pool[i].normal_()

        for j in range(self.pool_size // self.H.imle_batch):
            batch_slice = slice(j * self.H.imle_batch, (j + 1) * self.H.imle_batch)
            cur_latents = self.pool_latents[batch_slice]
            cur_snosie = [s[batch_slice] for s in self.snoise_pool]
            with torch.no_grad():
                self.pool_samples_proj[batch_slice] = self.get_projected(gen(cur_latents, cur_snosie), False)

    def imle_sample_force(self, dataset, gen, to_update=None):
        if to_update is None:
            to_update = self.entire_ds
        if to_update.shape[0] == 0:
            return

        t1 = time.time()
        print(torch.any(self.sample_pool_usage[to_update]), torch.any(self.sample_pool_usage))
        if torch.any(self.sample_pool_usage[to_update]):
            self.resample_pool(gen, dataset)
            self.sample_pool_usage[:] = False
            print(f'resampling took {time.time() - t1}')

        self.selected_dists_tmp[:] = np.inf
        self.sample_pool_usage[to_update] = True

        with torch.no_grad():
            for i in range(self.pool_size // self.H.imle_db_size):
                pool_slice = slice(i * self.H.imle_db_size, (i + 1) * self.H.imle_db_size)
                if not gen.module.dci_db:
                    device_count = torch.cuda.device_count()
                    gen.module.dci_db = MDCI(self.H.proj_dim, num_comp_indices=self.H.num_comp_indices,
                                                num_simp_indices=self.H.num_simp_indices, devices=[i for i in range(device_count)])
                gen.module.dci_db.add(self.pool_samples_proj[pool_slice])
                pool_latents = self.pool_latents[pool_slice]
                snoise_pool = [b[pool_slice] for b in self.snoise_pool]

                t0 = time.time()
                for ind, y in enumerate(DataLoader(TensorDataset(dataset[to_update]), batch_size=self.H.imle_batch)):
                    _, target = self.preprocess_fn(y)
                    batch_slice = slice(ind * self.H.imle_batch, ind * self.H.imle_batch + target.shape[0])
                    indices = to_update[batch_slice]
                    x = self.dataset_proj[indices]
                    nearest_indices, dci_dists = gen.module.dci_db.query(x.float(), num_neighbours=1)
                    nearest_indices = nearest_indices.long()[:, 0]
                    dci_dists = dci_dists[:, 0]

                    need_update = dci_dists < self.selected_dists_tmp[indices]
                    global_need_update = indices[need_update]

                    self.selected_dists_tmp[global_need_update] = dci_dists[need_update].clone()
                    self.selected_latents_tmp[global_need_update] = pool_latents[nearest_indices[need_update]].clone() + self.H.imle_perturb_coef * torch.randn((need_update.sum(), self.H.latent_dim))
                    for j in range(len(self.res)):
                        self.selected_snoise[j][global_need_update] = snoise_pool[j][nearest_indices[need_update]].clone()

                gen.module.dci_db.clear()

                if i % 100 == 0:
                    print("NN calculated for {} out of {} - {}".format((i + 1) * self.H.imle_db_size, self.pool_size, time.time() - t0))


        if self.H.latent_epoch > 0:
            for param in gen.parameters():
                param.requires_grad = False
        updatable_latents = self.selected_latents_tmp[to_update].clone().requires_grad_(True)
        latent_optimizer = AdamW([updatable_latents], lr=self.latent_lr)
        comb_dataset = ZippedDataset(TensorDataset(dataset[to_update]), TensorDataset(updatable_latents))

        for gd_epoch in range(self.H.latent_epoch):
            losses = []
            for cur, _ in DataLoader(comb_dataset, batch_size=self.H.n_batch):
                x = cur[0]
                latents = cur[1][0]
                _, target = self.preprocess_fn(x)
                gen.zero_grad()
                px_z = gen(latents)  # TODO fix this
                loss = self.calc_loss(px_z, target.permute(0, 3, 1, 2))
                loss.backward()
                latent_optimizer.step()
                updatable_latents.grad.zero_()

                losses.append(loss.detach())
            print('avg loss', gd_epoch, sum(losses) / len(losses))
        self.selected_latents[to_update] = updatable_latents.detach().clone()

        if self.H.latent_epoch > 0:
            for param in gen.parameters():
                param.requires_grad = True
        self.latent_lr = self.latent_lr * (1 - self.H.latent_decay)
