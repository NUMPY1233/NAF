import os
import os.path as osp
import torch
import imageio.v2 as iio
import numpy as np
import argparse

from src.config.configloading import load_config
from src.help_function.help import forward_project
from src.render import render, render_image
from src.trainer import Trainer
from src.loss import calc_mse_loss
from src.utils import get_psnr, get_mse, get_psnr_3d, get_ssim_3d, cast_to_image, get_psnr_2d, get_ssim_2d


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/pet.yaml",
                        help="configs file path")
    return parser

parser = config_parser()
args = parser.parse_args()
cfg = load_config(args.config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicTrainer(Trainer):
    def __init__(self):
        """
        Basic network trainer.
        """
        super().__init__(cfg, device)
        print(f"[Start] exp: {cfg['exp']['expname']}, net: Basic network")

    def compute_loss(self, data, global_step, idx_epoch):

        proj=data["sino"]
        ret=render_image(self.net)
        proj_pred = forward_project(ret)
        loss = {"loss": 0.}
        calc_mse_loss(loss, proj, proj_pred)

        # Log
        for ls in loss.keys():
            self.writer.add_scalar(f"train/{ls}", loss[ls].item(), global_step)

        return loss["loss"]

    def eval_step(self, global_step, idx_epoch):
        """
        Evaluation step
        """
        # Evaluate projection
        # select_ind = np.random.choice(len(self.eval_dset))
        sino = self.eval_dset[0]['sino']
        img = self.eval_dset[0]['img']
        H, W = sino.shape
        proj_pred = []
        # for i in range(0, rays.shape[0], self.n_rays):
        #     projs_pred.append(render(rays[i:i+self.n_rays], self.net, self.net_fine, **self.conf["render"])["acc"])
        # projs_pred = torch.cat(projs_pred, 0).reshape(H, W)

        # Evaluate density

        img_pred = render_image(self.net)
        sino_pred = forward_project(img_pred)
        # image_pred = image_pred.squeeze()
        loss = {
            "proj_mse": get_mse(sino_pred, sino),
            "proj_psnr": get_psnr(sino_pred, sino),
            "psnr_2d": get_psnr_2d(img_pred, img),
            "ssim_2d": get_ssim_2d(img_pred, img),
        }

        # Logging


        self.writer.add_image("eval/density (row1: gt, row2: pred)", img,img_pred, global_step, dataformats="HWC")
        self.writer.add_image("eval/projection (left: gt, right: pred)", sino,sino_pred, global_step, dataformats="HWC")

        for ls in loss.keys():
            self.writer.add_scalar(f"eval/{ls}", loss[ls], global_step)
            
        # Save
        # eval_save_dir = osp.join(self.evaldir, f"epoch_{idx_epoch:05d}")
        # os.makedirs(eval_save_dir, exist_ok=True)
        # np.save(osp.join(eval_save_dir, "image_pred.npy"), img_pred.cpu().detach().numpy())
        # np.save(osp.join(eval_save_dir, "image_gt.npy"), img.cpu().detach().numpy())
        # iio.imwrite(osp.join(eval_save_dir, "slice_show_row1_gt_row2_pred.png"), (cast_to_image(show_density)*255).astype(np.uint8))
        # iio.imwrite(osp.join(eval_save_dir, "proj_show_left_gt_right_pred.png"), (cast_to_image(show_proj)*255).astype(np.uint8))
        # with open(osp.join(eval_save_dir, "stats.txt"), "w") as f:
        #     for key, value in loss.items():
        #         f.write("%s: %f\n" % (key, value.item()))

        return loss


trainer = BasicTrainer()
trainer.start()
        
