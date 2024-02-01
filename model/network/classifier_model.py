from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Union

import docsaidkit as D
import docsaidkit.torch as DT
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import prettytable
import torch
import torch.nn as nn
from docsaidkit.torch import ArcFace, CosFace
from fitsne import FItSNE
from sklearn.decomposition import PCA
from torch.nn import Parameter
from torch.nn.functional import normalize
from torchmetrics.classification import AUROC, Accuracy, BinaryROC

from .component import *
from .partial_fc_v2 import PartialFC_V2

DIR = D.get_curdir(__file__)

INDOOR_ROOT = '/data/Dataset/indoor_scene_recognition/Images'


def get_num_classes(root: Union[str, Path] = None) -> int:

    if not (fp := DIR.parent.parent / 'data' / 'indoor_cache.json').is_file():
        fs_ind = D.get_files(INDOOR_ROOT, suffix=['.jpg', '.png', '.jpeg'])
        fs_ind_ = [str(f) for f in D.Tqdm(
            fs_ind, desc='Drop Empty images.') if D.imread(f) is not None]
        D.dump_json(fs_ind_, fp)
    else:
        fs_ind_ = D.load_json(fp)

    default_root = DIR.parent.parent / 'data' / 'unique_pool' \
        if root is None else root
    fs = D.get_files(default_root, suffix=['.jpg'])
    return (len(fs) + len(fs_ind_)) * 24  # withs augmentation


class IdentityMarginLoss(nn.Module):

    def __init__(self, s=64.0, m=0.4):
        super().__init__()
        self.s = s
        self.m = m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        logits *= self.s
        return logits


class ClassifierModel(DT.BaseMixin, L.LightningModule):

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.preview_batch = cfg['common']['preview_batch']
        self.apply_solver_config(cfg['optimizer'], cfg['lr_scheduler'])

        # Setup model
        cfg_model = cfg['model']
        self.backbone = DT.Identity()
        self.neck = DT.Identity()
        self.head = DT.Identity()

        if hasattr(cfg_model, 'backbone'):
            self.backbone = globals()[cfg_model['backbone']['name']](
                **cfg_model['backbone']['options'])

            channels = []
            if cfg_model['backbone']['name'] == 'Backbone':
                with torch.no_grad():
                    dummy = torch.rand(1, 3, 128, 128)
                    channels = [i.size(1) for i in self.backbone(dummy)]

        if hasattr(cfg_model, 'neck'):
            cfg_model['neck'].update({'in_channels_list': channels})
            self.neck = DT.build_neck(**cfg_model['neck'])

        if hasattr(cfg_model, 'head'):
            cfg_model['head']['options'].update({'in_channels_list': channels})
            self.head = globals()[cfg_model['head']['name']](
                **cfg_model['head']['options'])

        # Setup loss
        loss_name = cfg_model['loss']['name']
        num_classes = get_num_classes()
        embed_dim = cfg_model['loss']['embed_dim']
        margin_loss = globals()[loss_name](**cfg_model['loss']['options'])
        self.pfc = PartialFC_V2(margin_loss, embed_dim,
                                num_classes, sample_rate=0.3)

        # Setup metrics
        self.auc = AUROC(task='binary')
        self.roc = BinaryROC()

        # for validation
        self.validation_step_outputs = []

    def calc_loss(self, embeddings, labels):
        return self.pfc(embeddings, labels)

    def calc_combinations(self, norm_embeddings, labels):
        combinations_pair = torch.tensor(
            list(combinations(range(norm_embeddings.shape[0]), 2)))
        base_inds, tgt_inds = combinations_pair[:, 0], combinations_pair[:, 1]
        combinations_scores = torch.sum(
            norm_embeddings[base_inds] * norm_embeddings[tgt_inds], dim=-1).add(1).div(2)
        combinations_labels = torch.where(
            labels[base_inds] == labels[tgt_inds], 1, 0)
        return combinations_scores, combinations_labels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        embeddings = self.forward(imgs)
        loss = self.calc_loss(embeddings, labels)

        self.log_dict(
            {
                'lr': self.get_lr(),
                'loss': loss,
            },
            prog_bar=True,
            on_step=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        embeddings = self.forward(imgs)
        norm_embeddings = normalize(embeddings)
        comb_scores, comb_labels = \
            self.calc_combinations(norm_embeddings, labels)
        self.validation_step_outputs.append(
            (comb_scores, comb_labels, norm_embeddings, labels))

    def on_validation_epoch_end(self):

        comb_scores, comb_labels = [], []
        feats, labels = [], []
        for comb_scores_, comb_labels_, feats_, labels_ in self.validation_step_outputs:
            comb_scores.extend(comb_scores_)
            comb_labels.extend(comb_labels_)
            feats.extend(feats_)
            labels.extend(labels_)

        comb_scores = torch.stack(comb_scores)
        comb_labels = torch.stack(comb_labels)
        feats = torch.stack(feats)
        labels = torch.stack(labels)

        feats = feats.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        self.draw_pca(feats, labels)
        self.draw_tsne(feats, labels)
        self.draw_roc(comb_scores, comb_labels)

        self.validation_step_outputs = []

    def draw_roc(self, comb_scores, comb_labels):

        auc = self.auc(comb_scores, comb_labels)
        self.log('valid_auc', auc, prog_bar=True, sync_dist=True)

        fprs, tprs, ths = self.roc(comb_scores, comb_labels)
        fig, ax = plt.subplots()
        fprs = fprs.cpu().numpy()
        tprs = tprs.cpu().numpy()
        ths = ths.cpu().numpy()
        ax.plot(fprs, tprs, label=f'AUC = {auc * 100:.2f} %')

        fpr_row = ["FPR", 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1]
        tpr_row = ["TPR", ]
        th_row = ["Threshold"]

        fprs = np.flipud(fprs)
        tprs = np.flipud(tprs)
        ths = np.flipud(ths)

        for x_label in fpr_row[1:]:
            min_index = np.argmin(np.abs(fprs - x_label))
            tpr_tmp = tprs[min_index].astype(float).round(3)
            th_tmp = ths[min_index].astype(float).round(5)
            tpr_row.append(tpr_tmp)
            th_row.append(th_tmp)

        tpr_fpr_table = prettytable.PrettyTable(header=False)
        tpr_fpr_table.add_row(fpr_row)
        tpr_fpr_table.add_row(tpr_row)
        tpr_fpr_table.add_row(th_row)

        self.log('valid_fpr@4', tpr_row[2], prog_bar=True, sync_dist=True)

        tpr_fpr_table_txt_fpath = str(
            self.preview_dir / f"tpr_fpr_table_{self.current_epoch:04d}.txt")
        with open(tpr_fpr_table_txt_fpath, 'w') as f:
            f.write(tpr_fpr_table.get_string())

        ax.set_ylim([0, 1.0])
        ax.set_yticks(np.linspace(0, 1.0, 11, endpoint=True))
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        fig.legend()
        roc_curve_fpath = str(
            self.preview_dir / f"roc_curve_{self.current_epoch:04d}.png")
        plt.savefig(roc_curve_fpath, dpi=300)
        plt.close()

    def draw_pca(self, feats, labels):

        # plot featuers with pca
        pca = PCA(n_components=2)
        feats = pca.fit_transform(feats)
        explained_ = pca.explained_variance_ratio_

        label_mapper = {
            0: 'IDCardFront',
            1: 'IDCardBack',
            2: 'DriverLicenseFront',
            3: 'HealthIDCard',
            4: 'ResidentIDCardFront',
            5: 'ResidentIDCardBack',
            6: 'VehicleLicense'
        }

        unique_labels = list(label_mapper.keys())
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        markers = ['o', 's', 'D', '^', 'v', '*', '<', '>', 'p', 'h']

        # 繪圖
        plt.figure(figsize=(10, 8))
        for label in unique_labels:
            mapped_label = label_mapper[label]
            plt.scatter(feats[labels == label, 0], feats[labels == label, 1],
                        color=colors[label], marker=markers[label % len(markers)], label=mapped_label, s=3)

        plt.title(
            f"PCA of Features - PC1: {explained_[0]:.2f}, PC2: {explained_[1]:.2f}")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            self.preview_dir / f'epoch_{self.current_epoch}_pca.jpg', dpi=300)
        plt.close()

    def draw_tsne(self, feats, labels):

        # 使用t-SNE進行特徵降維
        feats = FItSNE(feats.astype('float'), perplexity=30, nthreads=8)

        label_mapper = {
            0: 'IDCardFront',
            1: 'IDCardBack',
            2: 'DriverLicenseFront',
            3: 'HealthIDCard',
            4: 'ResidentIDCardFront',
            5: 'ResidentIDCardBack',
            6: 'VehicleLicense'
        }

        unique_labels = list(label_mapper.keys())
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        markers = ['o', 's', 'D', '^', 'v', '*', '<', '>', 'p', 'h']

        # 繪圖
        plt.figure(figsize=(10, 8))
        for label in unique_labels:
            mapped_label = label_mapper[label]
            plt.scatter(feats[labels == label, 0], feats[labels == label, 1],
                        color=colors[label], marker=markers[label % len(markers)], label=mapped_label, s=3)

        plt.title("t-SNE Visualization of Features")
        plt.xlabel("t-SNE Feature 1")
        plt.ylabel("t-SNE Feature 2")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            self.preview_dir / f'epoch_{self.current_epoch}_tsne.jpg', dpi=300)
        plt.close()

    @ property
    def preview_dir(self):
        img_path = Path(self.cfg.root_dir) / "preview" / \
            f'epoch_{self.current_epoch}'
        if not img_path.exists():
            img_path.mkdir(parents=True)
        return img_path

    def preview(self, batch_idx, imgs, labels, logits, suffix='train'):

        # setup preview dir
        preview_dir = self.preview_dir / f'{suffix}_batch_{batch_idx}'
        if not preview_dir.exists():
            preview_dir.mkdir(parents=True)

        imgs = imgs.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        for img, gt, pred in zip(imgs, labels, logits):
            img = np.transpose(img, (1, 2, 0))
            img = (img * 255).astype('uint8')
            pred = np.argmax(pred)

            name = f'gt_{gt}_pred_{pred}.jpg'
            img_output_name = str(preview_dir / name)
            D.imwrite(img, img_output_name)
