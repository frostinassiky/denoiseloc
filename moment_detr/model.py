# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from moment_detr.span_utils import (
    generalized_temporal_iou,
    span_cxw_to_xx,
    span_xx_to_cxw,
)
from moment_detr.matcher import build_matcher
from moment_detr.position_encoding import build_position_encoding
from moment_detr.misc import accuracy
from moment_detr.head import build_encoder, DynamicHead

import random, math


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class MomentDenoise(nn.Module):
    """Replace Moment-Sparse module by Diffusion"""

    def __init__(
        self,
        backbone,
        head,
        position_embed,
        txt_position_embed,
        txt_dim,
        vid_dim,
        num_queries,
        input_dropout,
        aux_loss=False,
        use_txt_pos=False,
        n_input_proj=2,
        snr_scale=2.0,
        use_sparse_rcnn=False,
    ):
        """Initializes the model.
        Parameters:
            ~transformer: torch module of the transformer architecture. See transformer.py~
            backbone: transformer encoder
            head: transformer decoder, replaced by dynamic head
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Moment-DETR can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            contrastive_align_loss: If true, perform span - tokens contrastive learning
            contrastive_hdim: dimension used for projecting the embeddings before computing contrastive loss
            max_v_l: int, maximum #clips in videos
            span_loss_type: str, one of [l1, ce]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
            # foreground_thd: float, intersection over prediction >= foreground_thd: labeled as foreground
            # background_thd: float, intersection over prediction <= background_thd: labeled background
        """
        super().__init__()
        self.use_sparse_rcnn = use_sparse_rcnn
        self.num_queries = num_queries
        self.backbone = backbone
        self.head = head
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = head.d_model
        self.use_txt_pos = use_txt_pos
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # projection layer
        relu_args = [True] * 3
        relu_args[n_input_proj - 1] = False
        self.input_txt_proj = nn.Sequential(
            *[
                LinearLayer(
                    txt_dim,
                    hidden_dim,
                    layer_norm=True,
                    dropout=input_dropout,
                    relu=relu_args[0],
                ),
                LinearLayer(
                    hidden_dim,
                    hidden_dim,
                    layer_norm=True,
                    dropout=input_dropout,
                    relu=relu_args[1],
                ),
                LinearLayer(
                    hidden_dim,
                    hidden_dim,
                    layer_norm=True,
                    dropout=input_dropout,
                    relu=relu_args[2],
                ),
            ][:n_input_proj]
        )
        self.input_vid_proj = nn.Sequential(
            *[
                LinearLayer(
                    vid_dim,
                    hidden_dim,
                    layer_norm=True,
                    dropout=input_dropout,
                    relu=relu_args[0],
                ),
                LinearLayer(
                    hidden_dim,
                    hidden_dim,
                    layer_norm=True,
                    dropout=input_dropout,
                    relu=relu_args[1],
                ),
                LinearLayer(
                    hidden_dim,
                    hidden_dim,
                    layer_norm=True,
                    dropout=input_dropout,
                    relu=relu_args[2],
                ),
            ][:n_input_proj]
        )

        self.saliency_proj = nn.Linear(hidden_dim, 1)
        self.aux_loss = aux_loss

        # - Sparse RCNN
        self.init_proposal_features = nn.Embedding(self.num_queries, hidden_dim)

        # - build diffusion, new
        if self.use_sparse_rcnn:  # sparse rcnn
            self.init_proposal_boxes = nn.Embedding(self.num_queries, 2)
            nn.init.constant_(self.init_proposal_boxes.weight[:, :1], 0.5)
            nn.init.constant_(self.init_proposal_boxes.weight[:, 1:], 1.0)

        else:  # denoise
            timesteps = 1000
            self.objective = "pred_x0"
            betas = cosine_beta_schedule(timesteps)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
            (timesteps,) = betas.shape
            self.num_timesteps = int(timesteps)

            self.ddim_sampling_eta = 1.0
            self.self_condition = False
            self.scale = snr_scale
            self.box_renewal = True
            self.use_ensemble = True

            self.register_buffer("betas", betas)
            self.register_buffer("alphas_cumprod", alphas_cumprod)
            self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

            # calculations for diffusion q(x_t | x_{t-1}) and others
            self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
            self.register_buffer(
                "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
            )
            self.register_buffer(
                "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
            )
            self.register_buffer(
                "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
            )
            self.register_buffer(
                "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
            )

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, targets=None):
        """The forward expects two tensors:
           - src_txt: [batch_size, L_txt, D_txt] e.g. 32,25,512
           - src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
                will convert to 1 as padding later for transformer
           - src_vid: [batch_size, L_vid, D_vid] e.g. 32,75,2818
           - src_vid_mask: [batch_size, L_vid], containing 0 on padded pixels,
                will convert to 1 as padding later for transformer

        It returns a dict with the following elements:
           - "pred_spans": The normalized boxes coordinates for all queries, represented as
                           (center_x, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        # projection
        src_vid = self.input_vid_proj(src_vid)  # -> 256
        src_txt = self.input_txt_proj(src_txt)  # -> 256
        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat(
            [src_vid_mask, src_txt_mask], dim=1
        ).bool()  # (bsz, L_vid+L_txt)
        video_length = torch.sum(src_vid_mask, dim=-1)  # (bsz)

        # position embedding
        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = (
            self.txt_position_embed(src_txt)
            if self.use_txt_pos
            else torch.zeros_like(src_txt)
        )  # (bsz, L_txt, d)
        pos = torch.cat([pos_vid, pos_txt], dim=1)  # (bsz, L_vid+L_txt)

        # encoder
        memory = self.backbone(
            src.permute(1, 0, 2), src_key_padding_mask=~mask, pos=pos.permute(1, 0, 2)
        )
        memory = memory.permute(1, 2, 0)  # (batch_size, d, L)

        # decoder
        vid_mem = memory.transpose(1, 2)[:, : src_vid.shape[1]]  # (bsz, L_vid, d)
        out = {
            "saliency_scores": self.saliency_proj(vid_mem).squeeze(-1)
        }  # (bsz, L_vid)
        features = [memory]

        if self.use_sparse_rcnn:
            proposal_boxes = self.init_proposal_boxes.weight.clone()
            proposal_boxes = span_cxw_to_xx(proposal_boxes)
            proposal_boxes = (
                proposal_boxes[None] * video_length[:, None, None]
            )  # [1, 10, 2] x [32, 1, 1]

        else:  # diffusion
            if self.training:
                # prepare gt
                targets, x_boxes, noises, t = self.prepare_targets(
                    targets["span_labels"]
                )
                proposal_boxes = (x_boxes * video_length[:, None, None]).float()
            else:
                x = torch.randn(
                    (features[0].shape[0], self.num_queries, 2),
                    device=video_length.device,
                )
                x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
                x_boxes = ((x_boxes / self.scale) + 1) / 2  # goes to [0,1]
                x_boxes = span_cxw_to_xx(x_boxes)
                proposal_boxes = x_boxes * video_length[:, None, None]

        outputs_class, outputs_coord = self.head(
            features, proposal_boxes, self.init_proposal_features.weight
        )

        # the span should be [center, width] normalized by video length
        outputs_coord = span_xx_to_cxw(
            outputs_coord / video_length[None, :, None, None]
        )

        out.update({"pred_logits": outputs_class[-1], "pred_spans": outputs_coord[-1]})

        if self.aux_loss:
            out["aux_outputs"] = [
                {"pred_logits": a, "pred_spans": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
        return out

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def prepare_diffusion_concat(self, gt_boxes):
        """
        :param gt_boxes: (cx, d), normalized
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=gt_boxes.device).long()
        noise = torch.randn(self.num_queries, 2, device=gt_boxes.device)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor(
                [[0.5, 1.0]], dtype=torch.float, device=gt_boxes.device
            )
            num_gt = 1

        if num_gt < self.num_queries:
            box_placeholder = (
                torch.randn(self.num_queries - num_gt, 2, device=gt_boxes.device) / 6.0
                + 0.5
            )  # 3sigma = 1/2 --> sigma: 1/6
            box_placeholder[:, 1:] = torch.clip(box_placeholder[:, 1:], min=1e-4)
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        elif num_gt > self.num_queries:
            select_mask = [True] * self.num_queries + [False] * (
                num_gt - self.num_queries
            )
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]
        else:
            x_start = gt_boxes

        x_start = (x_start * 2.0 - 1.0) * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.0

        diff_boxes = span_cxw_to_xx(x)

        return diff_boxes, noise, t

    def prepare_targets(self, targets):
        new_targets = []
        diffused_boxes = []
        noises = []
        ts = []
        for targets_per_video in targets:
            target = {}
            gt_boxes = targets_per_video["spans"]  # normalized windows in cxw
            d_boxes, d_noise, d_t = self.prepare_diffusion_concat(gt_boxes)
            diffused_boxes.append(d_boxes)
            noises.append(d_noise)
            ts.append(d_t)
            # target["labels"] = 1 # always positive
            target["boxes"] = gt_boxes  # .to(self.device)
            # target["boxes_xyxy"] = targets_per_video.gt_boxes.tensor.to(self.device)
            # target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            # image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            # target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            # target["area"] = targets_per_video.gt_boxes.area().to(self.device)
            new_targets.append(target)

        return (
            new_targets,
            torch.stack(diffused_boxes),
            torch.stack(noises),
            torch.stack(ts),
        )


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        matcher,
        weight_dict,
        eos_coef,
        losses,
        temperature,
        span_loss_type,
        max_v_l,
        saliency_margin=1,
    ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            temperature: float, temperature for NCE loss
            span_loss_type: str, [l1, ce]
            max_v_l: int,
            saliency_margin: float
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.temperature = temperature
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.saliency_margin = saliency_margin

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = (
            self.eos_coef
        )  # lower weight for background (index 1, foreground index 0)
        self.register_buffer("empty_weight", empty_weight)

    def loss_spans(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
        The target spans are expected in format (center_x, w), normalized by the image size.
        """
        assert "pred_spans" in outputs
        targets = targets["span_labels"]
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs["pred_spans"][idx]  # (#spans, max_v_l * 2)
        tgt_spans = torch.cat(
            [t["spans"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )  # (#spans, 2)
        # print('src_spans',src_spans[:10])
        # print('tgt_spans',tgt_spans[:10])
        if self.span_loss_type == "l1":
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction="none")
            loss_giou = 1 - torch.diag(
                generalized_temporal_iou(
                    span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)
                )
            )
        else:  # ce
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_v_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction="none")

            # giou
            # src_span_indices = src_spans.max(1)[1]  # (#spans, 2)
            # src_span_indices[:, 1] += 1  # ed non-inclusive [st, ed)
            #
            # tgt_span_indices = tgt_spans
            # tgt_span_indices[:, 1] += 1
            # loss_giou = 1 - torch.diag(generalized_temporal_iou(src_span_indices, tgt_span_indices))
            loss_giou = loss_span.new_zeros([1])

        losses = {}
        losses["loss_span"] = loss_span.mean()
        losses["loss_giou"] = loss_giou.mean()
        return losses

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # TODO add foreground and background classifier.  use all non-matched as background.
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]  # (batch_size, #queries, #classes=2)
        # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #objects in batch
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(
            src_logits.shape[:2],
            self.background_label,
            dtype=torch.int64,
            device=src_logits.device,
        )  # (batch_size, #queries)
        target_classes[idx] = self.foreground_label

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            self.empty_weight,
            reduction="none",
        )
        losses = {"loss_label": loss_ce.mean()}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = (
                100 - accuracy(src_logits[idx], self.foreground_label)[0]
            )
        return losses

    def loss_saliency(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}
        saliency_scores = outputs["saliency_scores"]  # (N, L)
        pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
        neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
        num_pairs = pos_indices.shape[1]  # typically 2 or 4
        batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
        pos_scores = torch.stack(
            [
                saliency_scores[batch_indices, pos_indices[:, col_idx]]
                for col_idx in range(num_pairs)
            ],
            dim=1,
        )
        neg_scores = torch.stack(
            [
                saliency_scores[batch_indices, neg_indices[:, col_idx]]
                for col_idx in range(num_pairs)
            ],
            dim=1,
        )
        loss_saliency = (
            torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum()
            / (len(pos_scores) * num_pairs)
            * 2
        )  # * 2 to keep the loss the same scale
        return {"loss_saliency": loss_saliency}

    def loss_contrastive_align(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        normalized_text_embed = outputs[
            "proj_txt_mem"
        ]  # (bsz, #tokens, d)  text tokens
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed
        )  # (bsz, #queries, #tokens)
        logits = logits.sum(2) / self.temperature  # (bsz, #queries)
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        pos_term = positive_logits.sum(1)  # (bsz, )
        num_pos = positive_map.sum(1)  # (bsz, )
        neg_term = logits.logsumexp(1)  # (bsz, )
        loss_nce = -pos_term / num_pos + neg_term  # (bsz, )
        losses = {"loss_contrastive_align": loss_nce.mean()}
        return losses

    def loss_contrastive_align_vid_txt(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        # TODO (1)  align vid_mem and txt_mem;
        # TODO (2) change L1 loss as CE loss on 75 labels, similar to soft token prediction in MDETR
        normalized_text_embed = outputs[
            "proj_txt_mem"
        ]  # (bsz, #tokens, d)  text tokens
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed
        )  # (bsz, #queries, #tokens)
        logits = logits.sum(2) / self.temperature  # (bsz, #queries)
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        pos_term = positive_logits.sum(1)  # (bsz, )
        num_pos = positive_map.sum(1)  # (bsz, )
        neg_term = logits.logsumexp(1)  # (bsz, )
        loss_nce = -pos_term / num_pos + neg_term  # (bsz, )
        losses = {"loss_contrastive_align": loss_nce.mean()}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "contrastive_align": self.loss_contrastive_align,
            "saliency": self.loss_saliency,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)
        indices = self.matcher(outputs_without_aux, targets)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if "saliency" == loss:  # skip as it is only in the top layer
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, **kwargs
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [nn.Dropout(dropout), nn.Linear(in_hsz, out_hsz)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_model(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/moment_detr/issues/108#issuecomment-650269223
    device = torch.device(args.device)

    backbone = build_encoder(args)

    # head = build_head(args)
    head = DynamicHead(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_decoder_layers=args.dec_layers,
        num_classes=2,  # foreground and backgorund
        use_dynamic_conv=args.use_dynamic_conv,
        use_attention=args.use_attention,
    )

    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = MomentDenoise(
        backbone,
        head,
        position_embedding,
        txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        aux_loss=args.aux_loss,
        use_txt_pos=args.use_txt_pos,
        n_input_proj=args.n_input_proj,
        snr_scale=args.snr_scale,
        use_sparse_rcnn=args.use_sparse_rcnn,
    )

    matcher = build_matcher(args)
    weight_dict = {
        "loss_span": args.span_loss_coef,
        "loss_giou": args.giou_loss_coef,
        "loss_label": args.label_loss_coef,
        "loss_saliency": args.lw_saliency,
    }

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update(
                {k + f"_{i}": v for k, v in weight_dict.items() if k != "loss_saliency"}
            )
        weight_dict.update(aux_weight_dict)

    losses = ["spans", "labels", "saliency"]
    criterion = SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        eos_coef=args.eos_coef,
        temperature=args.temperature,
        span_loss_type=args.span_loss_type,
        max_v_l=args.max_v_l,
        saliency_margin=args.saliency_margin,
    )
    criterion.to(device)
    return model, criterion
