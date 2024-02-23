#
# Modified by frost
# Contact: xu.frost@gmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Temporal SparseRCNN Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import math

import torch
from torch import nn
import torch.nn.functional as F

from transformer import TransformerEncoderLayer, TransformerEncoder
from lib.align1d.align import Align1DLayer


_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DynamicHeadTime(nn.Module):

    def __init__(
        self,
        roi_input_shape=14,
        d_model=256,
        num_classes=80,
        dropout=0.0,
        nhead=8,
        dim_feedforward=2048,
        num_decoder_layers=6,
        use_dynamic_conv=True,
    ):
        super().__init__()

        # Build RoI.
        self.box_pooler = Align1DLayer(roi_input_shape)

        # Build heads.
        rcnn_head = RCNNHead(
            d_model=d_model,
            num_classes=num_classes,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            dropout=dropout,
            activation="relu",
            block_time_mlp=True,
            use_dynamic_conv=use_dynamic_conv,
        )
        num_heads = num_decoder_layers
        self.d_model = d_model
        self.head_series = _get_clones(rcnn_head, N=num_heads)
        self.return_intermediate = True

        # Gaussian random feature embedding layer for time
        self.d_model = d_model
        time_dim = d_model * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Init parameters.
        self.use_focal = True
        self.num_classes = num_classes
        if self.use_focal:
            prior_prob = 0.01
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss.
            if self.use_focal:
                if p.shape[-1] == self.num_classes:
                    if "delta" in name:
                        pass  # focal loss is only for classification
                    else:
                        nn.init.constant_(p, self.bias_value)


    def forward(self, features, init_bboxes, t, init_features):
        time = self.time_mlp(t)

        inter_class_logits = []
        inter_pred_bboxes = []

        bs = len(features[0])
        bboxes = init_bboxes
        num_boxes = bboxes.shape[1]

        if init_features is not None:
            init_features = init_features[None].repeat(1, bs, 1)
            proposal_features = init_features.clone()
        else:
            proposal_features = None

        for rcnn_head in self.head_series:
            class_logits, pred_bboxes, proposal_features = rcnn_head(
                features, bboxes, proposal_features, self.box_pooler, time
            )

            if self.return_intermediate:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
            bboxes = pred_bboxes.detach()

        if self.return_intermediate:
            return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes)

        return class_logits[None], pred_bboxes[None]


class DynamicHead(nn.Module):

    def __init__(
        self,
        roi_input_shape=14,
        d_model=256,
        num_classes=80,
        dropout=0.0,
        nhead=8,
        dim_feedforward=2048,
        num_decoder_layers=6,
        use_dynamic_conv=True,
        use_attention=True,
    ):
        super().__init__()

        # Build RoI.
        self.box_pooler = Align1DLayer(roi_input_shape)

        # Build heads.
        rcnn_head = RCNNHead(
            d_model=d_model,
            num_classes=num_classes,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            dropout=dropout,
            activation="relu",
            use_dynamic_conv=use_dynamic_conv,
            use_attention=use_attention,
        )
        num_heads = num_decoder_layers
        self.d_model = d_model
        self.head_series = _get_clones(rcnn_head, N=num_heads)
        self.return_intermediate = True

        # Init parameters.
        self.use_focal = True
        self.num_classes = num_classes
        if self.use_focal:
            prior_prob = 0.01
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss.
            if self.use_focal:
                if p.shape[-1] == self.num_classes:
                    if "delta" in name:
                        pass  # focal loss is only for classification
                    else:
                        nn.init.constant_(p, self.bias_value)


    def forward(self, features, init_bboxes, init_features):
        inter_class_logits = []
        inter_pred_bboxes = []

        bs = len(features[0])
        bboxes = init_bboxes

        init_features = init_features[None].repeat(1, bs, 1)
        proposal_features = init_features.clone()

        for rcnn_head in self.head_series:
            class_logits, pred_bboxes, proposal_features = rcnn_head(
                features, bboxes, proposal_features, self.box_pooler
            )

            if self.return_intermediate:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
            bboxes = pred_bboxes.detach()

        if self.return_intermediate:
            return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes)

        return class_logits[None], pred_bboxes[None]


class RCNNHead(nn.Module):

    def __init__(
        self,
        d_model,
        num_classes,
        dim_feedforward=2048,
        nhead=8,
        dropout=0.1,
        activation="relu",
        scale_clamp: float = _DEFAULT_SCALE_CLAMP,
        bbox_weights=(2.0, 1.0),
        block_time_mlp=False,
        use_dynamic_conv=True,
        use_attention=True,
    ):
        super().__init__()

        self.d_model = d_model
        self.use_dynamic_conv = use_dynamic_conv
        self.use_attention = use_attention

        # dynamic.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv()

        self.linear_align = nn.Linear(d_model * 2, d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # block time mlp for diffusion model
        if block_time_mlp:
            self.block_time_mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(d_model * 4, d_model * 2)
            )

        # cls.
        num_cls = 1  # binary
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        num_reg = 3  # MLP
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)

        # pred.
        self.use_focal = True
        if self.use_focal:
            self.class_logits = nn.Linear(d_model, num_classes)
        else:
            self.class_logits = nn.Linear(d_model, num_classes + 1)
        self.bboxes_delta = nn.Linear(d_model, 2)  # center and duration
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

    def forward(self, features, bboxes, pro_features, pooler, time_emb=None):
        """
        :param bboxes: (N, nr_boxes, 2)
        :param pro_features: (N, nr_boxes, d_model)
        """
        N, nr_boxes = bboxes.shape[:2]

        # roi_feature.
        proposal_boxes = list()
        for b in range(N):
            bbox_with_batch = F.pad(bboxes[b], (1, 0), "constant", b)
            proposal_boxes.extend(bbox_with_batch)
        proposal_boxes = torch.stack(proposal_boxes)
        roi_features = pooler(
            features[0], proposal_boxes
        )  #  torch.Size([1200, 10, 14])

        if pro_features is None:
            pro_features = roi_features.view(N, nr_boxes, self.d_model, -1).mean(-1)

        #  (49, N * nr_boxes, self.d_model)
        roi_features = roi_features.permute(2, 0, 1)

        # self_att.
        if self.use_attention:
            pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
            pro_features2 = self.self_attn(
                pro_features, pro_features, value=pro_features
            )[0]
            pro_features = pro_features + self.dropout1(pro_features2)
            pro_features = self.norm1(pro_features)

        # inst_interact.
        if self.use_dynamic_conv:
            pro_features = (
                pro_features.view(nr_boxes, N, self.d_model)
                .permute(1, 0, 2)
                .reshape(1, N * nr_boxes, self.d_model)
            )
            pro_features2 = self.inst_interact(pro_features, roi_features)
            pro_features = pro_features + self.dropout2(pro_features2)
            obj_features = self.norm2(pro_features)
        else:
            roi_features_mean = roi_features.permute(1, 2, 0).mean(
                dim=-1, keepdim=False
            )  #  (N * nr_boxes, self.d_model,1)
            roi_features_mean = roi_features_mean.view(*pro_features.shape)
            pro_features_concate = torch.cat((roi_features_mean, pro_features), -1)
            obj_features = self.activation(self.linear_align(pro_features_concate))
            obj_features = obj_features.view(N, nr_boxes, self.d_model)

        # obj_feature.
        obj_features2 = self.linear2(
            self.dropout(self.activation(self.linear1(obj_features)))
        )
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)

        # - time embedding
        if time_emb is not None:
            scale_shift = self.block_time_mlp(time_emb)
            scale_shift = torch.repeat_interleave(scale_shift, nr_boxes, dim=0)
            scale, shift = scale_shift.chunk(2, dim=1)
            fc_feature = fc_feature * (scale + 1) + shift

        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)

        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 2))

        return (
            class_logits.view(N, nr_boxes, -1),
            pred_bboxes.view(N, nr_boxes, -1),
            obj_features,
        )

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        dur = boxes[:, 1] - boxes[:, 0]
        ctr = boxes[:, 0] + 0.5 * dur

        wc, wd = self.bbox_weights
        dc = deltas[:, 0::2] / wc
        dd = deltas[:, 1::2] / wd

        # Prevent sending too large values into torch.exp()
        dd = torch.clamp(dd, max=self.scale_clamp)

        pred_ctr = dc * dur[:, None] + ctr[:, None]
        pred_dur = torch.exp(dd) * dur[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::2] = pred_ctr - 0.5 * pred_dur  # t1
        pred_boxes[:, 1::2] = pred_ctr + 0.5 * pred_dur  # t2

        return pred_boxes


class DynamicConv(nn.Module):

    def __init__(self):
        super().__init__()

        self.hidden_dim = 256
        self.dim_dynamic = 64
        self.num_dynamic = 2
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(
            self.hidden_dim, self.num_dynamic * self.num_params
        )

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        pooler_resolution = 14
        num_output = self.hidden_dim * pooler_resolution  # 1d case
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        """
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (C_roi, N * nr_boxes, self.d_model)
        """
        features = roi_features.permute(1, 0, 2)  # (M, 14, 256)
        parameters = self.dynamic_layer(pro_features).permute(
            1, 0, 2
        )  # (M, 1, 256x2x64)

        param1 = parameters[:, :, : self.num_params].view(
            -1, self.hidden_dim, self.dim_dynamic
        )  # (M, 256, 64)
        param2 = parameters[:, :, self.num_params :].view(
            -1, self.dim_dynamic, self.hidden_dim
        )  # (M, 64, 256)

        features = torch.bmm(features, param1)  # (M, 14, 64)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)  # (M, 14, 256)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)  # (M, 14x256)
        features = self.out_layer(features)  # (M, 256) -> become proposal
        features = self.norm3(features)
        features = self.activation(features)
        return features


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def build_encoder(args):
    encoder_layer = TransformerEncoderLayer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        normalize_before=args.pre_norm,
    )
    encoder_norm = nn.LayerNorm(args.hidden_dim) if args.pre_norm else None
    return TransformerEncoder(
        encoder_layer, num_layers=args.enc_layers, norm=encoder_norm
    )


def build_head(args=None):
    if args is None:
        return DynamicHeadTime()
    else:
        return DynamicHeadTime(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_decoder_layers=args.dec_layers,
            num_classes=2,  # foreground and background
        )


if __name__ == "__main__":
    d_head = build_head()
    d_head = d_head.cuda()
    B, T, C = 4, 10, 256
    N = 300  # n_prop
    features = [torch.randn(B, C, T).cuda()]
    init_bboxes = torch.rand(B, N, 2)
    init_bboxes = torch.tensor(init_bboxes).cuda().float()
    init_features = torch.randn(N, C).cuda()
    class_logits, pred_bboxes = d_head(features, init_bboxes, init_features)
    print("class_logits", class_logits.shape)
    print("pred_bboxes", pred_bboxes.shape)
