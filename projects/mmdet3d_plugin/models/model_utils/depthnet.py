import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.backbones.resnet import BasicBlock
from mmcv.cnn import build_conv_layer
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]
        self.aspp1 = _ASPPModule(
            inplanes,
            mid_channels,
            1,
            padding=0,
            dilation=dilations[0],
            BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(
            int(mid_channels * 5), inplanes, 1, bias=False)
        self.bn1 = BatchNorm(inplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        """
        Args:
            x: (B*N, C, fH, fW)
        Returns:
            x: (B*N, C, fH, fW)
        """
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(
            x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # (B*N, 5*C', fH, fW)

        x = self.conv1(x)   # (B*N, C, fH, fW)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        """
        Args:
            x: (B*N_views, 27)
        Returns:
            x: (B*N_views, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        """
        Args:
            x: (B*N_views, C_mid, fH, fW)
            x_se: (B*N_views, C_mid, 1, 1)
        Returns:
            x: (B*N_views, C_mid, fH, fW)
        """
        x_se = self.conv_reduce(x_se)     # (B*N_views, C_mid, 1, 1)
        x_se = self.act1(x_se)      # (B*N_views, C_mid, 1, 1)
        x_se = self.conv_expand(x_se)   # (B*N_views, C_mid, 1, 1)
        return x * self.gate(x_se)      # (B*N_views, C_mid, fH, fW)


class SemanticGatingModule(nn.Module):
    """
    Semantic-Gated Depth Module (SGDM).
    
    Uses SE-Block style channel attention to gate image features based on 
    semantic logits, explicitly guiding the ill-posed 2D-to-3D depth estimation.
    
    Mathematical formulation:
        D(u,v) = Φ(F_img(u,v) ⊕ Gating(S_sem(u,v)))
    
    Args:
        img_channels (int): Number of image feature channels.
        sem_channels (int): Number of semantic classes (logits channels).
        reduction (int): Channel reduction ratio for SE-block.
    """
    def __init__(self, img_channels, sem_channels, reduction=4):
        super(SemanticGatingModule, self).__init__()
        self.img_channels = img_channels
        self.sem_channels = sem_channels
        
        # Semantic feature projection: project softmax(sem_logits) to img_channels
        self.sem_proj = nn.Sequential(
            nn.Conv2d(sem_channels, img_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(img_channels),
            nn.ReLU(inplace=True)
        )
        
        # SE-Block style channel attention
        mid_channels = max(img_channels // reduction, 16)
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(img_channels * 2, mid_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, img_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, img_feat, sem_logits):
        """
        Args:
            img_feat: (B*N, C_img, H, W) Image features from backbone
            sem_logits: (B*N, C_sem, H, W) Semantic logits from SemanticInjector
        
        Returns:
            gated_feat: (B*N, C_img, H, W) Semantically gated image features
        """
        # Apply softmax to semantic logits to get probability distribution
        sem_prob = F.softmax(sem_logits, dim=1)  # (B*N, C_sem, H, W)
        
        # 软前景门控：计算前景类别的总概率
        # 前景类别索引 0-10（car, truck, construction_vehicle, bus, trailer, 
        #                     barrier, motorcycle, bicycle, pedestrian, traffic_cone, others）
        # 背景类别索引 11-16（driveable_surface, other_flat, sidewalk, terrain, manmade, vegetation）
        num_fg_classes = min(11, sem_prob.shape[1])  # 前景类别数
        fg_prob = sem_prob[:, :num_fg_classes, :, :].sum(dim=1, keepdim=True)  # (B*N, 1, H, W)
        
        # Project semantic features to image feature space
        sem_feat = self.sem_proj(sem_prob)  # (B*N, C_img, H, W)
        
        # Concatenate for SE attention computation
        combined = torch.cat([img_feat, sem_feat], dim=1)  # (B*N, 2*C_img, H, W)
        
        # SE-Block: compute channel attention weights
        attn = self.se_block(combined)  # (B*N, C_img, 1, 1)
        
        # 软前景门控：前景概率加权门控强度
        # fg_prob ≈ 1 (前景区域) → 强门控 img_feat * (1 + attn)
        # fg_prob ≈ 0 (背景区域) → 弱门控 img_feat * 1
        # fg_prob ≈ 0.5 (边界) → 平滑过渡
        gated_feat = img_feat * (1.0 + fg_prob * attn)  # (B*N, C_img, H, W)
        
        return gated_feat


class DepthNet(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 use_dcn=True,
                 use_aspp=True,
                 with_cp=False,
                 stereo=False,
                 bias=0.0,
                 aspp_mid_channels=-1,
                 use_semantic_gating=False,
                 sem_channels=17):
        super(DepthNet, self).__init__()
        
        # Semantic Gating Module (SGDM)
        self.use_semantic_gating = use_semantic_gating
        if use_semantic_gating:
            self.semantic_gating = SemanticGatingModule(
                img_channels=mid_channels,
                sem_channels=sem_channels,
                reduction=8  # 增加 reduction 以减少计算量
            )
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        # 生成context feature
        self.context_conv = nn.Conv2d(
            mid_channels, context_channels, kernel_size=1, stride=1, padding=0)

        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(in_features=27, hidden_features=mid_channels, out_features=mid_channels)
        self.depth_se = SELayer(channels=mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(in_features=27, hidden_features=mid_channels, out_features=mid_channels)
        self.context_se = SELayer(channels=mid_channels)  # NOTE: add camera-aware
        depth_conv_input_channels = mid_channels
        downsample = None

        if stereo:
            depth_conv_input_channels += depth_channels
            downsample = nn.Conv2d(depth_conv_input_channels,
                                    mid_channels, 1, 1, 0)
            cost_volumn_net = []
            for stage in range(int(2)):
                cost_volumn_net.extend([
                    nn.Conv2d(depth_channels, depth_channels, kernel_size=3,
                              stride=2, padding=1),
                    nn.BatchNorm2d(depth_channels)])
            self.cost_volumn_net = nn.Sequential(*cost_volumn_net)
            self.bias = bias

        # 3个残差blocks
        depth_conv_list = [BasicBlock(depth_conv_input_channels, mid_channels,
                                      downsample=downsample),
                           BasicBlock(mid_channels, mid_channels),
                           BasicBlock(mid_channels, mid_channels)]
        if use_aspp:
            if aspp_mid_channels < 0:
                aspp_mid_channels = mid_channels
            depth_conv_list.append(ASPP(mid_channels, aspp_mid_channels))
        if use_dcn:
            depth_conv_list.append(
                build_conv_layer(
                    cfg=dict(
                        type='DCN',
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        kernel_size=3,
                        padding=1,
                        groups=4,
                        im2col_step=128,
                    )))
        depth_conv_list.append(
            nn.Conv2d(
                mid_channels,
                depth_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        self.depth_conv = nn.Sequential(*depth_conv_list)
        self.with_cp = with_cp
        self.depth_channels = depth_channels

    # ----------------------------------------- 用于建立cost volume ----------------------------------
    def gen_grid(self, metas, B, N, D, H, W, hi, wi):
        """
        Args:
            metas: dict{
                k2s_sensor: (B, N_views, 4, 4)
                intrins: (B, N_views, 3, 3)
                post_rots: (B, N_views, 3, 3)
                post_trans: (B, N_views, 3)
                frustum: (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
                cv_downsample: 4,
                downsample: self.img_view_transformer.downsample=16,
                grid_config: self.img_view_transformer.grid_config,
                cv_feat_list: [feat_prev_iv, stereo_feat]
            }
            B: batchsize
            N: N_views
            D: D
            H: fH_stereo
            W: fW_stereo
            hi: H_img
            wi: W_img
        Returns:
            grid: (B*N_views, D*fH_stereo, fW_stereo, 2)
        """
        frustum = metas['frustum']      # (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
        # 逆图像增广:
        points = frustum - metas['post_trans'].view(B, N, 1, 1, 1, 3)
        points = torch.inverse(metas['post_rots']).view(B, N, 1, 1, 1, 3, 3) \
            .matmul(points.unsqueeze(-1))   # (B, N_views, D, fH_stereo, fW_stereo, 3, 1)

        # (u, v, d) --> (du, dv, d)
        # (B, N_views, D, fH_stereo, fW_stereo, 3, 1)
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)

        # cur_pixel --> curr_camera --> prev_camera
        rots = metas['k2s_sensor'][:, :, :3, :3].contiguous()
        trans = metas['k2s_sensor'][:, :, :3, 3].contiguous()
        combine = rots.matmul(torch.inverse(metas['intrins']))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points)
        points += trans.view(B, N, 1, 1, 1, 3, 1)   # (B, N_views, D, fH_stereo, fW_stereo, 3, 1)

        neg_mask = points[..., 2, 0] < 1e-3
        # prev_camera --> prev_pixel
        points = metas['intrins'].view(B, N, 1, 1, 1, 3, 3).matmul(points)
        # (du, dv, d) --> (u, v)   (B, N_views, D, fH_stereo, fW_stereo, 2, 1)
        points = points[..., :2, :] / points[..., 2:3, :]

        # 图像增广
        points = metas['post_rots'][..., :2, :2].view(B, N, 1, 1, 1, 2, 2).matmul(
            points).squeeze(-1)
        points += metas['post_trans'][..., :2].view(B, N, 1, 1, 1, 2)   # (B, N_views, D, fH_stereo, fW_stereo, 2)

        px = points[..., 0] / (wi - 1.0) * 2.0 - 1.0
        py = points[..., 1] / (hi - 1.0) * 2.0 - 1.0
        px[neg_mask] = -2
        py[neg_mask] = -2
        grid = torch.stack([px, py], dim=-1)    # (B, N_views, D, fH_stereo, fW_stereo, 2)
        grid = grid.view(B * N, D * H, W, 2)    # (B*N_views, D*fH_stereo, fW_stereo, 2)
        return grid

    def calculate_cost_volumn(self, metas):
        """
        Args:
            metas: dict{
                k2s_sensor: (B, N_views, 4, 4)
                intrins: (B, N_views, 3, 3)
                post_rots: (B, N_views, 3, 3)
                post_trans: (B, N_views, 3)
                frustum: (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
                cv_downsample: 4,
                downsample: self.img_view_transformer.downsample=16,
                grid_config: self.img_view_transformer.grid_config,
                cv_feat_list: [feat_prev_iv, stereo_feat]
            }
        Returns:
            cost_volumn: (B*N_views, D, fH_stereo, fW_stereo)
        """
        prev, curr = metas['cv_feat_list']    # (B*N_views, C_stereo, fH_stereo, fW_stereo)
        group_size = 4
        _, c, hf, wf = curr.shape   #
        hi, wi = hf * 4, wf * 4     # H_img, W_img
        B, N, _ = metas['post_trans'].shape
        D, H, W, _ = metas['frustum'].shape
        grid = self.gen_grid(metas, B, N, D, H, W, hi, wi).to(curr.dtype)   # (B*N_views, D*fH_stereo, fW_stereo, 2)

        prev = prev.view(B * N, -1, H, W)   # (B*N_views, C_stereo, fH_stereo, fW_stereo)
        curr = curr.view(B * N, -1, H, W)   # (B*N_views, C_stereo, fH_stereo, fW_stereo)
        cost_volumn = 0
        # process in group wise to save memory
        for fid in range(curr.shape[1] // group_size):
            # (B*N_views, group_size, fH_stereo, fW_stereo)
            prev_curr = prev[:, fid * group_size:(fid + 1) * group_size, ...]
            wrap_prev = F.grid_sample(prev_curr, grid,
                                      align_corners=True,
                                      padding_mode='zeros')     # (B*N_views, group_size, D*fH_stereo, fW_stereo)
            # (B*N_views, group_size, fH_stereo, fW_stereo)
            curr_tmp = curr[:, fid * group_size:(fid + 1) * group_size, ...]
            # (B*N_views, group_size, 1, fH_stereo, fW_stereo) - (B*N_views, group_size, D, fH_stereo, fW_stereo)
            # --> (B*N_views, group_size, D, fH_stereo, fW_stereo)
            # https://github.com/HuangJunJie2017/BEVDet/issues/278
            cost_volumn_tmp = curr_tmp.unsqueeze(2) - \
                              wrap_prev.view(B * N, -1, D, H, W)
            cost_volumn_tmp = cost_volumn_tmp.abs().sum(dim=1)      # (B*N_views, D, fH_stereo, fW_stereo)
            cost_volumn += cost_volumn_tmp  # (B*N_views, D, fH_stereo, fW_stereo)
        if not self.bias == 0:
            invalid = wrap_prev[:, 0, ...].view(B * N, D, H, W) == 0
            cost_volumn[invalid] = cost_volumn[invalid] + self.bias

        # matching cost --> prob
        cost_volumn = - cost_volumn
        cost_volumn = cost_volumn.softmax(dim=1)
        return cost_volumn
    # ----------------------------------------- 用于建立cost volume --------------------------------------

    def forward(self, x, mlp_input, stereo_metas=None, sem_logits=None):
        """
        Args:
            x: (B*N_views, C, fH, fW)
            mlp_input: (B, N_views, 27)
            stereo_metas:  None or dict{
                k2s_sensor: (B, N_views, 4, 4)
                intrins: (B, N_views, 3, 3)
                post_rots: (B, N_views, 3, 3)
                post_trans: (B, N_views, 3)
                frustum: (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
                cv_downsample: 4,
                downsample: self.img_view_transformer.downsample=16,
                grid_config: self.img_view_transformer.grid_config,
                cv_feat_list: [feat_prev_iv, stereo_feat]
            }
            sem_logits: (B*N_views, C_sem, fH, fW) or None
                Semantic logits from SemanticInjector for SGDM gating.
        Returns:
            x: (B*N_views, D+C_context, fH, fW)
        """
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))     # (B*N_views, 27)
        x = self.reduce_conv(x)     # (B*N_views, C_mid, fH, fW)
        
        # Apply Semantic Gating if enabled and sem_logits provided
        # 推理时跳过 SGDM 以提升帧率，只在训练时使用
        if self.use_semantic_gating and sem_logits is not None and self.training:
            # Interpolate sem_logits if size mismatch
            if sem_logits.shape[-2:] != x.shape[-2:]:
                sem_logits = F.interpolate(
                    sem_logits, size=x.shape[-2:], 
                    mode='bilinear', align_corners=True
                )
            x = self.semantic_gating(x, sem_logits)  # (B*N_views, C_mid, fH, fW)

        # (B*N_views, 27) --> (B*N_views, C_mid) --> (B*N_views, C_mid, 1, 1)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)    # (B*N_views, C_mid, fH, fW)
        context = self.context_conv(context)        # (B*N_views, C_context, fH, fW)

        # (B*N_views, 27) --> (B*N_views, C_mid) --> (B*N_views, C_mid, 1, 1)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)      # (B*N_views, C_mid, fH, fW)

        if not stereo_metas is None:
            if stereo_metas['cv_feat_list'][0] is None:
                BN, _, H, W = x.shape
                scale_factor = float(stereo_metas['downsample'])/\
                               stereo_metas['cv_downsample']
                cost_volumn = \
                    torch.zeros((BN, self.depth_channels,
                                 int(H*scale_factor),
                                 int(W*scale_factor))).to(x)
            else:
                with torch.no_grad():
                    # https://github.com/HuangJunJie2017/BEVDet/issues/278
                    cost_volumn = self.calculate_cost_volumn(stereo_metas)      # (B*N_views, D, fH_stereo, fW_stereo)
            cost_volumn = self.cost_volumn_net(cost_volumn)     # (B*N_views, D, fH, fW)
            depth = torch.cat([depth, cost_volumn], dim=1)      # (B*N_views, C_mid+D, fH, fW)
        if self.with_cp:
            depth = checkpoint(self.depth_conv, depth)
        else:
            # 3*res blocks +ASPP/DCN + Conv(c_mid-->D)
            depth = self.depth_conv(depth)  # x: (B*N_views, C_mid, fH, fW) --> (B*N_views, D, fH, fW)
        return torch.cat([depth, context], dim=1)


class DepthAggregation(nn.Module):
    """pixel cloud feature extraction."""

    def __init__(self, in_channels, mid_channels, out_channels):
        super(DepthAggregation, self).__init__()

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
        )

    @autocast(False)
    def forward(self, x):
        x = checkpoint(self.reduce_conv, x)
        short_cut = x
        x = checkpoint(self.conv, x)
        x = short_cut + x
        x = self.out_conv(x)
        return x