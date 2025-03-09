import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from model.caformer import caformer_s36_384
from torch.nn.parameter import Parameter
import math
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
from timm.models import create_model
from mmcv.cnn import build_norm_layer


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.PReLU()
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, bias=False,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        # self.bias = bias
        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=bias))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.randn([1, dim, shape]))

    def forward(self, x):
        B, C, N = x.shape
        x = x + F.interpolate(self.pos_embed, size=(N), mode='linear', align_corners=False)

        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class Sea_Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=2,
                 activation=None,
                 norm_cfg=dict(type='BN', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)

        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_row = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

        self.proj_encode_column = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

        # self.dwconv = Conv2d_BN(self.dh + 2 * self.nh_kd, 2 * self.nh_kd + self.dh, ks=3, stride=1, pad=1, dilation=1,
        #                         groups=2 * self.nh_kd + self.dh, norm_cfg=norm_cfg)
        self.act = activation()
        self.pwconv = Conv2d_BN(2 * self.nh_kd + self.dh, dim, ks=1, norm_cfg=norm_cfg)
        self.sigmoid = h_sigmoid()
        self.pct = PCT(nIn=self.dh + 2 * self.nh_kd, d=2)

    def forward(self, x):
        B, C, H, W = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # detail enhance
        qkv = torch.cat([q, k, v], dim=1)
        qkv = self.pct(qkv)
        qkv = self.pwconv(qkv)

        # squeeze axial attention
        ## squeeze row
        qrow = self.pos_emb_rowq(q.mean(-1)).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        krow = self.pos_emb_rowk(k.mean(-1)).reshape(B, self.num_heads, -1, H)
        vrow = v.mean(-1).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        attn_row = torch.matmul(qrow, krow) * self.scale
        attn_row = attn_row.softmax(dim=-1)
        xx_row = torch.matmul(attn_row, vrow)  # B nH H C
        xx_row = self.proj_encode_row(xx_row.permute(0, 1, 3, 2).reshape(B, self.dh, H, 1))

        ## squeeze column
        qcolumn = self.pos_emb_columnq(q.mean(-2)).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape(B, self.num_heads, -1, W)
        vcolumn = v.mean(-2).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale
        attn_column = attn_column.softmax(dim=-1)
        xx_column = torch.matmul(attn_column, vcolumn)  # B nH W C
        xx_column = self.proj_encode_column(xx_column.permute(0, 1, 3, 2).reshape(B, self.dh, 1, W))

        xx = xx_row.add(xx_column)
        xx = v.add(xx)
        xx = self.proj(xx)

        xx = self.sigmoid(xx) * qkv
        return xx


class Conv_fg(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)
        return output


def Split(x, p):
    c = int(x.size()[1])
    c1 = round(c * (1 - p))
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()
    return x1, x2


class TCA(nn.Module):
    def __init__(self, c, d=1, dropout=0, kSize=3, dkSize=3):
        super().__init__()

        self.conv3x3 = Conv_fg(c, c, kSize, 1, padding=1, bn_acti=True)

        self.dconv3x3 = Conv_fg(c, c, (dkSize, dkSize), 1,
                                padding=(1, 1), groups=c, bn_acti=True)

        self.ddconv3x3 = Conv_fg(c, c, (dkSize, dkSize), 1,
                                 padding=(1 * d, 1 * d), groups=c, dilation=(d, d), bn_acti=True)

        self.bp = BNPReLU(c)

    def forward(self, input):
        br = self.conv3x3(input)

        br1 = self.dconv3x3(br)
        br2 = self.ddconv3x3(br)
        br = br + br1 + br2

        output = self.bp(br)
        return output


class PCT(nn.Module):
    def __init__(self, nIn, d=1, dropout=0, p=0.5):
        super().__init__()
        self.p = p
        c = int(nIn) - round(int(nIn) * (1 - p))

        self.TCA = TCA(c, d)

        self.conv1x1 = Conv_fg(nIn, nIn, 1, 1, padding=0, bn_acti=True)

        self.conv_left_1x1 = ConvBNReLU(c, c, kernel_size=1)
        self.conv_left_3x3 = ConvBNReLU(c, c, kernel_size=3)

    def forward(self, input):
        output1, output2 = Split(input, self.p)

        output2 = self.TCA(output2)
        output1 = self.conv_left_1x1(output1) + self.conv_left_3x3(output1)
        output = torch.cat([output1, output2], dim=1)
        # output = self.conv1x1(output)
        return output


class StdPool(nn.Module):
    def __init__(self):
        super(StdPool, self).__init__()

    def forward(self, x):
        b, c, _, _ = x.size()

        std = x.view(b, c, -1).std(dim=2, keepdim=True)
        std = std.reshape(b, c, 1, 1)

        return std


class MCAGate(nn.Module):
    def __init__(self, k_size, pool_types=['avg', 'std']):
        """Constructs a MCAGate module.
        Args:
            k_size: kernel size
            pool_types: pooling type. 'avg': average pooling, 'max': max pooling, 'std': standard deviation pooling.
        """
        super(MCAGate, self).__init__()

        self.pools = nn.ModuleList([])
        for pool_type in pool_types:
            if pool_type == 'avg':
                self.pools.append(nn.AdaptiveAvgPool2d(1))
            elif pool_type == 'max':
                self.pools.append(nn.AdaptiveMaxPool2d(1))
            elif pool_type == 'std':
                self.pools.append(StdPool())
            else:
                raise NotImplementedError

        self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size), stride=1, padding=(0, (k_size - 1) // 2), bias=False)
        self.sigmoid = nn.Sigmoid()

        self.weight = nn.Parameter(torch.rand(2))

    def forward(self, x):
        feats = [pool(x) for pool in self.pools]

        if len(feats) == 1:
            out = feats[0]
        elif len(feats) == 2:
            weight = torch.sigmoid(self.weight)
            out = 1 / 2 * (feats[0] + feats[1]) + weight[0] * feats[0] + weight[1] * feats[1]
        else:
            assert False, "Feature Extraction Exception!"

        out = out.permute(0, 3, 2, 1).contiguous()
        out = self.conv(out)
        out = out.permute(0, 3, 2, 1).contiguous()

        out = self.sigmoid(out)
        out = out.expand_as(x)

        return x * out


class MCALayer(nn.Module):
    def __init__(self, inp, no_spatial=False):
        """Constructs a MCA module.
        Args:
            inp: Number of channels of the input feature maps
            no_spatial: whether to build channel dimension interactions
        """
        super(MCALayer, self).__init__()

        lambd = 1.5
        gamma = 1
        temp = round(abs((math.log2(inp) - gamma) / lambd))
        kernel = temp if temp % 2 else temp - 1

        self.h_cw = MCAGate(3)
        self.w_hc = MCAGate(3)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.c_hw = MCAGate(kernel)

    def forward(self, x):
        x_h = x.permute(0, 2, 1, 3).contiguous()
        x_h = self.h_cw(x_h)
        x_h = x_h.permute(0, 2, 1, 3).contiguous()

        x_w = x.permute(0, 3, 2, 1).contiguous()
        x_w = self.w_hc(x_w)
        x_w = x_w.permute(0, 3, 2, 1).contiguous()

        if not self.no_spatial:
            x_c = self.c_hw(x)
            x_out = 1 / 3 * (x_c + x_h + x_w)
        else:
            x_out = 1 / 2 * (x_h + x_w)

        return x_out


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.flatten(2).transpose(1, 2)  # [B, C, H, W] -> [B, C, H*W] ->[B, H*W, C]
        x = self.proj(x)
        return x


class AF(nn.Module):
    def __init__(self, in_dim):
        super(AF, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.linear_r = nn.Sequential(MLP(input_dim=in_dim, embed_dim=in_dim // 2),
                                      MLP(input_dim=in_dim // 2, embed_dim=in_dim // 4))
        self.linear_d = nn.Sequential(MLP(input_dim=in_dim, embed_dim=in_dim // 2),
                                      MLP(input_dim=in_dim // 2, embed_dim=in_dim // 4))
        self.linear_m = nn.Sequential(MLP(input_dim=in_dim // 4, embed_dim=in_dim // 4))
        self.conv_4 = nn.Sequential(Conv(in_dim, 1, 3, 1), nn.Sigmoid())
        self.conv_out = nn.Sequential(ConvBNReLU(in_dim * 2, in_dim, 3, 1),
                                      ConvBNReLU(in_dim, in_dim // 2, 3, 1),
                                      ConvBNReLU(in_dim // 2, in_dim, 3, 1))
        self.mca = MCALayer(in_dim)
        self.conv7x7_r = nn.Sequential(Conv(in_dim, in_dim, kernel_size=7, stride=1))
        self.conv7x7_t = nn.Sequential(Conv(in_dim, in_dim, kernel_size=7, stride=1))
        self.weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.sig = nn.Sigmoid()
        self.linear_r2 = nn.Sequential(MLP(input_dim=in_dim, embed_dim=in_dim // 2),
                                       MLP(input_dim=in_dim // 2, embed_dim=in_dim))
        self.linear_d2 = nn.Sequential(MLP(input_dim=in_dim, embed_dim=in_dim // 2),
                                       MLP(input_dim=in_dim // 2, embed_dim=in_dim))

    def forward(self, rgb, temp):
        # 比例
        m_r = self.avg(rgb)
        v_r = self.linear_r(m_r)

        m_t = self.avg(temp)
        v_t = self.linear_d(m_t)

        v_mix = self.linear_m(v_r * v_t)
        alpha = self.cos(v_r[:, 0, :], v_mix[:, 0, :])
        beta = self.cos(v_t[:, 0, :], v_mix[:, 0, :])
        a_r = (alpha / (alpha + beta)).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        b_d = (beta / (alpha + beta)).unsqueeze(1).unsqueeze(1).unsqueeze(1)

        # 共有部分
        rt_x = rgb * temp
        rt_sa_sig = self.conv_4(self.mca(rt_x))

        # 差分
        rgb_t = rgb - temp  # rgb独有
        t_rgb = temp - rgb  # t独有
        B, C, H, W = rgb_t.shape
        # rgb_t_max, _ = torch.max(rgb_t, dim=1, keepdim=True)
        # rgb_t_mean = torch.mean(rgb_t, dim=1, keepdim=True)
        # t_rgb_max, _ = torch.max(t_rgb, dim=1, keepdim=True)
        # t_rgb_mean = torch.mean(t_rgb, dim=1, keepdim=True)
        rgb_t_max = self.linear_r2(self.max(rgb_t)).transpose(1, 2).view(B, C, 1, 1)
        t_rgb_max = self.linear_d2(self.max(t_rgb)).transpose(1, 2).view(B, C, 1, 1)
        rgb_t_conv = self.conv7x7_r(self.sig(rgb_t_max) * rgb_t)
        t_rgb_conv = self.conv7x7_t(self.sig(t_rgb_max) * t_rgb)
        nor_weights = F.softmax(self.weight, dim=0)
        rgb_t_w = rgb_t_conv * nor_weights[0]
        t_rgb_w = t_rgb_conv * nor_weights[1]
        rgb_t_sig = self.sig(rgb_t_w)
        t_rgb_sig = self.sig(t_rgb_w)

        # 最后的结果
        rgb_out = rt_sa_sig * rgb * rgb_t_sig * a_r
        t_out = rt_sa_sig * temp * t_rgb_sig * b_d
        F_out = torch.cat((rgb_out, t_out), dim=1)
        F_out = self.conv_out(F_out)

        return F_out


class last_conv(nn.Module):
    def __init__(self, out_dim):
        super(last_conv, self).__init__()

        self.proj_1 = ConvBNReLU(64, out_dim, kernel_size=3, stride=1)
        self.proj_2 = ConvBNReLU(128, out_dim, kernel_size=3, stride=1)
        self.proj_3 = ConvBNReLU(320, out_dim, kernel_size=3, stride=1)
        self.proj_4 = ConvBNReLU(512, out_dim, kernel_size=3, stride=1)

    def forward(self, x4, x3, x2, x1):
        x4 = self.proj_4(x4)
        x3 = self.proj_3(x3)
        x2 = self.proj_2(x2)
        x1 = self.proj_1(x1)

        return x4, x3, x2, x1


class HFF_high(nn.Module):
    def __init__(self, channel):
        super(HFF_high, self).__init__()
        self.relu = nn.PReLU()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2 * channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(2 * channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(2 * channel, channel, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(channel)
        self.bn2 = nn.BatchNorm2d(channel)
        self.bn3 = nn.BatchNorm2d(channel)
        self.bn4 = nn.BatchNorm2d(channel)

        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x1, x2, x3, x4):
        x1 = self.relu(self.bn1(self.conv1(self.upsample(x1))))
        x1 = self.maxpool(self.upsample(x1))

        x2 = torch.cat((x1, x2), 1)
        x2 = self.relu(self.bn2(self.conv2(x2)))
        x2 = x2.mul(x1)
        x2 = self.upsample(self.maxpool(self.upsample(x2)))

        x3 = torch.cat((x2, x3), 1)
        x3 = self.relu(self.bn3(self.conv3(x3)))
        x3 = x3.mul(x2)
        x3 = self.upsample(self.maxpool(self.upsample(x3)))

        x4 = torch.cat((x3, x4), 1)
        x4 = self.relu(self.bn4(self.conv4(x4)))
        x4 = x4.mul(x3)
        x4 = self.upsample(self.maxpool(self.upsample(x4)))

        return x1, x2, x3, x4


class MGTANet(nn.Module):
    def __init__(self, channel=32):
        super(MGTANet, self).__init__()
        self.baseline_net = caformer_s36_384()
        self.baseline_net_t = caformer_s36_384()
        self.layer_dep0 = nn.Conv2d(1, 3, kernel_size=1)
        self.load_pretrained()

        # 模块
        self.Sea_at1 = Sea_Attention(dim=64, num_heads=8, key_dim=24, activation=nn.ReLU6)
        self.Sea_at2 = Sea_Attention(dim=128, num_heads=8, key_dim=24, activation=nn.ReLU6)
        self.Sea_at3 = Sea_Attention(dim=320, num_heads=8, key_dim=24, activation=nn.ReLU6)
        self.Sea_at4 = Sea_Attention(dim=512, num_heads=8, key_dim=24, activation=nn.ReLU6)

        # 融合
        self.af1 = AF(in_dim=64)
        self.af2 = AF(in_dim=128)
        self.af3 = AF(in_dim=320)
        self.af4 = AF(in_dim=512)

        # 解码
        self.hff = HFF_high(channel=64)
        self.conv_pre_f4 = Conv(64, 1, 1, stride=1)
        self.conv_pre_f3 = Conv(64, 1, 1, stride=1)
        self.conv_pre_f2 = Conv(64, 1, 1, stride=1)
        self.conv_pre_f1 = Conv(64, 1, 1, stride=1)
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.last_c = last_conv(out_dim=64)

    def forward(self, x_rgb, x_t):
        F1, F2, F3, F4 = self.baseline_net(x_rgb)
        F1_t, F2_t, F3_t, F4_t = self.baseline_net_t(self.layer_dep0(x_t))
        # 预先
        # F1, F2, F3, F4 = self.preconv_rgb(F1, F2, F3, F4)
        # F1_t, F2_t, F3_t, F4_t = self.preconv_t(F1_t, F2_t, F3_t, F4_t)

        # 融合模块
        af_1 = self.af1(F1, F1_t)
        af_2 = self.af2(F2, F2_t)
        af_3 = self.af3(F3, F3_t)
        af_4 = self.af4(F4, F4_t)

        # 全局和细节
        F1_Sea = self.Sea_at1(af_1)
        F2_Sea = self.Sea_at2(af_2)
        F3_Sea = self.Sea_at3(af_3)
        F4_Sea = self.Sea_at4(af_4)

        # 统一通道
        F4_Sea, F3_Sea, F2_Sea, F1_Sea = self.last_c(F4_Sea, F3_Sea, F2_Sea, F1_Sea)

        # 解码
        y4, y3, y2, y1 = self.hff(F4_Sea, F3_Sea, F2_Sea, F1_Sea)
        y4 = self.conv_pre_f4(y4)
        y3 = self.conv_pre_f3(y3)
        y2 = self.conv_pre_f2(y2)
        y1 = self.conv_pre_f1(y1)

        return self.upsample2(y1), self.upsample4(y2), self.upsample8(y3), self.upsample16(y4)

    def load_pretrained(self):
        # 获取当前模型的参数
        baseline_dict = self.baseline_net.state_dict()
        # 获取预训练的参数
        pretrained_large_dict = torch.load('./model/caformer_s36_384.pth')
        # 加载部分能用的参数
        pretrained_large_dict = {k: v for k, v in pretrained_large_dict.items() if k in baseline_dict}
        # 更新现有的model_dict
        baseline_dict.update(pretrained_large_dict)
        self.baseline_net.load_state_dict(baseline_dict)
        self.baseline_net_t.load_state_dict(baseline_dict)
