import torch
from torch import nn

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding.
    """
    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()

        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0

        # angles
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

        # rotation
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)

class LinearFusionAttention(nn.Module):
    r""" Linear Attention with LePE and RoPE.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, **kwargs):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.elu = nn.ELU()
        self.rope = RoPE(shape=(input_resolution[0], input_resolution[1], dim))

        self.qk_l = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.lepe_l = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        self.qk_i = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.lepe_i = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x, y):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        b, n, c = x.shape
        h, w = self.input_resolution[0], self.input_resolution[1]
        num_heads = self.num_heads
        head_dim = c // num_heads

        qk_l = self.qk_l(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q_l, k_l, v_l = qk_l[0], qk_l[1], x
        # q, k, v: b, n, c

        q_l = self.elu(q_l) + 1.0
        k_l = self.elu(k_l) + 1.0
        q_rope_l = self.rope(q_l.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope_l = self.rope(k_l.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q_l = q_l.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_l = k_l.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v_l = v_l.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        qk_i = self.qk_i(y).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q_i, k_i, v_i = qk_i[0], qk_i[1], y
        # q, k, v: b, n, c

        q_i = self.elu(q_i) + 1.0
        k_i = self.elu(k_i) + 1.0
        q_rope_i = self.rope(q_i.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope_i = self.rope(k_i.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q_i = q_i.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_i = k_i.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v_i = v_i.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        # bidirection-fusion
        z_l = 1 / (q_i @ k_l.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv_l = (k_rope_l.transpose(-2, -1) * (n ** -0.5)) @ (v_l * (n ** -0.5))
        x = q_rope_i @ kv_l * z_l

        z_i = 1 / (q_l @ k_i.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv_i = (k_rope_i.transpose(-2, -1) * (n ** -0.5)) @ (v_i * (n ** -0.5))
        y = q_rope_l @ kv_i * z_i

        x = x.transpose(1, 2).reshape(b, n, c)
        v_l = v_l.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe_l(v_l).permute(0, 2, 3, 1).reshape(b, n, c)

        y = y.transpose(1, 2).reshape(b, n, c)
        v_i = v_i.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        y = y + self.lepe_i(v_i).permute(0, 2, 3, 1).reshape(b, n, c)
        return x, y

class LinearFusion(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, out_channel, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.out_channel = out_channel

        # for lidar_features
        self.cpe_l1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm_l1 = norm_layer(dim)
        self.in_proj_l = nn.Linear(dim, dim)
        self.act_proj_l = nn.Linear(dim, dim)
        self.dwc_l = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act_l = nn.SiLU()
        self.out_proj_l = nn.Linear(dim, dim)
        self.drop_path_l = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.cpe_l2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm_l2 = norm_layer(dim)
        self.mlp_l = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        # for img_features
        self.cpe_i1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm_i1 = norm_layer(dim)
        self.in_proj_i = nn.Linear(dim, dim)
        self.act_proj_i = nn.Linear(dim, dim)
        self.dwc_i = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act_i = nn.SiLU()
        self.out_proj_i = nn.Linear(dim, dim)
        self.drop_path_i = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.cpe_i2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm_i2 = norm_layer(dim)
        self.mlp_i = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        self.fusion = LinearFusionAttention(dim=dim, input_resolution=input_resolution, num_heads=num_heads, qkv_bias=qkv_bias)

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.out_channel, 1, padding=0),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(True)
            )
            
    def forward(self, lidar, img):
        H, W = self.input_resolution
        B, L, C = lidar.shape
        assert L == H * W, "input feature has wrong size"
        assert lidar.shape == img.shape, "inputs should have the same shape"

        lidar = lidar + self.cpe_l1(lidar.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut_l = lidar
        lidar = self.norm_l1(lidar)
        act_res_l = self.act_l(self.act_proj_l(lidar))
        lidar = self.in_proj_l(lidar).view(B, H, W, C)
        lidar = self.act_l(self.dwc_l(lidar.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).view(B, L, C)

        img = img + self.cpe_i1(img.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut_i = img
        img = self.norm_i1(img)
        act_res_i = self.act_i(self.act_proj_i(img))
        img = self.in_proj_i(img).view(B, H, W, C)
        img = self.act_i(self.dwc_i(img.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).view(B, L, C)

        # Linear Attention
        lidar, img = self.fusion(lidar, img)

        lidar = self.out_proj_l(lidar * act_res_l)
        lidar = shortcut_l + self.drop_path_l(lidar)
        lidar = lidar + self.cpe_l2(lidar.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        lidar = lidar + self.drop_path_l(self.mlp_l(self.norm_l2(lidar)))
        lidar = lidar.reshape(B, H, W, C).permute(0, 3, 1, 2)

        img = self.out_proj_i(img * act_res_i)
        img = shortcut_i + self.drop_path_i(img)
        img = img + self.cpe_i2(img.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        img = img + self.drop_path_i(self.mlp_i(self.norm_i2(img)))
        img = img.reshape(B, H, W, C).permute(0, 3, 1, 2)

        fusion_features = torch.cat([lidar, img], dim=1)
        fusion_features = self.conv1(fusion_features)
        return fusion_features

class CrossLinearFusion(nn.Module):
    def __init__(self,model_cfg) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        in_channel = self.model_cfg.IN_CHANNEL
        fusion_dim = self.model_cfg.FUSION_DIM
        feature_size = self.model_cfg.FEATURE_SIZE
        num_heads = self.model_cfg.NUM_HEADS
        lidar_out_channel = self.model_cfg.LIDAR_OUT_CHANNEL
        out_channel = self.model_cfg.OUT_CHANNEL

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, lidar_out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(lidar_out_channel),
            nn.ReLU(True)
            )

        self.fusion = LinearFusion(fusion_dim, feature_size, num_heads, out_channel, drop_path=0.1)

    def forward(self,batch_dict):
        """
        Args:
            batch_dict:
                spatial_features_img (tensor): Bev features from image modality
                spatial_features (tensor): Bev features from lidar modality

        Returns:
            batch_dict:
                spatial_features (tensor): Bev features after muli-modal fusion
        """
        img_bev = batch_dict['spatial_features_img']        # (2,128,200,176)
        lidar_bev = batch_dict['spatial_features']          # (2,512,200,176)
        lidar_bev_features = self.conv(lidar_bev)           # (2,128,200,176)

        lidar1 = lidar_bev_features.flatten(2).transpose(1, 2)
        img1 = img_bev.flatten(2).transpose(1, 2)
        mm_bev = self.fusion(lidar1, img1)
        batch_dict['spatial_features'] = mm_bev
        return batch_dict
