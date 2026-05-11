import torch
import torch.nn as nn
import torch.nn.functional as F


# CONV BLOCk


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# BACKBONE TINY @ mobilenet style

class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # stem: 1/2
        self.stem = ConvBNReLU(3, 32, kernel_size=3, stride=2, padding=1)

        # stage2: 1/4, 64 canale
        self.stage2 = nn.Sequential(
            ConvBNReLU(32, 64, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1),
        )

        # stage3: 1/8, 96 canale
        self.stage3 = nn.Sequential(
            ConvBNReLU(64, 96, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(96, 96, kernel_size=3, stride=1, padding=1),
        )

        # stage4: 1/16, 128 canale
        self.stage4 = nn.Sequential(
            ConvBNReLU(96, 128, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(128, 128, kernel_size=3, stride=1, padding=1),
        )

        # stage5: 1/32, 160 canale
        self.stage5 = nn.Sequential(
            ConvBNReLU(128, 160, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(160, 160, kernel_size=3, stride=1, padding=1),
        )

        self.channels = {
            "c2": 64,
            "c3": 96,
            "c4": 128,
            "c5": 160,
        }

    def forward(self, x):
        x = self.stem(x)          # 1/2
        c2 = self.stage2(x)       # 1/4
        c3 = self.stage3(c2)      # 1/8
        c4 = self.stage4(c3)      # 1/16
        c5 = self.stage5(c4)      # 1/32
        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}

# PIXEL DECODER


class PixelDecoder(nn.Module):
    def __init__(self, channels_dict, dim=64):

        super().__init__()

        self.l5 = nn.Conv2d(channels_dict["c5"], dim, 1)
        self.l4 = nn.Conv2d(channels_dict["c4"], dim, 1)
        self.l3 = nn.Conv2d(channels_dict["c3"], dim, 1)
        self.l2 = nn.Conv2d(channels_dict["c2"], dim, 1)

        self.s4 = nn.Conv2d(dim, dim, 3, padding=1)
        self.s3 = nn.Conv2d(dim, dim, 3, padding=1)
        self.s2 = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, feats):
        c2, c3, c4, c5 = feats["c2"], feats["c3"], feats["c4"], feats["c5"]

        p5 = self.l5(c5)                                  # 1/32
        p4 = self.s4(self.l4(c4) + F.interpolate(p5, scale_factor=2, mode="bilinear", align_corners=False))  # 1/16
        p3 = self.s3(self.l3(c3) + F.interpolate(p4, scale_factor=2, mode="bilinear", align_corners=False))  # 1/8
        p2 = self.s2(self.l2(c2) + F.interpolate(p3, scale_factor=2, mode="bilinear", align_corners=False))  # 1/4

        return p2


# TRANSFORMER DECODER TINY

class MaskTransformer(nn.Module):
    def __init__(self, num_classes=5, num_queries=8,
                 hidden=64, num_layers=1, nheads=4, dim_feedforward=128):
        super().__init__()

        self.hidden = hidden
        self.num_queries = num_queries

        self.query_embed = nn.Embedding(num_queries, hidden)

        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden,
                nhead=nheads,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                activation="relu"
            )
            for _ in range(num_layers)
        ])

        self.class_embed = nn.Linear(hidden, num_classes)
        self.mask_embed = nn.Linear(hidden, hidden)

    def forward(self, mask_features):

        B, C, H, W = mask_features.shape

        src = mask_features.flatten(2).transpose(1, 2)  # [B, HW, C]

        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        x = queries  # [B, Q, C]
        for layer in self.layers:
            x = layer(x, src)

        cls = self.class_embed(x)     # [B, Q, num_classes]
        mask = self.mask_embed(x)     # [B, Q, C]

        masks = torch.einsum("bqc, bchw -> bqhw", mask, mask_features)  # [B, Q, H, W]

        return cls, masks



# MASK2FORMER-TINY


class Mask2Former(nn.Module):
    def __init__(self,
                 num_classes=5,
                 backbone_name="tiny",
                 num_queries=8,
                 num_layers=1,
                 hidden_dim=64):

        super().__init__()


        self.backbone = TinyBackbone()

        # Pixel decoder
        self.pixel_decoder = PixelDecoder(self.backbone.channels, dim=hidden_dim)

        # Transformer tiny
        self.transformer = MaskTransformer(
            num_classes=num_classes,
            num_queries=num_queries,
            hidden=hidden_dim,
            num_layers=num_layers,
            nheads=4,
            dim_feedforward=hidden_dim * 2,
        )

    def forward(self, x):

        feats = self.backbone(x)
        mask_features = self.pixel_decoder(feats)  # [B, hidden, H, W]

        cls, masks = self.transformer(mask_features)  # cls: [B,Q,C], masks: [B,Q,H,W]

        cls = cls.softmax(-1)  # [B, Q, num_classes]
        final = torch.einsum("bqc, bqhw -> bchw", cls, masks)  # [B, num_classes, H, W]

        return final
