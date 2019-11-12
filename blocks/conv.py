import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class AtrousConvolutionBlock(nn.Module):
    """
    Normal Convolution Layer Output Size = (W - F + 2P) / S + 1
    DeepLabV3 참고함.
    """
    def __init__(self, in_channel: int, output_channel: int, num_classes):
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = output_channel
        self.num_classes = num_classes

        self.conv_1x1_1 = nn.Conv2d(self.in_channel, output_channel, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(self.out_channel)

        self.conv_3x3_1 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, stride=1, padding=3, dilation=3)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(self.out_channel)

        self.conv_3x3_2 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(self.out_channel)

        self.conv_3x3_3 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, stride=1, padding=9, dilation=9)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(self.out_channel)

        self.avg_pool = nn.AdaptiveMaxPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(self.out_channel)

        self.conv_1x1_3 = nn.Conv2d(5 * self.out_channel, self.out_channel, kernel_size=1)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(self.out_channel)

        self.conv_1x1_4 = nn.Conv2d(self.out_channel, num_classes, kernel_size=1)

    def forward(self, feature_map):
        feature_map_h = feature_map.size()[2]
        feature_map_w = feature_map.size()[3]

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))

        out_img = self.avg_pool(feature_map)
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w), align_corners=True, mode='bilinear')

        print([i.shape for i in[out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img]])

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))
        out = self.conv_1x1_4(out)

        return out


block = AtrousConvolutionBlock(3, 3, 4)
data = torch.randn([8, 3, 32, 32])
res = block(data)
print(res.shape)

for i in res.detach().numpy()[0]:
    plt.imshow(i)
    plt.show()
