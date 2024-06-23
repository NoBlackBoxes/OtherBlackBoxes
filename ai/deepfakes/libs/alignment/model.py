import torch

# Define modules
def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=strd, padding=padding, bias=bias)

class ConvBlock(torch.nn.Module):
    def __init__(self, in_planes, out_planes, use_instance_norm):
        super(ConvBlock, self).__init__()
        self.bn1 = torch.nn.InstanceNorm2d(in_planes) if use_instance_norm else torch.nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = (torch.nn.InstanceNorm2d(int(out_planes / 2)) if use_instance_norm
                    else torch.nn.BatchNorm2d(int(out_planes / 2)))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = (torch.nn.InstanceNorm2d(int(out_planes / 4)) if use_instance_norm
                    else torch.nn.BatchNorm2d(int(out_planes / 4)))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = torch.nn.Sequential(torch.nn.InstanceNorm2d(in_planes) if use_instance_norm
                                            else torch.nn.BatchNorm2d(in_planes),
                                            torch.nn.ReLU(True),
                                            torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False))
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = torch.nn.functional.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = torch.nn.functional.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = torch.nn.functional.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class HourGlass(torch.nn.Module):
    def __init__(self):
        super(HourGlass, self).__init__()
        self.crop_ratio = 0.55
        self.input_size = 256
        self.num_modules = 2
        self.hg_num_features = 256
        self.hg_depth = 4
        self.use_avg_pool = False
        self.use_instance_norm = False
        self.stem_conv_kernel_size = 7
        self.stem_conv_stride = 2
        self.stem_pool_kernel_size = 2
        self.num_landmarks = 68

        self._generate_network(self.hg_depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.hg_num_features,
                                                      self.hg_num_features,
                                                      self.use_instance_norm))

        self.add_module('b2_' + str(level), ConvBlock(self.hg_num_features,
                                                      self.hg_num_features,
                                                      self.use_instance_norm))
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level),ConvBlock(self.hg_num_features,
                                                              self.hg_num_features,
                                                              self.use_instance_norm))
        self.add_module('b3_' + str(level), ConvBlock(self.hg_num_features,
                                                      self.hg_num_features,
                                                      self.use_instance_norm))

    def _forward(self, level, inp):
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        if self.use_avg_pool:
            low1 = torch.nn.functional.avg_pool2d(inp, 2)
        else:
            low1 = torch.nn.functional.max_pool2d(inp, 2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = torch.nn.functional.interpolate(low3, scale_factor=2, mode='nearest')

        return up1 + up2

    def forward(self, x):
        return self._forward(self.hg_depth, x)


class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.crop_ratio = 0.55
        self.input_size = 256
        self.num_modules = 2
        self.hg_num_features = 256
        self.hg_depth = 4
        self.use_avg_pool = False
        self.use_instance_norm = False
        self.stem_conv_kernel_size = 7
        self.stem_conv_stride = 2
        self.stem_pool_kernel_size = 2
        self.num_landmarks = 68
        
        # Stem
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=self.stem_conv_kernel_size,
                               stride=self.stem_conv_stride,
                               padding=self.stem_conv_kernel_size // 2)
        self.bn1 = torch.nn.InstanceNorm2d(64) if self.use_instance_norm else torch.nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128, self.use_instance_norm)
        self.conv3 = ConvBlock(128, 128, self.use_instance_norm)
        self.conv4 = ConvBlock(128, self.hg_num_features, self.use_instance_norm)

        # Hourglasses
        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module), HourGlass())
            self.add_module('top_m_' + str(hg_module), ConvBlock(self.hg_num_features,
                                                                 self.hg_num_features,
                                                                 self.use_instance_norm))
            self.add_module('conv_last' + str(hg_module), torch.nn.Conv2d(self.hg_num_features,
                                                                    self.hg_num_features,
                                                                    kernel_size=1, stride=1, padding=0))
            self.add_module('bn_end' + str(hg_module),
                            torch.nn.InstanceNorm2d(self.hg_num_features) if self.use_instance_norm
                            else torch.nn.BatchNorm2d(self.hg_num_features))
            self.add_module('l' + str(hg_module), torch.nn.Conv2d(self.hg_num_features,
                                                            self.num_landmarks,
                                                            kernel_size=1, stride=1, padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module('bl' + str(hg_module), torch.nn.Conv2d(self.hg_num_features,
                                                                 self.hg_num_features,
                                                                 kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), torch.nn.Conv2d(self.num_landmarks,
                                                                 self.hg_num_features,
                                                                 kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x = self.conv2(torch.nn.functional.relu(self.bn1(self.conv1(x)), True))
        if self.stem_pool_kernel_size > 1:
            if self.use_avg_pool:
                x = torch.nn.functional.avg_pool2d(x, self.stem_pool_kernel_size)
            else:
                x = torch.nn.functional.max_pool2d(x, self.stem_pool_kernel_size)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x
        hg_feats = []
        tmp_out = None
        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = torch.nn.functional.relu(self._modules['bn_end' + str(i)](self._modules['conv_last' + str(i)](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

            hg_feats.append(ll)

        return tmp_out, x, tuple(hg_feats)