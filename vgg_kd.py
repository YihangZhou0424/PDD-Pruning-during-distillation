import math
import torch.nn as nn
import torch


__all__ = ['VGG', 'cvgg8_bn', 'cvgg11_bn_small', 'cvgg11_bn', 'cvgg13_bn', 'cvgg16_bn', 'cvgg19_bn']


class Mask(nn.Module):
    def __init__(self, size=(1, 128, 1, 1), finding_masks=True):
        super(Mask, self).__init__()
        self.mask = nn.Parameter(torch.randn(size, requires_grad=True)*0.001)  # 0.001让权重变敏感
        self.size = size
        self.finding_masks = finding_masks

    def forward(self, x):
        # True
        if self.finding_masks:
            out_forward = torch.sign(self.mask)
            mask1 = self.mask < -1
            mask2 = self.mask < 0
            mask3 = self.mask < 1
            out1 = (-1) * mask1.type(torch.float32) + (self.mask * self.mask + 2 * self.mask) * (1 - mask1.type(torch.float32))
            out2 = out1 * mask2.type(torch.float32) + (-self.mask * self.mask + 2 * self.mask) * (1 - mask2.type(torch.float32))
            out3 = out2 * mask3.type(torch.float32) + 1 * (1 - mask3.type(torch.float32))
            out = out_forward.detach() - out3.detach() + out3
            return (out+1)/2 * x
        # False
        else:
            return x

    def extra_repr(self):
        s = ('size={size}')
        return s.format(**self.__dict__)


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, finding_masks, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        # self.mask = Mask((1, 512, 1, 1), finding_masks)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )

        self.handlers = []
        self.masks_outputs = {}
        self.origs_outputs = {}
        self.masks = {}
        self.get_masks()

         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    # get masks outputs
    def hook_mask(self, layer, mask_name):
        def hook_function(module, input, output):
            self.masks_outputs[mask_name] = output

        return layer.register_forward_hook(hook_function)


    def hook_masks(self):
        ind = 0
        sz = len(self.features)
        for layer_index in range(sz):
            if type(self.features[layer_index]) == Mask:
                layer_name = 'mask.' + str(ind)
                ind += 1
                self.handlers.append(self.hook_mask(self.features[layer_index], layer_name))
        layer_name = 'mask.' + str(ind)
        # ind += 1
        # self.handlers.append(self.hook_mask(self.mask, layer_name))


    def get_masks_outputs(self):
        return self.masks_outputs

    def remove_hooks(self):
        while len(self.handlers) > 0:
            self.handlers.pop().remove()

    def get_masks(self):
        ind = 0
        sz = len(self.features)
        for layer_index in range(sz):
            if type(self.features[layer_index]) == Mask:
                layer_name = 'mask.' + str(ind)
                ind += 1
                self.masks[layer_name] = self.features[layer_index]
        layer_name = 'mask.' + str(ind)
        # ind += 1
        # self.masks[layer_name] = self.mask
        return self.masks

    def save_masks(self, path):
        tmp_masks = {}
        masks = self.get_masks()
        for key in masks.keys():
            tmp_masks[key] = masks[key].mask
        torch.save(tmp_masks, path)
        return path

    def load_masks(self, path):
        trained_masks = torch.load(path)
        ind = 0
        sz = len(self.features)
        for layer_index in range(sz):
            if type(self.features[layer_index]) == Mask:
                layer_name = 'mask.' + str(ind)
                ind += 1
                self.features[layer_index].data = trained_masks[layer_name].data
        layer_name = 'mask.' + str(ind)
        # ind += 1
        # self.mask.data = trained_masks[layer_name].data
        return path

    def forward(self, x):
        x = self.features(x)
        # x = self.mask(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm, finding_masks):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'msk':
            mask = Mask((1, in_channels, 1, 1), finding_masks)
            layers += [mask]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    "v8": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M"],
    'A': [64, 'msk', 'M', 128, 'msk', 'M', 256, 'msk', 256, 'msk', 'M', 512, 'msk', 512, 'msk', 'M', 512, 'msk', 512, 'msk', 'M'],

    'A_v16_v11_c10': [35, 'msk', 'M', 69, 'msk', 'M', 195, 'msk', 96, 'msk', 'M', 300, 'msk', 109, 'msk', 'M', 227, 'msk', 368, 'msk', 'M'],
    'www': [35, 'msk', 'M', 69, 'msk', 'M', 195, 'msk', 103, 'msk', 'M', 307, 'msk', 119, 'msk', 'M', 227, 'msk', 398, 'msk', 'M'],
    'A_r56_v11_c10': [34, 'msk', 'M', 56, 'msk', 'M', 193, 'msk', 98, 'msk', 'M', 298, 'msk', 115, 'msk', 'M', 221, 'msk', 371, 'msk', 'M'],
    'A_r56_v11_c100': [39, 'msk', 'M', 67, 'msk', 'M', 194, 'msk', 104, 'msk', 'M', 315, 'msk', 128, 'msk', 'M', 244, 'msk', 360, 'msk', 'M'],# 手动调整
    'A_v16_v11_c100': [38, 'msk', 'M', 69, 'msk', 'M', 183, 'msk', 105, 'msk', 'M', 328, 'msk', 138, 'msk', 'M', 261, 'msk', 365, 'msk', 'M'],
    # 'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    # 'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    # 'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'B': [64, 'msk', 64, 'msk', 'M', 128, 'msk', 128, 'msk', 'M', 256, 'msk', 256, 'msk', 'M', 512, 'msk', 512, 'msk', 'M', 512, 'msk', 512, 'msk', 'M'],
    'D': [64, 'msk', 64, 'msk', 'M', 128, 'msk', 128, 'msk', 'M', 256, 'msk', 256, 'msk', 256, 'msk', 'M', 512, 'msk', 512, 'msk', 512, 'msk', 'M', 512, 'msk', 512, 'msk', 512, 'msk', 'M'],
    'E': [64, 'msk', 64, 'msk', 'M', 128, 'msk', 128, 'msk', 'M', 256, 'msk', 256, 'msk', 256, 'msk', 256, 'msk', 'M', 512, 'msk', 512, 'msk', 512, 'msk', 512, 'msk', 'M', 512, 'msk', 512, 'msk', 512, 'msk', 512, 'msk', 'M'],
}


def cvgg8_bn(finding_masks, num_classes, batch_norm=True):
    return VGG(finding_masks, make_layers(cfg['v8'], batch_norm=batch_norm, finding_masks=finding_masks), num_classes=num_classes)


def cvgg11_bn(finding_masks, num_classes, batch_norm=True):
    return VGG(finding_masks, make_layers(cfg['A'], batch_norm=batch_norm, finding_masks=finding_masks), num_classes=num_classes)


def cvgg11_bn_small(finding_masks, num_classes, batch_norm=True):
    return VGG(finding_masks, make_layers(cfg['www'], batch_norm=batch_norm, finding_masks=finding_masks), num_classes=num_classes)  # 手动调整


def cvgg13_bn(finding_masks, num_classes, batch_norm=True):
    return VGG(finding_masks, make_layers(cfg['B'], batch_norm=batch_norm, finding_masks=finding_masks), num_classes=num_classes)


def cvgg16_bn(finding_masks, num_classes, batch_norm=True):
    return VGG(finding_masks, make_layers(cfg['D'], batch_norm=batch_norm, finding_masks=finding_masks), num_classes=num_classes)


def cvgg19_bn(finding_masks, num_classes, batch_norm=True):
    return VGG(finding_masks, make_layers(cfg['E'], batch_norm=batch_norm, finding_masks=finding_masks), num_classes=num_classes)


if __name__ == "__main__":
    import torch

    inter_feature = []
    def hook(module, input, output):
        inter_feature.append(output.clone().detach())

    input = torch.randn(5, 3, 32, 32)
    model = cvgg11_bn(finding_masks=True, num_classes=10, batch_norm=True)
    model.hook_masks()
    output = model(input)
    # masks_outputs = model.get_masks_outputs()
    # print(masks_outputs['mask.0'].shape)

    # masks = model.get_masks()
    # for key in masks.keys():
    #     print(masks[key].mask)


