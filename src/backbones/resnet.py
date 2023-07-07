import torch
import torch.nn as nn 


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.prelu1 = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.prelu2 = nn.PReLU(planes)

        if stride == 1:
            self.downsample = None 
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=stride, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes, eps=1e-05,)
            )

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        out = self.prelu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.prelu2(out)
        return out


class ResNet(nn.Module):
    def __init__(self, layers, input_size, num_features=512, fp16=False):
        super(ResNet, self).__init__()
        self.fp16 = fp16

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-05)
        self.prelu1 = nn.PReLU(32)

        self.layer1 = self._make_layer(32, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(256, 512, layers[3], stride=2)

        self.gdc = nn.Sequential(
            nn.Conv2d(512, 512, groups=512, kernel_size=(input_size[0]//16, input_size[1]//16), stride=1, padding=0, bias=False), 
            nn.BatchNorm2d(512, eps=1e-05),
        )
        
        self.features = nn.Sequential(
            nn.Linear(in_features=512, out_features=num_features, bias=False),
            nn.BatchNorm1d(num_features=num_features, eps=2e-5)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, inplanes, planes, blocks, stride=1):
        layers = [BasicBlock(inplanes, planes, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.prelu1(self.bn1(self.conv1(x)))

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        
            x = self.gdc(x)
            x = x.flatten(1)
            features = self.features(x)

        if self.fp16: features = features.float()
        features = nn.functional.normalize(features, dim=-1, p=2)

        return features 
 