from kornia.filters.median import MedianBlur

from models.layers import *

cfg = {
    'vgg5' : [[64, 'A'], 
              [128, 128, 'A'],
              [],
              [],
              []],
    'vgg11': [
        [64, 'A'],
        [128, 256, 'A'],
        [512, 512, 512, 'A'],
        [512, 512],
        []
    ],
    'vgg13': [
        [64, 64, 'A'],
        [128, 128, 'A'],
        [256, 256, 'A'],
        [512, 512, 'A'],
        [512, 512, 'A']
    ],
    'vgg16': [
        [64, 64, 'A'],
        [128, 128, 'A'],
        [256, 256, 256, 'A'],
        [512, 512, 512, 'A'],
        [512, 512, 512, 'A']
    ],
    'vgg19': [
        [64, 64, 'A'],
        [128, 128, 'A'],
        [256, 256, 256, 256, 'A'],
        [512, 512, 512, 512, 'A'],
        [512, 512, 512, 512, 'A']
    ],
    'vggdvs': [
        [64, 128, 'A'],
        [256, 256, 'A'],
        [512, 512, 'A'],
        [512, 512, 'A'],
        []
    ],
    'vgggesture': [
        [16, 32, 'A'],
        [32, 32, 'A'],
        [],
        [],
        []
    ]
}

class VGG(nn.Module):
    def __init__(self, vgg_name, T, num_class, norm, tau=1., init_c=3):
        super(VGG, self).__init__()
        if norm is not None and isinstance(norm, tuple):
            self.norm = TensorNormalization(*norm)
        else:
            self.norm = TensorNormalization((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        self.T = T
        self.tau = tau
        self.init_channels = init_c

        if vgg_name == 'vgg11' or vgg_name == 'vgg5':
            self.W = 16
        else:
            self.W = 1
        
        self.layer1 = self._make_layers(cfg[vgg_name][0])
        self.layer2 = self._make_layers(cfg[vgg_name][1])
        self.layer3 = self._make_layers(cfg[vgg_name][2])
        self.layer4 = self._make_layers(cfg[vgg_name][3])
        self.layer5 = self._make_layers(cfg[vgg_name][4])
        self.classifier = self._make_classifier(num_class)
        
        self.merge = MergeTemporalDim(T)
        self.expand = ExpandTemporalDim(T)
        self.encode = Poisson()
        self.poisson = False
        self.hooked = None
        # self.blur = MedianBlur((3,3))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def hook(self, model, gradin, gradout):
        self.hooked = gradin[0]

    def act_hook(self, model, input, output):
        x = input[0]
        if self.T > 0:
            x = self.expand(x)
            x = x.mean(0)
        self.hooked = x

    def _make_layers(self, cfg):
        layers = []
        for x in cfg:
            if x == 'A':
                layers.append(nn.AvgPool2d(2))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(x))
                layers.append(LIFSpike(T=self.T, tau=self.tau))
                self.init_channels = x
        return nn.Sequential(*layers)

    def _make_classifier(self, num_class):
        layer = [
        nn.Flatten(),
        nn.Linear(512*self.W, 4096),
        LIFSpike(self.T, tau=self.tau),
        nn.Linear(4096, 4096),
        LIFSpike(self.T, tau=self.tau), 
        nn.Linear(4096, num_class)
        ]
        return nn.Sequential(*layer)
    
    # pass T to determine whether it is an ANN or SNN
    def set_simulation_time(self, T, mode='bptt'):
        self.T = T
        for module in self.modules():
            if isinstance(module, (LIFSpike, ExpandTemporalDim)):
                module.T = T
                if isinstance(module, LIFSpike):
                    module.mode = mode
        return

    def forward(self, input):
        if not self.poisson:
            input = self.norm(input)
        # input = self.blur(input)
        if self.T > 0:
            input = add_dimention(input, self.T)
            if self.poisson:
                input = self.encode(input)
            input = self.merge(input)
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.classifier(out)
        if self.T > 0:
            out = self.expand(out)
        return out

class VGGDVS(nn.Module):
    def __init__(self, vgg_name, T, num_class, norm, tau=1., init_c=2, init_s=48):
        super(VGGDVS, self).__init__()
        self.T = T
        self.tau = tau
        self.init_channels = init_c
        
        cnt = 0
        for l in cfg[vgg_name]:
            if len(l)>0:
                cnt += 1
        
        self.W = int(init_s/(1<<cnt))**2

        self.layer1 = self._make_layers(cfg[vgg_name][0])
        self.layer2 = self._make_layers(cfg[vgg_name][1])
        self.layer3 = self._make_layers(cfg[vgg_name][2])
        self.layer4 = self._make_layers(cfg[vgg_name][3])
        self.classifier = self._make_classifier(num_class, cfg[vgg_name][cnt-1][1])
        
        self.merge = MergeTemporalDim(T)
        self.expand = ExpandTemporalDim(T)
        self.poisson = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layers(self, cfg):
        layers = []
        for x in cfg:
            if x == 'A':
                layers.append(nn.AvgPool2d(2))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(x))
                layers.append(LIFSpike(T=self.T, tau=self.tau))
                self.init_channels = x
        return nn.Sequential(*layers)

    def _make_classifier(self, num_class, channels):
        layer = [nn.Linear(channels*self.W, num_class)]
        return nn.Sequential(*layer)
    
    # pass T to determine whether it is an ANN or SNN
    def set_simulation_time(self, T, mode='bptt'):
        for module in self.modules():
            if isinstance(module, (LIFSpike, ExpandTemporalDim)):
                if isinstance(module, LIFSpike):
                    module.mode = mode
        return

    def forward(self, input):
        input = self.merge(input)
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        out = self.expand(out)
        return out

class Print(nn.Module):
    def __init__(self) -> None:
        super(Print, self).__init__()
    def forward(self, x):
        print(x.shape)
        return x


class CNNDVS(nn.Module):
    def __init__(self, T, num_class, tau=1., init_c=2, init_s=48):
        super(CNNDVS, self).__init__()
        self.T = T
        self.tau = tau
        self.init_channels = init_c

        self.merge = MergeTemporalDim(T)
        self.expand = ExpandTemporalDim(T)
        self.poisson = False

        self.w = int(int((int((init_s+2-5)/2)+1)/2)/2)

        self.layer = nn.Sequential(
            nn.Conv2d(2, 16, 5, 2, 1),
            # Print(),
            nn.BatchNorm2d(16, ),
            LIFSpike(T=self.T, tau=self.tau),
            nn.AvgPool2d(2, 2),
            LIFSpike(T=self.T, tau=self.tau),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32, ),
            LIFSpike(T=self.T, tau=self.tau),

            nn.AvgPool2d(2, 2),
            LIFSpike(T=self.T, tau=self.tau),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * self.w * self.w, 512),
            LIFSpike(T=self.T, tau=self.tau),
            nn.Linear(512, num_class)
        )

    # pass T to determine whether it is an ANN or SNN
    def set_simulation_time(self, T, mode='bptt'):
        for module in self.modules():
            if isinstance(module, (LIFSpike, ExpandTemporalDim)):
                if isinstance(module, LIFSpike):
                    module.mode = mode
        return

    def forward(self, x: torch.Tensor):
        x = self.merge(x)
        x = self.layer(x)
        x = self.classifier(x)
        x = self.expand(x)
        return x
        


    