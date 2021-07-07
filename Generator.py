import torch.nn as nn
import torch

from torchvision.models import resnet18
class Generator(nn.Module):
    def __init__(self, ngpu, nz=100, ngf=64, nc=3):
        """
        :param ngpu: number of gpu
        :param nz: input size
        :param ngf: base number of output feature map
        :param nc: number of channels of image
        """
    
        super(Generator, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self
        # self.resnet=RestNet18().to(self.device)
        self.resnet=resnet18()
        self.resnet.load_state_dict(torch.load("./resnet18-5c106cde.pth"))
        self.c1=nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1)
        self.ngpu = ngpu
        self.nz1=nn.ConvTranspose2d(nz, ngf * 4, (4,8), 1, 0, bias=False)
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # nn.ConvTranspose2d(nz, ngf * 8, (4,8), 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1,bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 128
        )
        
    def forward(self, inputz,inputx):
        inputz=self.nz1(inputz)
        inputx=self.resnet.conv1(inputx)
        inputx = self.resnet.layer1(inputx)
        inputx = self.resnet.layer2(inputx)
        inputx = self.resnet.layer3(inputx)
        inputx = self.resnet.layer4(inputx)
        inputx=self.c1(inputx)
        print(inputx.shape)
        input=torch.cat((inputz,inputx),1)
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
net=Generator(0)
a=torch.randn(1,100,1,1)
x=torch.randn(1,3,64,128)
d=net(a,x)