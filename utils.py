import torch
from torch.autograd import Variable
from torchvision.utils import save_image
from torch import random
def sample_images(generator, test_dataloader, args, epoch,i):
    """用于保存img"""
    imgs = next(iter(test_dataloader))
    if torch.cuda.is_available():
        real_A = Variable(imgs['A'].type(torch.FloatTensor).cuda())
        real_B = Variable(imgs['B'].type(torch.FloatTensor).cuda())
    else:
        real_A = Variable(imgs['A'].type(torch.FloatTensor))
        real_B = Variable(imgs['B'].type(torch.FloatTensor))
    nz=torch.randn(real_A.shape[0],1)
    fake_B = generator(nz,real_A)
    #按照从上到下的顺序拼接图片
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(img_sample, '%s/%s/%s_%s.png' % (args.save_dir, args.img_result_dir,  epoch,i), nrow=5, normalize=True)
class ReplayBuffer:
    """
    缓存队列，若不足则新增，否则随机替换
    """

    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))