from torch.autograd import Variable
from options import TrainOptions
from optimizer import *
from torch.utils.data import DataLoader
from datasets import ImageDataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
from PIL import ImageFilter
import torchvision.datasets as datasets
#保存要测试的图片的路径
test_data="./test_img"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#加载参数
args = TrainOptions().parse()
D_out_size = args.img_height//(2**args.n_D_layers) - 2
patch = (1, D_out_size, D_out_size)
#加载保存好的模型

generatorA2B = torch.load("./save_dir/saved_net/generatorA2B_45.pkl",map_location=device)
# discriminatorA = torch.load("./save_dir/saved_net/discriminatorA_35.pkl",map_location='cpu')
transforms_ = transforms.Compose([transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_imgs=os.listdir(test_data+"/data")
# batch=len(test_imgs)
batch=10
for i in test_imgs:
    img=Image.open(test_data+"/data/"+i)
    img = img.convert('L')
    img=img.resize((256,256))
    img.save(test_data+"/data/"+i)

test_dataloader = DataLoader(
    datasets.ImageFolder(root=test_data,transform=transforms_),
    batch_size=batch, shuffle=True, num_workers=0)
for i,data in enumerate(test_dataloader,0):

    fake_B = generatorA2B(data[0])
    # 按照从上到下的顺序拼接图片
    img_sample = torch.cat((data[0].data, fake_B.data), -2)
    save_image(img_sample, 'test_result/test_result{}.png'.format(i), nrow=5, normalize=True)
