from options import TrainOptions
from optimizer import *
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
from PIL import ImageFilter
import torchvision.datasets as datasets
"""该部分主要用于结果的测试，即输入简笔画生成对应的图像"""
#保存要测试的图片的路径
test_data="./test_img"
result_dir="./saved"
test_result="./test_result"
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
if not os.path.exists(test_result):
    os.mkdir(test_result)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#加载参数
args = TrainOptions().parse()
D_out_size = args.img_height//(2**args.n_D_layers) - 2
patch = (1, D_out_size, D_out_size)
#加载保存好的模型
generatorA2B = torch.load("./save_dir/saved_net/cartoon/generatorA2B_54.pkl",map_location=device)
# generatorA2B = torch.load("./save_dir/saved_net/human_face/generatorA2B_199.pkl",map_location=device)
# generatorA2B = torch.load("./save_dir/saved_net/car/generatorA2B_62.pkl",map_location=device)

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
    # for ii in range(fake_B.shape[0]):
    #     save_image(fake_B[ii,:,:,:].data,result_dir+"/result_{}_{}.jpg".format(i,ii),normalize=True)
    save_image(img_sample, test_result+'/test_result{}.png'.format(i), nrow=5, normalize=True)
