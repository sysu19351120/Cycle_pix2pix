import argparse

import numpy as np
import time

import sys
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import itertools
from datasets import Get_dataloader
from options import TrainOptions
from optimizer import *
from utils import sample_images,ReplayBuffer


lossd=[]
lossg=[]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#加载参数
args = TrainOptions().parse()
D_out_size = args.img_height//(2**args.n_D_layers) - 2
patch = (1, D_out_size, D_out_size)
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()
# 初始化生成器和判别器
generatorA2B, discriminatorA = Create_nets(args,device)
generatorB2A, discriminatorB = Create_nets(args,device)
# generatorA2B=torch.load("./save_dir/saved_net/generatorA2B_56.pkl")
# discriminatorA=torch.load("./save_dir/saved_net/discriminatorA_56.pkl")
# generatorB2A=torch.load("./save_dir/saved_net/generatorB2A_56.pkl")
# discriminatorB=torch.load("./save_dir/saved_net/discriminatorB_56.pkl")
# 生成损失函数
criterion_GAN, criterion_pixelwise = Get_loss_func(device)
# 得到优化器Optimizers
_, optimizer_DA = Get_optimizers(args, generatorA2B, discriminatorA)
_, optimizer_DB = Get_optimizers(args, generatorB2A, discriminatorB)
optimizer_G = torch.optim.Adam(
    itertools.chain(generatorA2B.parameters(), generatorB2A.parameters()),
    lr=args.lr, betas=(args.b1, args.b2))
# 生成dataloader
train_dataloader,test_dataloader = Get_dataloader(args)


# ----------
#  训练部分
# ----------
prev_time = time.time()
#Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
for epoch in range(args.epoch_start, args.epoch_num):
    for i, batch in enumerate(train_dataloader):

        if torch.cuda.is_available():
            real_A = Variable(batch['A'].type(torch.FloatTensor).cuda())
            real_B = Variable(batch['B'].type(torch.FloatTensor).cuda())
            valid = Variable(torch.FloatTensor(np.ones((real_A.size(0), *patch))).cuda(), requires_grad=False)
            fake = Variable(torch.FloatTensor(np.zeros((real_A.size(0), *patch))).cuda(), requires_grad=False)
        else:
            real_A = Variable(batch['A'].type(torch.FloatTensor))
            real_B = Variable(batch['B'].type(torch.FloatTensor))
            # Adversarial ground truths
            valid = Variable(torch.FloatTensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
            fake = Variable(torch.FloatTensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)
        print(real_A.shape)
        inA=torch.cat((real_A,real_B),3)
        inB=torch.cat((real_B,real_A),3)
        print(inA.shape)
        nz1=torch.randn(real_A.shape[0],100,1,1)
        nz2 = torch.randn(real_A.shape[0], 100, 1, 1)
        optimizer_G.zero_grad()
        sameB=generatorA2B(nz1,inB)
        loss_identity_B=criterion_pixelwise(sameB[:,:,:,:64],real_B)
        sameA=generatorB2A(nz1,inA)
        loss_identity_A=criterion_pixelwise(sameA[:,:,:,:64],real_A)

        fakeB=generatorA2B(nz1,inA)
        pred_fake=discriminatorB(fakeB[:,:,:,:64],real_A)
        loss_GAN_A2B=criterion_GAN(pred_fake,valid)
        pixel_GAN_A2B = criterion_pixelwise(fakeB[:,:,:,:64], real_B)*100.0
        fakeA=generatorB2A(nz2,inB)
        pred_fake = discriminatorB(fakeA[:,:,:,:64],real_B)
        loss_GAN_B2A = criterion_GAN(pred_fake,valid)
        pixel_GAN_B2A=criterion_pixelwise(fakeA[:,:,:,:64],real_A)*100.0

        recovered_A = generatorB2A(nz2,fakeB)
        loss_cycle_ABA = criterion_pixelwise(recovered_A[:,:,:,:64], real_A)
        recovered_B = generatorA2B(nz1,fakeA)
        loss_cycle_BAB = criterion_pixelwise(recovered_B[:,:,:,:64], real_B)
        loss_G = pixel_GAN_A2B+pixel_GAN_B2A+loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        optimizer_G.step()
        #训练判别器
        #A
        optimizer_DA.zero_grad()
        pred_real = discriminatorA(real_A,real_B)
        loss_D_real = criterion_GAN(pred_real, valid)
        # fake_A1 = fake_A_buffer.push_and_pop(fakeA)
        pred_fake = discriminatorA(fakeA[:,:,:,:64].detach(),real_B)
        loss_D_fake = criterion_GAN(pred_fake, fake)
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()
        optimizer_DA.step()
        #B
        optimizer_DB.zero_grad()
        pred_real = discriminatorB(real_B,real_A)
        loss_D_real = criterion_GAN(pred_real, valid)
        # fake_B1 = fake_B_buffer.push_and_pop(fakeB)
        pred_fake = discriminatorB(fakeB[:,:,:,:64].detach(),real_A)
        loss_D_fake = criterion_GAN(pred_fake, fake)
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()
        optimizer_DB.step()
        # 显示当前运行信息
        sys.stdout.write("\r[Epoch%d/%d]-[Batch%d/%d]-[DlossA:%f]-[Gloss:%f]-[DlossB:%f]-[Gloss:%f] " %
                                                        (epoch+1, args.epoch_num,
                                                        i, len(train_dataloader),
                                                        loss_D_A.item(), loss_G.item(),loss_D_B.item(), loss_G.item()))

        # 每次保存一次test的结果
        if(i%10==0):
            sample_images(generatorA2B, test_dataloader, args, epoch,i)


    # 每一个epoch保存一次图片
    torch.save(generatorA2B, './save_dir/saved_net/generatorA2B_{}.pkl' .format(epoch))
    torch.save(discriminatorA, './save_dir/saved_net/discriminatorA_{}.pkl' .format(epoch))
    torch.save(generatorB2A, './save_dir/saved_net/generatorB2A_{}.pkl' .format(epoch))
    torch.save(discriminatorB, './save_dir/saved_net/discriminatorB_{}.pkl' .format(epoch))
#保存loss的信息
with open('visualize/Loss.txt','wt') as f:
    print(lossd, file=f)
    print('\n', file=f)
    print(lossg, file=f)

D, = plt.plot(lossd, color='r', label='Loss_D')
G, = plt.plot(lossg, color='b', label='Loss_G')
plt.legend()
plt.savefig('visualize/GAN_Loss.png', dpi=200)
