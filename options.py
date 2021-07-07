import argparse
import os
"""用于保存模型训练过程中所需要的各种参数"""
class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--save_dir', type=str, default="save_dir", help='the name of the experiment')
        self.parser.add_argument('--epoch_start', type=int, default=0, help='epoch to start training from')
        self.parser.add_argument('--epoch_num', type=int, default=200, help='number of epochs of training')
        self.parser.add_argument('--data_root', type=str, default='./data/', help='dir of the dataset')
        self.parser.add_argument('--dataset_name', type=str, default="facades", help='name of the dataset')
        self.parser.add_argument('--batch_size', type=int, default=4, help='size of the batches')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
        self.parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
        self.parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
        self.parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
        self.parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
        self.parser.add_argument('--img_height', type=int, default=64, help='size of image height')
        self.parser.add_argument('--img_width', type=int, default=64, help='size of image width')
        self.parser.add_argument('--in_channels', type=int, default=3, help='number of input image channels')
        self.parser.add_argument('--out_channels', type=int, default=3, help='number of output image channels')
        self.parser.add_argument('--sample_interval', type=int, default=200, help='interval between sampling of images from generators')
        self.parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--n_D_layers', type=int, default=3, help='used to decision the patch_size in D-net, should less than 8')
        self.parser.add_argument('--lambda_pixel', type=int, default=100, help=' Loss weight of L1 pixel-wise loss between translated image and real image')
        self.parser.add_argument('--img_result_dir', type=str, default='result_images', help=' where to save the result images')
        self.parser.add_argument('--model_result_dir', type=str, default='saved_net', help=' where to save the checkpoints')


    def parse(self):
        if not self.initialized:
            self.initialize()
        args = self.parser.parse_args()

        os.makedirs('%s/%s' % (args.save_dir, args.img_result_dir), exist_ok=True)
        os.makedirs('%s/%s' % (args.save_dir, args.model_result_dir), exist_ok=True)
        self.args = args
        return self.args
