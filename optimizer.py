import torch
# Optimizers
def Get_optimizers(args, generator, discriminator):
    optimizer_G = torch.optim.Adam(
                    generator.parameters(),
                    lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(
                    discriminator.parameters(),
                    lr=args.lr, betas=(args.b1, args.b2))

    return optimizer_G, optimizer_D
# Loss functions"""unet的下采样部分"""
def Get_loss_func(device):
    """unet的下采样部分"""
    criterion_GAN = torch.nn.BCELoss()
    criterion_pixelwise = torch.nn.L1Loss()

    criterion_GAN.to(device)
    criterion_pixelwise.to(device)
    return criterion_GAN, criterion_pixelwise
