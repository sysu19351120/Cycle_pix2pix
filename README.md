# Cycle_pix2pix

运行pix2pix_train.py即可对图片进行训练

训练数据集应当放在同级data文件夹中，将训练数据文件夹命名为train，测试集命名为test

saved_dir下有save_net和result_img两个文件夹分别用来存放保存的模型以及训练过程中测试的图片

（当前项目中已经保存了生成器的一个模型）
# painter
用于绘制测试的图片，请在弹出的窗口中使用鼠标来对图像进行绘制，并且依照指导来保存

# img_process
该文件夹中存放的是使用sdog算子生成训练数据集的文件
请将要处理的文件放入cartoon文件夹中并且运行data_loader.py

# test
-----------------------------------若要进行测试-------------------------------------


将待测试的简笔画图片放置于test_img文件夹中的data文件夹里

运行test.py文件（运行前需要将文件中的net_root改成已经保存的模型的路径）

生成的图片会保存在test_result 文件夹中


一下是我们训练过的模型，包括卡通头像的模型
链接：https://pan.baidu.com/s/1u2OhbGZT6WVClYgYnp9rKw 
提取码：cp53 
将模型下载后放入save_net文件夹中
