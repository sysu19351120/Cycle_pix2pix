# Cycle_pix2pix
运行pix2pix_train.py即可对图片进行训练
训练数据集应当放在同级data文件夹中，将训练数据文件夹命名为train，测试集命名为test
saved_dir下有save_net和result_img两个文件夹分别用来存放保存的模型以及训练过程中测试的图片
（当前项目中已经保存了生成器的一个模型）
-----------------------------------若要进行测试-------------------------------------
将待测试的简笔画图片放置于test_img文件夹zhong
运行test.py文件（运行前需要将文件中的net_root改成已经保存的模型的路径）
生成的图片会保存在test_result 文件夹中
