from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans


#将图像归一化
def adjustData(img,mask):
    img = img / 255         #归一化
    mask = mask /255
    mask[mask > 0.5] = 1    #二值化
    mask[mask <= 0.5] = 0
    return (img,mask)

#生成训练数据
def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    #图像生成器对数据进行增强 扩大数据集大小，增强模型泛化能力。
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,                         #目标文件夹路径
        classes = [image_folder],           #子文件夹路径
        class_mode = None,                  #确定返回标签数组的类型
        color_mode = image_color_mode,      #颜色模式，rgb三通道，grayscale灰度图
        target_size = target_size,          #目标图片大小
        batch_size = batch_size,            #每一批的图像数量
        save_to_dir = save_to_dir,          #保存路径
        save_prefix  = image_save_prefix,   #保存图片的前缀
        seed = seed)                        #随机种子，用于shuffle
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator) #将image和mask的训练器打包成元组同时进行增强
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask)
        yield (img,mask)


#生成测试集
def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape) #(1,width,heigth)
        yield img


#保存预测的数据
def saveResult(save_path,npyfile):
    for i,item in enumerate(npyfile):
        img = item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)