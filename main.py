from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#数据增强字典
data_gen_args = dict(rotation_range=0.2,    #旋转
                    width_shift_range=0.05, #宽度变化
                    height_shift_range=0.05,#高度变化
                    shear_range=0.05,       #剪切变换
                    zoom_range=0.05,        #缩放
                    horizontal_flip=True,   #水平反转
                    fill_mode='nearest')    #填充模式


myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)

model = unet()

model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
#在每个epoch后保存模型到filepath
#filepath为保存的路径
#monitor 需要监视的值
#verbose 是否展示信息 1 展示
#save_best_only 只保存最好的模型

model.fit_generator(myGene,steps_per_epoch=300,epochs=5,callbacks=[model_checkpoint])
#myGene  训练的数据 这里传入的是一个生成器 每次返回一个（image，mask）
#steps_per_epoch 每轮的训练数量
#epoch 训练多少论次
#callback 回调函数 用于查询模型的内在状态和统计信息


testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene,30,verbose=1)
#generator 生成器
#steps 在声明一个 epoch 完成并开始下一个 epoch 之前从 generator 产生的总步数（批次样本）。 它通常应该等于你的数据集的样本数量除以批量大小。 对于 Sequence，它是可选的：如果未指定，将使用len(generator) 作为步数。
#workers: 整数。使用的最大进程数量，如果使用基于进程的多线程。 如未指定，workers 将默认为 1。如果为 0，将在主线程上执行生成器。
#verbose: 日志显示模式，0 或 1。
saveResult("data/membrane/test",results)