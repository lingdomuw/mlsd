#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
import os
import numpy as np
import tensorflow as tf
import time
import tqdm
from modules.models import WireFrameModel
# 模型的checkpoint文件地址
'''
--model_path=./ckpt_models/M-LSD_512_tiny \
--model_tflite_path=./tflite_models/M-LSD_512_tiny_fp32.tflite \
--input_size=512 \
--map_size=256 \
--batch_size=1 \
--dilate=5 \
--with_alpha=True \
--backbone_type=MLSD \
--topk=200 \
--fp16=False
'''


flags.DEFINE_string('model_path', './ckpt_models/M-LSD_512_tiny', 'path to save folder')
flags.DEFINE_string('model_tflite_path', './tflite_models/M-LSD_512_tiny_fp32.tflite', 'path_to_save_tflite_model')
flags.DEFINE_boolean('with_alpha', True, 'whether support RGBA image')
flags.DEFINE_boolean('fp16', False, '')

# input images
flags.DEFINE_integer('batch_size', 1, 'size of input batch')
flags.DEFINE_integer('input_size', 512, 'size of input image')
flags.DEFINE_integer('map_size', 256, 'size of lmap, jmap, and joff')

# encoder
flags.DEFINE_string('backbone_type', 'MLSD', 'MLSD | MLSD_large')
flags.DEFINE_boolean('pretrain', True, 'whether use imagenet pretrained weights')
flags.DEFINE_integer('out_channel', 256, 'n_channels of output encoded spatial features')
flags.DEFINE_integer('dilate', 5, 'dilation rate')
flags.DEFINE_boolean('final_last', False, '')
flags.DEFINE_boolean('final_act', True, '')
flags.DEFINE_boolean('final_res1', False, '')
flags.DEFINE_boolean('final_res2', False, '')
flags.DEFINE_integer('residual_type', 0, '')
flags.DEFINE_string('post_name', '_extractor', '_extractor | _extrator')
flags.DEFINE_integer('type_a_ksize', 1, 'type_a_ksize')

# decoder
flags.DEFINE_integer('topk', 200, 'topk')
flags.DEFINE_boolean('final_padding_same', True, '')
flags.DEFINE_float('center_thr', 0.001, 'weight for loss_center_map')

flags.DEFINE_float('wd', 0.0001, 'weight decay value')



def main(_):
    # initialize systems
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    cfg = FLAGS # I love FLAGS!!!
    
    # define network
    model = WireFrameModel(cfg, training=False)
    model.summary(line_length=80)

    # load checkpoint
    checkpoint_dir = cfg.model_path
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name='step'),
                                     model=model)
    manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                         directory=checkpoint_dir,
                                         max_to_keep=3)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print('[*] load ckpt from {} at step {}.'.format(
            manager.latest_checkpoint, checkpoint.step.numpy()))
    else:
        print("[*] training from imagenet pretrained weights.")
        
    '''
    input: RGB image
    output: [center points, center scores, displacement vector map]
    '''
    model = tf.keras.Model(model.input, [model.output[-6], model.output[-5], model.output[-7]], name='WireFrameModel')
    input1 = tf.constant(np.random.rand(3,cfg.input_size,cfg.input_size,3), dtype=tf.float32)
    output1 = model(input1)
    model.save("my_model")

    
if __name__ == '__main__': 
    app.run(main)


# In[ ]:




