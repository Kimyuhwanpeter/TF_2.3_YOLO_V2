# -*- coding:utf-8 -*-
from random import shuffle
from absl import flags
from model import *

import tensorflow as tf
import numpy as np
import os
import sys

flags.DEFINE_string("tr_txt_path", "D:/[1]DB/[3]detection_DB/PascalVoc2012/pascal_voc_2012/VOC2012/test", "Training text path")

flags.DEFINE_string("tr_img_path", "D:/[1]DB/[3]detection_DB/PascalVoc2012/pascal_voc_2012/VOC2012/JPEGImages", "Training image path")

flags.DEFINE_integer("img_size", 416, "Image size (Original is 416, but to use pre-trained)")

flags.DEFINE_integer("output_size", 13, "Model output")

flags.DEFINE_integer("batch_size", 10, "Batch size")

flags.DEFINE_integer("num_classes", 20, "Number of classes")

flags.DEFINE_integer("epochs", 100, "Total epochs")

flags.DEFINE_bool("train", True, "True or False")

flags.DEFINE_bool("pre_checkpoint", False, "True or False")

flags.DEFINE_string("pre_checkpoint_path", "", "Saved checkpoint path")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
ANCHORS = np.array(ANCHORS)
ANCHORS = ANCHORS.reshape(len(ANCHORS) // 2, 2)
optim = tf.keras.optimizers.Adam(1e-5)

def func_(image, label):

    list_ = image
    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image, 3)
    shape = tf.shape(image)
    image = tf.image.resize(image, [FLAGS.img_size, FLAGS.img_size])
    image = tf.image.convert_image_dtype(image, tf.float32) / 255

    #image = tf.image.per_image_standardization(image)

    return image, label, shape, list_

def read_label(file, batch_size):
    # https://github.com/jmpap/YOLOV2-Tensorflow-2.0/blob/master/Yolo_V2_tf_eager.ipynb

    anchor_count = 5
    responsibleGrid = np.zeros([FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, 5, 25])

    full_target_grid = []
    for b in range(batch_size):
        f = open(tf.compat.as_bytes(file[b].numpy()), 'r')
        cla = []
        traget_grid = []
        while True:
            line = f.readline()
            if not line: break
            line = line.split('\n')[0]

            xmin = int(int(line.split(',')[0]))
            xmax = int(int(line.split(',')[2]))
            ymin = int(int(line.split(',')[1]))
            ymax = int(int(line.split(',')[3]))
            height = int(line.split(',')[4])
            width = int(line.split(',')[5])
            classes = int(line.split(',')[6])

            x = (xmin + xmax) // 2
            y = (ymin + ymax) // 2
            w = xmax - xmin
            h = ymax - ymin

            x = x * (1/width)
            w = w * (1/width)
            y = y * (1/height)
            h = h * (1/height)

            i, j = int(FLAGS.output_size * y), int(FLAGS.output_size * x)
            x_cell, y_cell = FLAGS.output_size * x - j, FLAGS.output_size * y - i
            width_cell = w * FLAGS.output_size
            height_cell = h * FLAGS.output_size

            boxData = [x_cell, y_cell, width_cell, height_cell]

            responsibleGridX = i
            responsibleGridY = j

            if width_cell * height_cell > 0:
                best_box = 0
                best_anchor = 0
                for i in range(anchor_count):
                    intersect = np.minimum(width_cell, ANCHORS[i, 0]) * np.minimum(height_cell, ANCHORS[i, 1])
                    union = ANCHORS[i, 0] * ANCHORS[i, 1] + (width_cell * height_cell) - intersect
                    iou = intersect / union
                    if iou > best_box:
                        best_box = iou
                        best_anchor = i
                if best_box > 0:
                    responsibleGrid[b][responsibleGridX][responsibleGridY][best_anchor][classes] = 1.    # class
                    responsibleGrid[b][responsibleGridX][responsibleGridY][best_anchor][21:25] = boxData    # box
                    responsibleGrid[b][responsibleGridX][responsibleGridY][best_anchor][FLAGS.num_classes] = 1. # confidence

                traget_grid.append([classes, 1, x, y, w, h])

        full_target_grid.append(traget_grid)

    responsibleGrid = np.array(responsibleGrid, dtype=np.float32)
    traget_grid = np.array(traget_grid, dtype=np.float32)

    return responsibleGrid, full_target_grid

def grid_offset(grid_H, grid_W):
    x = tf.reshape(tf.tile(tf.range(grid_W), [grid_H]),
                   [1, grid_H, grid_W, 1, 1])
    y = tf.reshape(tf.tile(tf.range(grid_H), [grid_W]),
                   [1, grid_W, grid_H, 1, 1])
    y = tf.transpose(y, (0, 2, 1, 3, 4))

    coords = tf.tile(tf.concat([x, y], -1), [FLAGS.batch_size, 1, 1, 5, 1])

    return coords

def IOU(predict_box, label_box):

    box1_x1 = predict_box[..., 0:1] - predict_box[..., 2:3] / 2
    box1_y1 = predict_box[..., 1:2] - predict_box[..., 3:4] / 2
    box1_x2 = predict_box[..., 0:1] + predict_box[..., 2:3] / 2
    box1_y2 = predict_box[..., 1:2] + predict_box[..., 3:4] / 2

    box2_x1 = label_box[..., 0:1] - label_box[..., 2:3] / 2
    box2_y1 = label_box[..., 1:2] - label_box[..., 3:4] / 2
    box2_x2 = label_box[..., 0:1] + label_box[..., 2:3] / 2
    box2_y2 = label_box[..., 1:2] + label_box[..., 3:4] / 2

    intersect_x1 = tf.maximum(box1_x1, box2_x1)
    intersect_y1 = tf.maximum(box1_y1, box2_y1)
    intersect_x2 = tf.maximum(box1_x2, box2_x2)
    intersect_y2 = tf.maximum(box1_y2, box2_y2)
    intersect_x1 = intersect_x1.numpy()
    intersect_y1 = intersect_y1.numpy()
    intersect_x2 = intersect_x2.numpy()
    intersect_y2 = intersect_y2.numpy()

    intersect = (intersect_x2 - intersect_x1).clip(0) * (intersect_y2 - intersect_y1).clip(0)

    box1_area = tf.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = tf.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    iou = intersect / (box1_area + box2_area - intersect + 1e-6)

    return iou

@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def cal_loss(model, images, labels):
    with tf.GradientTape() as tape:
        # 라벨 shape을 다시 맞추고 아래 코드들 다시 수정하자!!! 기억해!!
        # 조금만 더 고치면 될 것 같다 (조금만더 힘내고 기억해!!!!!!!!!!!!!!!)
        logits = run_model(model, images, True)
        offsets = tf.cast(grid_offset(FLAGS.output_size, FLAGS.output_size), tf.float32)
        pred = tf.reshape(logits,
                          [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, 5, 4+1+FLAGS.num_classes])
        I_obj = labels[:, :, :, :, 20]
        nI_obj = tf.keras.backend.sum(tf.cast(I_obj > 0.0, tf.float32))
        I_obj = tf.expand_dims(I_obj, -1)
        predict_xy = tf.nn.sigmoid(pred[:, :, :, :, 0:2])
        predict_xy = predict_xy + offsets
        predict_wh = tf.exp(pred[:, :, :, :, 2:4]) * ANCHORS
        predict_box = tf.concat([predict_xy, predict_wh], -1)
        
        object_conf = tf.nn.sigmoid(pred[:, :, :, :, 4:5])

        ####################################################################################
        # 박스 좌표 loss
        xy_loss = 1 * tf.keras.backend.sum(I_obj * tf.square(labels[:, :, :, :, 21:23] - predict_xy)) / (nI_obj + 1e-7)
        wh_loss = 1 * tf.keras.backend.sum(I_obj * tf.square(
            tf.sqrt(labels[:, :, :, :, 23:25]) - tf.sqrt(predict_wh))) / (nI_obj + 1e-7)
        coord_loss = xy_loss + wh_loss  # 논문의 첫 번째 수식
        ####################################################################################

        ####################################################################################
        # obect loss (오브젝트가 있을 때의 loss)
        iou = IOU(predict_box, labels[:, :, :, :, 21:25])
        best_box = tf.keras.backend.max(iou, 3)
        object_loss = 5 * tf.keras.backend.sum(I_obj * tf.square(iou-object_conf)) / (nI_obj + 1e-7)
        ####################################################################################

        ####################################################################################
        # no object loss (오브젝트가 없을 때의 loss)
        no_I_obj = tf.cast(best_box < 0.6, tf.float32)
        no_I_obj = tf.expand_dims(no_I_obj, -1)
        no_I_obj = no_I_obj * (1 - I_obj)
        no_I_obj = tf.keras.backend.sum(tf.cast(no_I_obj > 0.0, tf.float32))
        no_object_loss = 1 * tf.keras.backend.sum(no_I_obj * tf.square(-object_conf)) / (nI_obj + 1e-7)
        ####################################################################################

        ####################################################################################
        # class loss
        pred_class = pred[:, :, :, :, 5:]   # [B, 13, 13, 5, 20]
        true_class = labels[:, :, :, :, :20]   # [B, 13, 13, 5, 20]
        class_loss = tf.keras.losses.categorical_crossentropy(true_class,
                                                              pred_class,
                                                              from_logits=True)
        class_loss = tf.expand_dims(class_loss, -1) * I_obj
        class_loss = 1 * tf.keras.backend.sum(class_loss) / (nI_obj + 1e-7)
        ####################################################################################

        total_loss = coord_loss + object_loss + no_object_loss + class_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss

def main():
    model = Yolo_V2()
    model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored the checkpoint!!")

    if FLAGS.train:
        count = 0
        text_list = os.listdir(FLAGS.tr_txt_path)
        text_list = [FLAGS.tr_txt_path + '/' + data for data in text_list]

        image_list = os.listdir(FLAGS.tr_img_path)
        image_list = [FLAGS.tr_img_path + '/' + data for data in image_list]

        A = list(zip(image_list, text_list))
        shuffle(A)
        image_list, text_list = zip(*A)

        image_list = np.array(image_list)
        text_list = np.array(text_list)

        for epoch in range(FLAGS.epochs):

            data = tf.data.Dataset.from_tensor_slices((image_list, text_list))
            data = data.shuffle(len(text_list))
            data = data.map(func_)
            data = data.batch(FLAGS.batch_size)
            data = data.prefetch(tf.data.experimental.AUTOTUNE)

            batch_idx = len(text_list) // FLAGS.batch_size
            it = iter(data)
            for step in range(batch_idx):
                image, label, shape, list_ = next(it)
                original_height, original_width = shape[:, 0], shape[:, 1]
                tr_label, target_label = read_label(label, FLAGS.batch_size)
                tr_label = tf.convert_to_tensor(tr_label)

                loss = cal_loss(model, image, tr_label)
                
                #a = cal_mAP(predict_boxes, target_label)    # Not implement

                if count % 10 == 0:
                    print("Epoch: {} [{}/{}] loss = {}".format(epoch, step + 1, batch_idx, loss))

                count += 1

if __name__ == "__main__":
    main()