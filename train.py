# -*- coding:utf-8 -*-
from random import shuffle
from absl import flags
from model import *
from keras_radam.training import RAdamOptimizer

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys

flags.DEFINE_string("tr_txt_path", "D:/[1]DB/[3]detection_DB/PascalVoc2012/pascal_voc_2012/VOC2012/test", "Training text path")

flags.DEFINE_string("tr_img_path", "D:/[1]DB/[3]detection_DB/PascalVoc2012/pascal_voc_2012/VOC2012/JPEGImages", "Training image path")

flags.DEFINE_integer("img_size", 416, "Image size (Original is 416, but to use pre-trained)")

flags.DEFINE_integer("output_size", 13, "Model output")

flags.DEFINE_integer("batch_size", 10, "Batch size")

flags.DEFINE_integer("num_classes", 20, "Number of classes")

flags.DEFINE_integer("epochs", 1000, "Total epochs")

flags.DEFINE_bool("train", True, "True or False")

flags.DEFINE_bool("pre_checkpoint", False, "True or False")

flags.DEFINE_string("pre_checkpoint_path", "", "Saved checkpoint path")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
ANCHORS = np.array(ANCHORS)

ANCHORS_box = ANCHORS.reshape(len(ANCHORS) // 2, 2)
ANCHORS = np.reshape(ANCHORS, [1,1,1,5,2])

optim = RAdamOptimizer(1e-5)

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

    detector_mask = np.zeros((FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, anchor_count, 1))
    matching_true_boxes = np.zeros((FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, anchor_count, 5))

    full_target_grid = []
    for b in range(batch_size):
        f = open(tf.compat.as_bytes(file[b].numpy()), 'r')
        cla = []
        traget_grid = []
        while True:
            line = f.readline()
            if not line: break
            line = line.split('\n')[0]
            # 이 부분을 고쳐야 하나?
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

            i, j = int(FLAGS.output_size * x), int(FLAGS.output_size * y)
            x_cell, y_cell = FLAGS.output_size * x - i, FLAGS.output_size * y - j
            width_cell = w * FLAGS.output_size
            height_cell = h * FLAGS.output_size

            boxData = [x_cell, y_cell, width_cell, height_cell]

            responsibleGridX = i
            responsibleGridY = j

            if width_cell * height_cell > 0:
                best_box = 0
                best_anchor = 0
                for i in range(anchor_count):
                    intersect = np.minimum(width_cell, ANCHORS_box[i, 0]) * np.minimum(height_cell, ANCHORS_box[i, 1])
                    union = ANCHORS_box[i, 0] * ANCHORS_box[i, 1] + (width_cell * height_cell) - intersect
                    iou = intersect / union
                    if iou > best_box:
                        best_box = iou
                        best_anchor = i
                if best_box > 0:
                    #detector_mask[b, responsibleGridX, responsibleGridY, best_anchor] = 1
                    #yolo_box = np.array([x_cell, y_cell, width_cell, height_cell, ])
                    # 이 부분 고쳐야한다.

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

    box1_x1 = predict_box[..., 0:1] - predict_box[..., 2:3] / 2 # xmin1
    box1_y1 = predict_box[..., 1:2] - predict_box[..., 3:4] / 2 # ymin1
    box1_x2 = predict_box[..., 0:1] + predict_box[..., 2:3] / 2 # xmax1
    box1_y2 = predict_box[..., 1:2] + predict_box[..., 3:4] / 2 # ymax1

    box2_x1 = label_box[..., 0:1] - label_box[..., 2:3] / 2 # xmin2
    box2_y1 = label_box[..., 1:2] - label_box[..., 3:4] / 2 # ymin2
    box2_x2 = label_box[..., 0:1] + label_box[..., 2:3] / 2 # xmax2
    box2_y2 = label_box[..., 1:2] + label_box[..., 3:4] / 2 # ymax2

    intersect_x1 = tf.maximum(box1_x1, box2_x1)
    intersect_y1 = tf.maximum(box1_y1, box2_y1)
    intersect_x2 = tf.minimum(box1_x2, box2_x2)
    intersect_y2 = tf.minimum(box1_y2, box2_y2)
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

        logits = run_model(model, images, True)
        offsets = tf.cast(grid_offset(FLAGS.output_size, FLAGS.output_size), tf.float32)
        pred = tf.reshape(logits,
                          [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, 5, 4+1+FLAGS.num_classes])
        I_obj = labels[:, :, :, :, 20]
        I_obj = tf.expand_dims(I_obj, -1)
        nI_obj = tf.reduce_sum(tf.cast(I_obj > 0.0, tf.float32))
        predict_xy = tf.nn.sigmoid(pred[:, :, :, :, 0:2])
        predict_xy = predict_xy + offsets
        predict_wh = tf.exp(pred[:, :, :, :, 2:4]) * ANCHORS
        predict_box = tf.concat([predict_xy, predict_wh], -1)
        
        object_conf = tf.nn.sigmoid(pred[:, :, :, :, 4:5])
        # loss가 잘못된곳은 없는지 확인해봐야한다. 눈에띄게 감소하는 폭이 늘어나지 않았기 때문에

        ####################################################################################
        # 박스 좌표 loss
        xy_loss = tf.reduce_sum(1 * I_obj * tf.square(
            labels[:, :, :, :, 21:23] - predict_xy)) / (nI_obj + 1e-6) / 2.
        wh_loss = tf.reduce_sum(1 * I_obj * tf.square(
            tf.sqrt(labels[:, :, :, :, 23:25]) - tf.sqrt(predict_wh))) / (nI_obj + 1e-6) / 2.
        coord_loss = xy_loss + wh_loss  # 논문의 첫 번째 수식
        print(coord_loss)
        ####################################################################################

        ####################################################################################
        # obect loss (오브젝트가 있을 때와 없을때의 loss)
        iou = IOU(predict_box, labels[:, :, :, :, 21:25])
        best_box = tf.reduce_max(iou, 4)
        best_box = tf.expand_dims(best_box, -1)

        conf_mask = tf.cast(best_box < 0.6, tf.float32) * (1 - I_obj) * 1
        conf_mask = conf_mask + I_obj * 5
        nb_conf_mask = tf.reduce_sum(tf.cast(conf_mask > 0.0, tf.float32))
        ob_loss = tf.reduce_sum(tf.square(I_obj - object_conf) * conf_mask) / (nb_conf_mask + 1e-6)
        print(ob_loss)
        ####################################################################################

        ####################################################################################
        # class loss
        pred_class = pred[:, :, :, :, 5:]   # [B, 13, 13, 5, 20]
        true_class = labels[:, :, :, :, :20]   # [B, 13, 13, 5, 20]
        true_class = tf.argmax(true_class, -1)  
        class_loss = tf.keras.losses.sparse_categorical_crossentropy(true_class,
                                                                      pred_class,
                                                                      from_logits=True)
        class_loss = tf.expand_dims(class_loss, -1) * I_obj
        class_loss = 1 * tf.reduce_sum(class_loss) / (nI_obj + 1e-6)
        #print(class_loss)
        ####################################################################################

        total_loss = (coord_loss + ob_losss + class_loss)

    grads = tape.gradient(total_loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss

def generate_images(model, images, SCORE_threhold, IOU_threshold):

    logits = run_model(model, images, False)
    logits = tf.reshape(logits,
                        [FLAGS.batch_size,
                         FLAGS.output_size,
                         FLAGS.output_size,
                         5,
                         25])
    offsets = tf.cast(grid_offset(FLAGS.output_size, FLAGS.output_size), tf.float32)
    for batch in range(FLAGS.batch_size):
        image = images[batch].numpy()
        logits_ = tf.expand_dims(logits[batch], 0)  # [1, 13, 13, 5, 25]
        re_scale = tf.keras.backend.cast_to_floatx(tf.keras.backend.int_shape(logits_)[1:3]) # [1,] --> {13., 13.}
        re_scale = tf.reshape(re_scale, [1,1,1,1,2])    # rescale by 13

        predict_xy = tf.nn.sigmoid(logits_[..., 0:2])
        predict_xy = (predict_xy + offsets) / re_scale

        predict_wh = (tf.exp(logits_[..., 2:4]) * ANCHORS) / re_scale

        confidence_score = tf.nn.sigmoid(logits_[..., 4:5])
        predict_class = tf.nn.softmax(logits_[..., 5:], 4)

        predict_xy = predict_xy[0, ...] # squezze batch index
        predict_wh = predict_wh[0, ...] # squezze batch index
        confidence_score = confidence_score[0, ...] # squezze batch index
        predict_class = predict_class[0, ...] # squezze batch index

        box_xy1 = predict_xy - 0.5 * predict_wh # xmin, ymin [13, 13, 5, 2]
        box_xy2 = predict_xy + 0.5 * predict_wh # xmax, ymax [13, 13, 5, 2]
        box = tf.concat([box_xy1, box_xy2], -1) # [13, 13, 5, 4]
        
        box_score = confidence_score * predict_class    # [13, 13, 5, 20]
        box_class_index = tf.argmax(box_score, -1)  # [13, 13, 5]
        box_class_score = tf.keras.backend.max(box_score, -1)   # [13, 13, 5]
        prediction_mask = box_class_score >= SCORE_threhold # [13, 13, 5]

        boxes = tf.boolean_mask(box, prediction_mask) # box_class_score >= SCORE_threhold 인 성분만 가지고옴
        scores = tf.boolean_mask(box_class_score, prediction_mask)
        classes = tf.boolean_mask(box_class_index, prediction_mask)

        # NMS
        selected_indices = tf.image.non_max_suppression(boxes, scores, 50, IOU_threshold)
        boxes = tf.gather(boxes, selected_indices)
        scores = tf.gather(scores, selected_indices)
        classes = tf.gather(classes, selected_indices)

        im = np.array(image)
        height, width, _ = im.shape

        # Create figure and axes
        fig, ax = plt.subplots(1)
        # Display the image
        ax.imshow(im)

        # Create a Rectangle potch
        count_detected = boxes.shape[0]
        for j in range(count_detected):
            box = boxes[j]
            assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
            x = box[0] * width
            y = box[1] * height
            w = (box[2] - box[0]) * width
            h = (box[3] - box[1]) * height
            class_ = classes[j].numpy()

            rect = patches.Rectangle(
                (x.numpy(), y.numpy()),
                w.numpy(),
                h.numpy(),
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            # Add the patch to the Axes
            ax.add_patch(rect)

        plt.show()

    # dim??!?!? 필요한가?

    #return img

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

                if count % 10 == 0:
                    print("Epoch: {} [{}/{}] loss = {}".format(epoch, step + 1, batch_idx, loss))
                
                count += 1

            if epoch % 20 == 0:
                generate_images(model, image, 0.01, 0.01)

if __name__ == "__main__":
    main()