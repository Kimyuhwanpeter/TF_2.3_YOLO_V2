# -*- coding:utf-8 -*-
from random import shuffle
from absl import flags
from model import *
from keras_radam.training import RAdamOptimizer

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys

flags.DEFINE_string("tr_txt_path", "D:/[1]DB/[3]detection_DB/archive/100example_txt", "Training text path")

flags.DEFINE_string("tr_img_path", "D:/[1]DB/[3]detection_DB/archive/images", "Training image path")

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

optim = tf.keras.optimizers.Adam(0.5e-4)

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

            xmin = (float(line.split(' ')[1]))
            xmax = (float(line.split(' ')[2]))
            ymin = (float(line.split(' ')[3]))
            ymax = (float(line.split(' ')[4]))
            #height = int(line.split(' ')[4])
            #width = int(line.split(' ')[5])
            classes = int(line.split(' ')[0])

            xmin = xmin * FLAGS.img_size
            xmin = max(min(xmin, FLAGS.img_size), 0)
            xmax = xmax * FLAGS.img_size
            xmax = max(min(xmax, FLAGS.img_size), 0)

            ymin = ymin * FLAGS.img_size
            ymin = max(min(ymin, FLAGS.img_size), 0)
            ymax = ymax * FLAGS.img_size
            ymax = max(min(ymax, FLAGS.img_size), 0)

            if xmax > xmin and ymax > ymin:
                x = (xmin + xmax) * 0.5
                x = x / FLAGS.img_size / FLAGS.output_size
                y = (ymin + ymax) * 0.5
                y = y / FLAGS.img_size / FLAGS.output_size
                grid_x = int(np.floor(x))
                grid_y = int(np.floor(y))
                if grid_x < FLAGS.output_size and grid_y < FLAGS.output_size:
                    w = (xmax - xmin) / FLAGS.img_size / FLAGS.output_size
                    h = (ymax - ymin) / FLAGS.img_size / FLAGS.output_size

                    boxData = [x, y, w, h]
                    best_box = 0
                    best_anchor = 0
                    for i in range(anchor_count):
                        intersect = np.minimum(w, ANCHORS_box[i, 0]) * np.minimum(h, ANCHORS_box[i, 1])
                        union = ANCHORS_box[i, 0] * ANCHORS_box[i, 1] + (w * h) - intersect
                        iou = intersect / union
                        if iou > best_box:
                            best_box = iou
                            best_anchor = i

                    responsibleGrid[b][grid_x][grid_y][best_anchor][classes] = 1.    # class
                    responsibleGrid[b][grid_x][grid_y][best_anchor][21:25] = boxData    # box
                    responsibleGrid[b][grid_x][grid_y][best_anchor][FLAGS.num_classes] = 1. # confidence

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
            labels[:, :, :, :, 21:23] - predict_xy)) / (nI_obj + 1e-6)
        wh_loss = tf.reduce_sum(1 * I_obj * tf.square(
            tf.sqrt(labels[:, :, :, :, 23:25]) - tf.sqrt(predict_wh))) / (nI_obj + 1e-6)
        coord_loss = xy_loss + wh_loss  # 논문의 첫 번째 수식
        #print(coord_loss)
        #print("loss_xywh = {:4.3f}".format(coord_loss))
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
        #print(ob_loss)
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

        total_loss = (coord_loss + ob_loss + class_loss)

    grads = tape.gradient(total_loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, confidence=None,classes=None):
        self.xmin, self.ymin = xmin, ymin
        self.xmax, self.ymax = xmax, ymax
        ## the code below are used during inference
        # probability
        self.confidence      = confidence
        # class probaiblities [c1, c2, .. cNclass]
        self.set_class(classes)
        
    def set_class(self,classes):
        self.classes = classes
        self.label   = np.argmax(self.classes) 
        
    def get_label(self):  
        return(self.label)
    
    def get_score(self):
        return(self.classes[self.label])

def generate_images(model, images, obj_threhold, iou_threshold):

    def adjust_minmax(c,_max):
        if c < 0:
            c = 0   
        if c > _max:
            c = _max
        return c

    # https://www.maskaravivek.com/post/yolov2/
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
        logits_ = logits[batch].numpy()   # [13, 13, 5, 25]
        re_scale = tf.keras.backend.cast_to_floatx(tf.keras.backend.int_shape(logits_)[1:3]) # [1,] --> {13., 13.}
        re_scale = tf.reshape(re_scale, [1,1,1,2])    # rescale by 13

        logits_[..., 0:2] = (tf.nn.sigmoid(logits_[..., 0:2]) + offsets[batch]) / re_scale # x,y
        logits_[..., 2:4] = (tf.exp(logits_[..., 2:4]) * ANCHORS) / re_scale    # w,h
        logits_[..., 0:2] = logits_[..., 0:2] - logits_[..., 2:4] * 0.5 # xmin, ymin
        logits_[..., 2:4] = logits_[..., 0:2] + logits_[..., 2:4] * 0.5 # xmax, ymax

        logits_[..., 4:5] = tf.nn.sigmoid(logits_[..., 4:5])    # confidence
        logits_[..., 5:] = tf.nn.softmax(logits_[..., 5:], 3)   # class
        
        logits_[..., 5:] = logits_[..., 4:5] * logits_[..., 5:]    # [13, 13, 5, 25]   # box_scores
        box_classes = tf.argmax(logits_[..., 5:], 3)
        box_classes_scores = tf.keras.backend.max(logits_[..., 5:], axis=-1)

        prediction_mask = box_classes_scores >= obj_threhold
        boxes = tf.boolean_mask(logits_[..., 0:4], prediction_mask)
        scores = tf.boolean_mask(box_classes_scores, prediction_mask)
        classes = tf.boolean_mask(box_classes, prediction_mask)

        #nms
        selected_idx = tf.image.non_max_suppression(boxes, 
                                                    scores, 
                                                    1, 
                                                    iou_threshold=iou_threshold)
        boxes = tf.keras.backend.gather(boxes, selected_idx)
        scores = tf.keras.backend.gather(scores, selected_idx)
        classes = tf.keras.backend.gather(classes, selected_idx)

        #boxes_ = []
        #for row in range(FLAGS.output_size):
        #    for col in range(FLAGS.output_size):
        #        for b in range(5):
        #            classes = logits_[row, col, b, 5:]

        #            if np.sum(classes) > 0:
        #                x, y, w, h = logits_[row, col, b, :4]
        #                confid = logits_[row, col, b, 4]
        #                box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, confid, classes)
        #                if box.get_score() > obj_threhold:
        #                    boxes_.append(box) # 적당한 값을 저장을 해놔야 하는데...

        #final_boxes = nms_process(boxes_, iou_threshold, iou_threshold)

        im = np.array(image)
        height, width, _ = im.shape

        # Create figure and axes
        fig, ax = plt.subplots(1)
        # Display the image
        ax.imshow(im)

        # Create a Rectangle potch
        for box in boxes:
            assert len(box) == 4, "Got more values than in x1, y1, x2, y2, in a box!"
            
            # 여기가 뭔가 이상하다... 학습은 되는데... 박스가 이상하게 생긴다..
            rect = patches.Rectangle(
                (box[1] * height, box[0] * width),
                (box[3] - box[1]) * width,
                (box[2] - box[0]) * height,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            # Add the patch to the Axes
            ax.add_patch(rect)


        plt.show()

def nms_process(boxes, iou_threshold, obj_threshold):

    CLASS = len(boxes[0].classes)
    index_box = []
    # NMS
    for c in range(CLASS):
        class_probability_from_bbx = [box.classes[c] for box in boxes]

        sorted_indices = list(reversed(np.argsort(class_probability_from_bbx)))
        for i in range(len(sorted_indices)):
            index = sorted_indices[i]

            if boxes[index].classes[c] == 0:
                continue # ignore the zero probability
            else:
                index_box.append(index)
                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    box_iou = test_IOU(boxes[index], boxes[index_j])
                    if box_iou >= iou_threshold:
                        classes = boxes[index_j].classes
                        classes[c] = 0  # set prob 0
                        boxes[index_j].set_class(classes)

    newbox = [boxes[i] for i in index_box if boxes[i].get_score() > obj_threshold]

    return newbox

def test_IOU(box1, box2):

    box1_x1 = box1.xmin # xmin1
    box1_y1 = box1.ymin # ymin1
    box1_x2 = box1.xmax # xmax1
    box1_y2 = box1.ymax # ymax1

    box2_x1 = box2.xmin # xmin2
    box2_y1 = box2.ymin # ymin2
    box2_x2 = box2.xmax # xmax2
    box2_y2 = box2.ymax # ymax2

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

            if epoch % 100 == 0 and epoch != 0:
                generate_images(model, image, 0.015, 0.01)

if __name__ == "__main__":
    main()
