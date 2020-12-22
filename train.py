# -*- coding:utf-8 -*-
from random import shuffle
from absl import flags
from model import *
from keras_radam.training import RAdamOptimizer

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
import cv2

flags.DEFINE_string("tr_txt_path", "D:/[1]DB/[3]detection_DB/PascalVoc2012/pascal_voc_2012/VOC2012/Annotations_text", "Training text path")

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

optim = tf.keras.optimizers.Adam(0.5e-5)

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

            xmin = (float(line.split(',')[0]))
            xmax = (float(line.split(',')[2]))
            ymin = (float(line.split(',')[1]))
            ymax = (float(line.split(',')[3]))
            height = int(line.split(',')[4])
            width = int(line.split(',')[5])
            classes = int(line.split(',')[6])

            xmin = xmin
            xmax = xmax

            ymin = ymin
            ymax = ymax

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

    return(coords)

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

        pred = run_model(model, images, True)   # 입력이 두개어야 한다.
        offsets = tf.cast(grid_offset(FLAGS.output_size, FLAGS.output_size), tf.float32)

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
            labels[:, :, :, :, 21:23] - predict_xy)) / (nI_obj + 1e-6) / 2
        wh_loss = tf.reduce_sum(1 * I_obj * tf.square(
            tf.sqrt(labels[:, :, :, :, 23:25]) - tf.sqrt(predict_wh))) / (nI_obj + 1e-6) / 2
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
        ob_loss = tf.reduce_sum(tf.square(I_obj - object_conf) * conf_mask) / (nb_conf_mask + 1e-6) / 2.
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

    def _sigmoid(x):
        return 1. / (1. + np.exp(-x))
    def _softmax(x, axis=-1, t=-100.):
        x = x - np.max(x)

        if np.min(x) < t:
            x = x/np.min(x)*t

        e_x = np.exp(x)
        return e_x / e_x.sum(axis, keepdims=True)

    def adjust_minmax(c,_max):
        if c < 0:
            c = 0   
        if c > _max:
            c = _max
        return c

    # https://www.maskaravivek.com/post/yolov2/
    logits = run_model(model, images, False)

    # offset은 그대로 잘 되고 있음 --> offset 문제는 아니다
    for batch in range(FLAGS.batch_size):
        image = images[batch].numpy()
        logits_ = logits[batch].numpy()   # [13, 13, 5, 25]
        re_scale = tf.keras.backend.cast_to_floatx(tf.keras.backend.int_shape(logits_)[1:3]) # [1,] --> {13., 13.}
        re_scale = tf.reshape(re_scale, [1,1,1,2])    # rescale by 13

        (mat_GRID_W,
         mat_GRID_H,
         mat_ANCHOR_W,
         mat_ANCHOR_H) = get_shifting_matrix(logits_)

        logits_[..., 0] = (_sigmoid(logits_[..., 0]) + mat_GRID_W) / FLAGS.output_size
        logits_[..., 1] = (_sigmoid(logits_[..., 1]) + mat_GRID_H) / FLAGS.output_size
        logits_[..., 2] = (np.exp(logits_[..., 2]) + mat_ANCHOR_W) / FLAGS.output_size
        logits_[..., 3] = (np.exp(logits_[..., 3]) + mat_ANCHOR_H) / FLAGS.output_size

        logits_[..., 0] = logits_[..., 0] - 0.5 * logits_[..., 2]
        logits_[..., 1] = logits_[..., 1] - 0.5 * logits_[..., 3]
        logits_[..., 2] = logits_[..., 0] + 0.5 * logits_[..., 2]
        logits_[..., 3] = logits_[..., 1] + 0.5 * logits_[..., 3]

        logits_[..., 4:5] = _sigmoid(logits_[..., 4:5])    # confidence
        logits_[..., 5:] = _softmax(logits_[..., 5:], -1)   # class
        
        logits_[..., 5:] = logits_[..., 4:5] * logits_[..., 5:]    # [13, 13, 5, 25]   # box_scores

        # 이 부분에다가 추가

        #nms
        boxes = find_high_class_prob_bbox(logits_, obj_threhold)
        boxes_ = nms_process(boxes, obj_threhold, iou_threshold)

        im = np.array(image)
        height, width, _ = im.shape

        # Create figure and axes
        fig, ax = plt.subplots(1)
        # Display the image
        ax.imshow(im)

        # Create a Rectangle potch
        for box in boxes_:
            #assert len(box) == 4, "Got more values than in x1, y1, x2, y2, in a box!"
            xmin = adjust_minmax(int(box.xmin * FLAGS.img_size), FLAGS.img_size)
            ymin = adjust_minmax(int(box.ymin * FLAGS.img_size), FLAGS.img_size)
            xmax = adjust_minmax(int(box.xmax * FLAGS.img_size), FLAGS.img_size)
            ymax = adjust_minmax(int(box.ymax * FLAGS.img_size), FLAGS.img_size)

            # 여기가 뭔가 이상하다... 학습은 되는데... 박스가 이상하게 생긴다..
            cv2.rectangle(image, 
                          pt1=(xmin,ymin), 
                          pt2=(xmax,ymax), 
                          color=(255,0,0), 
                          thickness=3)
            #rect = patches.Rectangle(
            #    (xmin, ymin),
            #    (xmax - xmin),
            #    (ymax - ymin),
            #    linewidth=1,
            #    edgecolor="r",
            #    facecolor="none",
            #)
            # Add the patch to the Axes
            #ax.add_patch(rect)


        cv2.imshow("name", image)
        cv2.waitKey(0)

def nms_process(boxes, obj_threshold, iou_threshold):      # 이 부분 고쳐야한다.

    '''
    boxes : list containing "good" BoundBox of a frame
            [BoundBox(),BoundBox(),...]
    '''
    bestAnchorBoxFinder = BestAnchorBoxFinder([])
    
    CLASS    = len(boxes[0].classes)
    index_boxes = []   
    # suppress non-maximal boxes
    for c in range(CLASS):
        # extract class probabilities of the c^th class from multiple bbox
        class_probability_from_bbxs = [box.classes[c] for box in boxes]

        #sorted_indices[i] contains the i^th largest class probabilities
        sorted_indices = list(reversed(np.argsort( class_probability_from_bbxs)))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            
            # if class probability is zero then ignore
            if boxes[index_i].classes[c] == 0:  
                continue
            else:
                index_boxes.append(index_i)
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    
                    # check if the selected i^th bounding box has high IOU with any of the remaining bbox
                    # if so, the remaining bbox' class probabilities are set to 0.
                    bbox_iou = bestAnchorBoxFinder.bbox_iou(boxes[index_i], boxes[index_j])
                    if bbox_iou >= iou_threshold:
                        classes = boxes[index_j].classes
                        classes[c] = 0
                        boxes[index_j].set_class(classes)
                        
    newboxes = [ boxes[i] for i in index_boxes if boxes[i].get_score() > obj_threshold ]                
    
    return newboxes

def get_shifting_matrix(netout):

    Anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
    Anchors = np.array(Anchors)

    GRID_H, GRID_W, BOX = netout.shape[:3]
    no = netout[...,0]
        
    ANCHORSw = Anchors[::2]
    ANCHORSh = Anchors[1::2]
       
    mat_GRID_W = np.zeros_like(no)
    for igrid_w in range(GRID_W):
        mat_GRID_W[:,igrid_w,:] = igrid_w

    mat_GRID_H = np.zeros_like(no)
    for igrid_h in range(GRID_H):
        mat_GRID_H[igrid_h,:,:] = igrid_h

    mat_ANCHOR_W = np.zeros_like(no)
    for ianchor in range(BOX):    
        mat_ANCHOR_W[:,:,ianchor] = ANCHORSw[ianchor]

    mat_ANCHOR_H = np.zeros_like(no) 
    for ianchor in range(BOX):    
        mat_ANCHOR_H[:,:,ianchor] = ANCHORSh[ianchor]

    return(mat_GRID_W,mat_GRID_H,mat_ANCHOR_W,mat_ANCHOR_H)

def find_high_class_prob_bbox(output, obj_threshold):

    GRID_H, GRID_W, BOX = output.shape[:3]
    boxes = []
    for row in range(GRID_H):
        for col in range(GRID_W):
            for b in range(BOX):
                classes = output[row, col, b, 5:]

                if np.sum(classes) > 0:
                    x, y, w, h = output[row, col, b, :4]
                    confidence = output[row, col, b, 4]
                    box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, confidence, classes)
                    if box.get_score() > obj_threshold:
                        boxes.append(box)

    return(boxes)

class BestAnchorBoxFinder(object):
    def __init__(self, ANCHORS):
        '''
        ANCHORS: a np.array of even number length e.g.
        
        _ANCHORS = [4,2, ##  width=4, height=2,  flat large anchor box
                    2,4, ##  width=2, height=4,  tall large anchor box
                    1,1] ##  width=1, height=1,  small anchor box
        '''
        self.anchors = [BoundBox(0, 0, ANCHORS[2*i], ANCHORS[2*i+1]) 
                        for i in range(int(len(ANCHORS)//2))]
        
    def _interval_overlap(self,interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2,x4) - x1
        else:
            if x2 < x3:
                 return 0
            else:
                return min(x2,x4) - x3  

    def bbox_iou(self,box1, box2):
        intersect_w = self._interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = self._interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  

        intersect = intersect_w * intersect_h

        w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
        w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin

        union = w1*h1 + w2*h2 - intersect

        return float(intersect) / union
    
    def find(self,center_w, center_h):
        # find the anchor that best predicts this box
        best_anchor = -1
        max_iou     = -1
        # each Anchor box is specialized to have a certain shape.
        # e.g., flat large rectangle, or small square
        shifted_box = BoundBox(0, 0,center_w, center_h)
        ##  For given object, find the best anchor box!
        for i in range(len(self.anchors)): ## run through each anchor box
            anchor = self.anchors[i]
            iou    = self.bbox_iou(shifted_box, anchor)
            if max_iou < iou:
                best_anchor = i
                max_iou     = iou
        return(best_anchor,max_iou)  

def main():
    model = Other_model()
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
                
                if count % 500 == 0 and count != 0:
                    generate_images(model, image, 0.15, 0.5)

                count += 1

            #if epoch % 1 == 0 and epoch != 0:
            #generate_images(model, image, 0.2, 0.5)

if __name__ == "__main__":
    main()
