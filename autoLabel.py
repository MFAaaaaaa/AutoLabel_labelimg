# 引用opencv-python库
import glob
from xml.etree import ElementTree  # 导入ElementTree模块
import cv2
import numpy as np
import os
from os import getcwd
from xml.etree import ElementTree as ET

# weightsPath = "./newDate/date.weights"
# configPath = "./newDate/date.cfg"
# labelsPath = "./newDate/date.names"

# weightsPath = "./rotor/rotor_16000.weights"
# configPath = "./rotor/rotor.cfg"
# labelsPath = "./rotor/rotor.names"

weightsPath = "./bingxian/bingxian.weights"
configPath = "./bingxian/bingxian.cfg"
labelsPath = "./bingxian/bingxian.names"

# weightsPath = "./cfg/yolov3.weights"
# configPath = "./cfg/yolov3.cfg"
# labelsPath = "./cfg/coco.names"

# 读取names文件中的类别名
LABELS = open(labelsPath).read().strip().split("\n")

# 使用opencv加载Darknet模型
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


# 下面是通过检测获取坐标的函数
def coordinate_get(img):
    # print("coordinate_get执行")
    coordinates_list = []  # 创建坐标列表
    boxes = []
    confidences = []
    classIDs = []
    (H, W) = img.shape[:2]
    # 得到 YOLO需要的输出层
    ln = net.getLayerNames()
    # print(net.getLayerNames())
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # print(ln)  #['yolo_89', 'yolo_101', 'yolo_113']
    # 从输入图像构造一个blob，然后通过加载的模型，给我们提供边界框和相关概率
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    # print(blob,type(blob))

    net.setInput(blob)
    layerOutputs = net.forward(ln)
    # print(layerOutputs)

    # 在每层输出上循环
    for output in layerOutputs:
        # 对每个检测进行循环
        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # print(classID,confidence) #0 0.0
            # 过滤掉那些置信度较小的检测结果
            if confidence > 0.01:
                # 框后接框的宽度和高度
                box = detection[0:4] * np.array([W, H, W, H])
                # print(box)  #框的四个坐标
                (centerX, centerY, width, height) = box.astype("int")
                # 边框的左上角
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # 更新检测出来的框
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                # print(classID, confidence)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.3)
    # print(idxs,len(idxs))  #len(idxs)即为目标数
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            xmin = int(x)
            ymin = int(y)
            xmax = int(x + w)
            ymax = int(y + h)
            coordinates_list.append([xmin, ymin, xmax, ymax, classIDs[i]])
    # print(len(coordinates_list))
    return coordinates_list


# 定义一个创建一级分支object的函数
def create_object(root, xi, yi, xa, ya, obj_name):  # 参数依次，树根，xmin，ymin，xmax，ymax
    # print("create_object执行")
    # 创建一级分支object
    _object = ET.SubElement(root, 'object')
    # 创建二级分支
    name = ET.SubElement(_object, 'name')
    # print(obj_name)
    name.text = str(obj_name)
    pose = ET.SubElement(_object, 'pose')
    pose.text = 'Unspecified'
    truncated = ET.SubElement(_object, 'truncated')
    truncated.text = '0'
    difficult = ET.SubElement(_object, 'difficult')
    difficult.text = '0'
    # 创建bndbox
    bndbox = ET.SubElement(_object, 'bndbox')
    xmin = ET.SubElement(bndbox, 'xmin')
    xmin.text = '%s' % xi
    ymin = ET.SubElement(bndbox, 'ymin')
    ymin.text = '%s' % yi
    xmax = ET.SubElement(bndbox, 'xmax')
    xmax.text = '%s' % xa
    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = '%s' % ya


# 创建xml文件的函数
def create_tree(image_name, h, w):
    # print("create_tree执行")
    global annotation

    # 创建树根annotation
    annotation = ET.Element('annotation')
    # 创建一级分支folder
    folder = ET.SubElement(annotation, 'folder')
    # 添加folder标签内容
    # folder.text = ("1111")

    # 创建一级分支filename
    filename = ET.SubElement(annotation, 'filename')
    filename.text = image_name

    # 创建一级分支path
    path = ET.SubElement(annotation, 'path')

    # path.text = getcwd() + '\{}'.format(image_name)  # 用于返回当前工作目录
    path.text = image_name
    # 创建一级分支source
    source = ET.SubElement(annotation, 'source')
    # 创建source下的二级分支database
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'

    # 创建一级分支size
    size = ET.SubElement(annotation, 'size')
    # 创建size下的二级分支图像的宽、高及depth
    width = ET.SubElement(size, 'width')
    width.text = str(w)
    height = ET.SubElement(size, 'height')
    height.text = str(h)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'

    # 创建一级分支segmented
    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'


# 设置xml文件换行和制表符
def pretty_xml(element, indent, newline, level=0):  # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
    if element:  # 判断element是否有子元素
        if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
            # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
            # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将element转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作


def deal_img(img_dir):
    i = 0
    for root, dirs, files in os.walk(img_dir, topdown=False):
        print("该目录下共", len(files), "张图片")
        # print('*********')
    # for ilist in glob.glob(imagespath):
    #     IMAGES_LIST.append(files[i]
    # print(IMAGES_LIST[i])
    # image_name = files[i]
    for image_name in files:
        # print(image_name)
        image = cv2.imread(os.path.join(img_dir, image_name))
        # print(image)
        coordinates_list = coordinate_get(image)
        # print(coordinates_list) #[[428, 395, 847, 668, 0]]
        print("检测到", len(coordinates_list), "个目标")
        (h, w) = image.shape[:2]
        # print(image.shape[:2]) #图片的宽高
        create_tree(image_name, h, w)
        # print(coordinates_list)
        for coordinate in coordinates_list:
            # print(coordinate[4])  #标签名称
            # print(coordinate[0],coordinate[1],coordinate[2],coordinate[3]) 4个坐标
            # print(annotation)
            label_id = coordinate[4]
            create_object(annotation, coordinate[0], coordinate[1], coordinate[2], coordinate[3], LABELS[label_id])
            tree = ET.ElementTree(annotation)
            # print(files[i].strip('.jpg'))
            # print(files[i].replace('.jpg',''),type(files[i]))
            root = tree.getroot()  # 得到根元素，Element类
            pretty_xml(root, '\t', '\n')  # 执行美化方法
            a = files[i].replace('.jpg', '')
            tree.write('E:/Data/2021-10-15/Annotations/{0}.xml'.format(a))

        i += 1
        # print(i, '***************')
        print("第", i, "个xml文件生成")


if __name__ == "__main__":
    # imagesPath = 'D:/autolabel/images'
    imgdir = "E:/Data/2021-10-15/JPEGS"  # 待标注的图片
    deal_img(imgdir)
