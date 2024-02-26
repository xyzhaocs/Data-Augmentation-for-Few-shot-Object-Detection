import cv2
import numpy as np
import random
import os
import math


# 首先随机从模板图片里选取球的图片
def select_ball_from_template(template_image_path, template_label_path,BALL_NUM):
    # 读取模板图片
    template_image = cv2.imread(template_image_path)
    # 读取模板图片的球位置信息
    with open(template_label_path, "r") as f:
        template_bboxes = f.readlines()
    # 创建列表来存储球的标号和对应图片
    ball_index= []
    ball_img=[]
    for i in range(BALL_NUM): 
        # 随机选择一个球的位置信息
        random_bbox = random.choice(template_bboxes).split()
        class_id = random_bbox[0]
        class_id=int(class_id)
        x_center = float(random_bbox[1]) * template_image.shape[1]
        y_center = float(random_bbox[2]) * template_image.shape[0]
        width = float(random_bbox[3]) * template_image.shape[1]
        height = float(random_bbox[4]) * template_image.shape[0]

        # 获取球在原有图片中的位置和尺寸
        # (x_min,y_min)：球的左上角坐标
        # (x_max,y_max)：球的右下角坐标
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)
        # 对球进行缩小
        ball_width = int((x_max - x_min))
        ball_height = int((y_max - y_min))


        # 将截取的球part调整为目标位置和尺寸
        ball = template_image[y_min:y_max, x_min:x_max]
        ball_resized = cv2.resize(ball, (ball_width, ball_height))
        
        ball_index.append(class_id)
        ball_img.append(ball_resized)
    return ball_index,ball_img

# 然后粘贴到一个coco图片中  保存处理后的图片和label标签信息
def paste_to_coco(coco_image_folder,coco_index,ball_index,ball_img,BALL_NUM,i):

    # 在COCO数据集中随机选择一个图片路径
    coco_image_files = os.listdir(coco_image_folder)
    random_image_file = coco_image_files[coco_index]
    # 读入图片的路径和输出图片的路径和输出图片标签的路径
    coco_image_path = coco_image_folder +'/'+ random_image_file
    coco=random_image_file.lstrip('0')
    print(str(i)+"_"+coco)
    # output_fold = "out"+str(i)+"_"+str(coco_index)+"_"+random_image_file
    output_fold =str(i)+ "_coco"+coco
    output_image_path = coco_outimg_folder+'/' + output_fold
    label_txt_path = coco_label_folder+'/' + output_fold.replace(".jpg", ".txt")
    # 读取COCO图片
    coco_image = cv2.imread(coco_image_path)
   
    coco_image = coco_image.astype(np.float32)
   
   # coco_image = cv2.resize(coco_image,(640,800))
    coco_height, coco_width = coco_image.shape[:2]
    
    ball_centers = []
    ball_radius = []
    for i in range(BALL_NUM):
        # 获取球的图片的宽度和高度
        class_id = ball_index[i]
        ball_resized = ball_img[i]
        ball_height, ball_width = ball_resized.shape[:2]
        while True:
            # 随机选择球在coco图片中的目标位置(并控制其不超出边界范围)
            x_target = random.randint(0, coco_width - ball_width)
            y_target = random.randint(0, coco_height - ball_height)
            overlap = False
            for center,radius in zip(ball_centers,ball_radius):
                if np.linalg.norm(np.array(center) - np.array([x_target + ball_width/2, y_target + ball_height/2])) < radius+math.sqrt((ball_height/2)**2 + (ball_width/2)**2):
                    overlap = True
                    break
            # 如果不重叠，将球粘贴到 COCO 图像上
            # print(f"overlap={overlap}",end='')
            if not overlap:
                ball_centers.append([x_target + ball_width/2, y_target + ball_height/2])
                ball_radius.append(math.sqrt((ball_height/2)**2 + (ball_width/2)**2))
                # 将球放置到COCO图片中的目标位置
                coco_image[y_target:y_target+ball_height, x_target:x_target+ball_width] = ball_resized
                # 生成Yolov5标签txt并保存到output_label中
                x_center = float(x_target +  ball_width/ 2) / coco_width
                y_center = float(y_target + ball_height / 2) / coco_height
                width = float(ball_width) / coco_width
                height = float(ball_height) / coco_height

                if i == 0:
                    with open(label_txt_path, "w") as f:
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"+"\n")
                else:
                    with open(label_txt_path, "a+") as f:
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"+"\n")
                break
    # 将处理后的图片保存到output_img中
    cv2.imwrite(output_image_path, coco_image)

def info2dict(pimg,ptxt):
    info={}
    dir=os.listdir(pimg)
    for img in dir:
        if img[-4:]=='.jpg':
            img_path=pimg+img
            txt_path=ptxt+img
            txt_path=txt_path.replace('.jpg','.txt')
            # print(img_path)
            template_image = cv2.imread(img_path)
            # 读取模板图片的球位置信息
            with open(txt_path, "r") as f:
                template_bboxes = f.readlines()
            # print(template_bboxes)
            for random_bbox in template_bboxes:
                # print(random_bbox)
                # print(template_image.shape)
                random_bbox=random_bbox.split()
                class_id = random_bbox[0]
                class_id=int(class_id)
                x_center = float(random_bbox[1]) * template_image.shape[1]
                y_center = float(random_bbox[2]) * template_image.shape[0]
                width = float(random_bbox[3]) * template_image.shape[1]
                height = float(random_bbox[4]) * template_image.shape[0]

                # 获取球在原有图片中的位置和尺寸
                # (x_min,y_min)：球的左上角坐标
                # (x_max,y_max)：球的右下角坐标
                x_min = int(x_center - width / 2)
                y_min = int(y_center - height / 2)
                x_max = int(x_center + width / 2)
                y_max = int(y_center + height / 2)
                # 对球进行缩小
                ball_width = int((x_max - x_min))
                ball_height = int((y_max - y_min))


                # 将截取的球part调整为目标位置和尺寸
                ball = template_image[y_min:y_max, x_min:x_max]
                ball_resized = cv2.resize(ball, (ball_width, ball_height))
                if class_id not in info.keys():
                    info[class_id]=[ball_resized]
                
                info[class_id].append(ball_resized)
                
    return info

if __name__ == '__main__':
    # 模板图片路径
    CLASS_NUM=17
    BALL_NUM=5#生成的图片中的目标
    K=100#扩展倍数
    template_image_path='./simg/'
    template_label_path='./slabel/'
    # COCO数据集路径
    coco_image_folder = "./coco128/images/train2017"  # coco数据集的路径
    # 处理后的结果的存放路径
    coco_label_folder = "./images"
    coco_outimg_folder = "./label"
    # 从coco数据集source_img中选取图片进行粘贴
    tuku=info2dict(template_image_path,template_label_path)
    print(tuku.keys())
    
    crops=[]
    for i in range(K*BALL_NUM):
        class_list = [i for i in range(CLASS_NUM)]
        random.shuffle(class_list)
        crops+=class_list

    pics=[]
    for i in range(K*CLASS_NUM):
        pics.append(crops[i*BALL_NUM:(i+1)*BALL_NUM])
        
    for i in range(K*CLASS_NUM):
        ball_img=[]
        ball_index=pics[i]
        for j in ball_index:
            ball_img+=random.sample(tuku[j],1)
        paste_to_coco(coco_image_folder,random.randint(0, 127), ball_index, ball_img, BALL_NUM,i)
        