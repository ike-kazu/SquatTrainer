from utils import Singleton
import json
import trt_pose.coco
import trt_pose.models
#from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import torch
"""pytorch -> tensorRTのコンバーター"""
import torch2trt
from torch2trt import TRTModule
import cv2
import torchvision.transforms as transforms
import PIL.Image
import os
import math
import numpy as np

class ML(Singleton):
    count = 0
    def __init__(self):
        if ML.count == 0:
            # jsonの読み込み
            #with open('human_pose.json', 'r') as f:
            with open('human_pose copy.json', 'r') as f:
                self.human_pose = json.load(f)
            """
            Gets list of parts name from a COCO category
            骨格のリストを回して位相(topology)を作成
            """
            self.topology = trt_pose.coco.coco_category_to_topology(self.human_pose)
            """骨格のポイントの長さ"""
            self.num_parts = len(self.human_pose['keypoints'])
            """骨格の線の本数"""
            self.num_links = len(self.human_pose['skeleton'])

            """モデル、重みをロード"""
            MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
            OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

            print("loading model...")
            if os.path.exists(OPTIMIZED_MODEL):
                print('exist')
                self.model_trt = TRTModule()
                self.model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
            else:
                self.model = trt_pose.models.resnet18_baseline_att(self.num_parts, 2 * self.num_links).cuda().eval()
                self.model.load_state_dict(torch.load(MODEL_WEIGHTS))
                self.data = torch.zeros((1, 3, 224, 224)).cuda()
                print("converting model...")
                self.model_trt = torch2trt.torch2trt(self.model, [self.data], fp16_mode=True, max_workspace_size=1<<25)
                print("saving converted model...")
                torch.save(self.model_trt.state_dict(), OPTIMIZED_MODEL)
            
            self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
            self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
            self.device = torch.device('cuda')

            self.parse_objects = ParseObjects(self.topology)
            self.draw_objects = DrawObjects(self.topology)
            print("model is ready.")
            ML.count += 1
    
    def preprocess(self, image):
        self.device = torch.device('cuda')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(self.device)
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return image[None, ...]

    def execute(self, image):
            data = self.preprocess(image)
            cmap, paf = self.model_trt(data)
            cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
            counts, objects, peaks = self.parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
            image = cv2.flip(image, 1)
            image_1 = self.draw_objects(image, counts, objects, peaks)
            return image_1[:, ::-1, :]


class DrawObjects(object):
    
    def __init__(self, topology):
        self.topology = topology
        self.isDone = False
        #self.canCount = True
        self.leg_right = np.array([0, 0])
        self.foot_right = np.array([1, 1])
        self.foot_right_knee = np.array([2, 2])
        self.leg_right_knee = np.array([3, 3])
        self.leg_left = np.array([4, 4])
        self.foot_left = np.array([5, 5])
        self.foot_left_knee = np.array([6, 6])
        self.leg_left_knee = np.array([7, 7])
        self.knee_degree = 0
        self.count = 0
        
    def __call__(self, image, object_counts, objects, normalized_peaks):
        topology = self.topology
        height = image.shape[0]
        width = image.shape[1]        
        K = topology.shape[0]
        count = int(object_counts[0])
        K = topology.shape[0]
        for i in range(count):
            color = (225, 0, 0)
            obj = objects[0][i]
            C = obj.shape[0]
            for j in range(C):
                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    x = 224 - round(float(peak[1]) * width)
                    y = round(float(peak[0]) * height)
                    cv2.circle(image, (x, y), 3, color, 2)

            for k in range(K):
                c_a = topology[k][2]
                c_b = topology[k][3]
                if obj[c_a] >= 0 and obj[c_b] >= 0:
                    peak0 = normalized_peaks[0][c_a][obj[c_a]]
                    peak1 = normalized_peaks[0][c_b][obj[c_b]]
                    x0 = 224 - round(float(peak0[1]) * width)
                    y0 = round(float(peak0[0]) * height)
                    x1 = 224 - round(float(peak1[1]) * width)
                    y1 = round(float(peak1[0]) * height)
                    cv2.line(image, (x0, y0), (x1, y1), color, 2)
                        
                    if(k == 0):
                        self.foot_left = [x0, y0]
                        self.foot_left_knee = [x1, y1]
                    if(k == 1):
                        self.leg_left_knee = [x0, y0]
                        self.leg_left = [x1, y1]
                    if(k == 2):
                        self.foot_right = [x0, y0]
                        self.foot_right_knee = [x1, y1]
                    if(k == 3):
                        self.leg_right_knee = [x0, y0]
                        self.leg_right = [x1, y1]

                    if (self.foot_right_knee[0] == self.leg_right_knee[0] and self.foot_right_knee[1] == self.leg_right_knee[1]):
                        self.knee_degree = (math.atan2(self.leg_right[1] - self.foot_right_knee[1], self.leg_right[0] - self.foot_right_knee[0]) - math.atan2(self.foot_right[1] - self.foot_right_knee[1], self.foot_right[0] - self.foot_right_knee[0])) / math.pi * 180
                        #print('右膝の角度は' + str(self.knee_degree))
                    if (self.foot_left_knee[0] == self.foot_left_knee[0] and self.foot_left_knee[1] == self.foot_left_knee[1]):
                        self.knee_degree = (math.atan2(self.leg_left[1] - self.foot_left_knee[1], self.leg_left[0] - self.foot_left_knee[0]) - math.atan2(self.foot_left[1] - self.foot_left_knee[1], self.foot_left[0] - self.foot_left_knee[0])) / math.pi * 180
                        #print('左膝の角度は' + str(self.knee_degree))
        if((self.knee_degree >= -105 or self.knee_degree <= -255)):
            self.isDone = True
            #self.canCount = False
        else:
            self.isDone = False
            #self.canCount = True
        if(self.isDone):
            cv2.putText(image, 'Good', (0, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3, cv2.LINE_AA)
            image_1 = cv2.flip(image, 1)
        else:
            cv2.putText(image, 'Bad', (0, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
            image_1 = cv2.flip(image, 1)
        return image_1
