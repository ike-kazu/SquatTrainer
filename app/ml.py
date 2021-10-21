from utils import Singleton
import json
import trt_pose.coco
import trt_pose.models
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import torch
import torch2trt
from torch2trt import TRTModule
import cv2
import torchvision.transforms as transforms
import PIL.Image
import os

class ML(Singleton):
    count = 0
    def __init__(self):
        if ML.count == 0:
            with open('human_pose.json', 'r') as f:
                self.human_pose = json.load(f)
            self.topology = trt_pose.coco.coco_category_to_topology(self.human_pose)
            self.num_parts = len(self.human_pose['keypoints'])
            self.num_links = len(self.human_pose['skeleton'])

            MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
            OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

            print("loading model...")
            if os.path.exists(OPTIMIZED_MODEL):
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
            self.draw_objects(image, counts, objects, peaks)
            return image[:, ::-1, :]

