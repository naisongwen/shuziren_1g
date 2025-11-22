import argparse
import copy
import math
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time
import warnings
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image

os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO

from torchalign import FacialLandmarkDetector

from video_utils import get_video_fps, decompose_tfm, img_warp, metrix_M, laplacianSmooth


warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))


class Runner():
    
    def __init__(self, args, img_size=(256, 256)):
        self.device = device

        self.face_det = YOLO(f'{args.pretrained_model_dir}/yolov8n-face/yolov8n-face.pt')

        lmk_net = FacialLandmarkDetector(f'{args.pretrained_model_dir}/wflw/hrnet18_256x256_p1/')
        lmk_net = lmk_net.to(self.device)
        self.lmk_net = lmk_net.eval()

        self.pads = args.pads

        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        if args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
            self.fps = args.fps
        else:
            self.fps = get_video_fps(args.face)
        print(f'input video fps:{self.fps}')
        self.face = args.face

        if self.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
            frame = cv2.imread(self.face)
            print(f'Read image as face {self.face}')
            self.full_frames = [frame]
        else:
            os.makedirs(f"{self.save_dir}/full_imgs", exist_ok=True)
            print(f'video2imgs:{args.face}')
            self.video2imgs(args.face, f"{self.save_dir}/full_imgs", ext = 'png')
            '''
            video_stream = cv2.VideoCapture(self.face)
            print(f'Read video as face {self.face}')
            self.full_frames = []
            while True:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                self.full_frames.append(frame)
            '''
        self.frame_info = {}

        self.kpts_smoother = laplacianSmooth()
        self.abox_smoother = laplacianSmooth()

        self.img_size = img_size

        self.avatar_name = args.avatar_name

    def video2imgs(self,vid_path, save_path, ext = '.png',cut_frame = 10000000):
        cap = cv2.VideoCapture(vid_path)
        count = 0
        while True:
            if count > cut_frame:
                break
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
                count += 1
            else:
                cap.release()
                break

    @staticmethod
    def landmark_to_keypoints(landmark):
        lefteye = np.mean(landmark[60:68, :], axis=0)
        righteye = np.mean(landmark[68:76, :], axis=0)
        nose = landmark[54, :]
        leftmouth = (landmark[76, :] + landmark[88, :]) / 2
        rightmouth = (landmark[82, :] + landmark[92, :]) / 2
        return (lefteye, righteye, nose, leftmouth, rightmouth)

    @torch.no_grad()
    def detect_face(self, face_img):
        boxes = self.face_det(face_img,
                              imgsz=640,
                              conf=0.01,
                              iou=0.5,
                              half=True,
                              augment=False,
                              device=self.device)[0].boxes
        bboxes = boxes.xyxy.cpu().numpy()
        return bboxes

    @torch.no_grad()
    def detect_lmk(self, image, bbox=None):
        if isinstance(bbox, list):
            bbox = np.array(bbox)
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        bbox_tensor = torch.from_numpy(bbox[:, :4])
        landmark = self.lmk_net(img_pil, bbox=bbox_tensor, device=self.device).cpu().numpy()
        return landmark

    @torch.no_grad()
    def get_input_imginfo(self,image_folder):
        imginfo=[]
        os.makedirs(f"{self.save_dir}/face_imgs", exist_ok=True)
        os.makedirs(f"{self.save_dir}/align_imgs", exist_ok=True)
        full_imgs_path=f"{self.save_dir}/full_imgs"
        input_img_list = sorted(glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
        start_pro = time.time()
        for image_path in tqdm(input_img_list):
            file_name_without_extension = os.path.splitext(os.path.basename(image_path))[0]
            #frame = self.full_frames[idx].copy()
            frame= cv2.imread(image_path)
            bbox = self.detect_face(frame.copy())[0][:5]
            landmark = self.detect_lmk(frame.copy(), [bbox])[0]
            keypoints = self.landmark_to_keypoints(landmark)

            keypoints = self.kpts_smoother.smooth(np.array(keypoints))

            m = metrix_M(face_size=200, expand_size=256, keypoints=keypoints)

            align_frame = img_warp(frame, m, 256, adjust=0)
            align_bbox = self.detect_face(align_frame.copy())[0][:4]

            align_bbox = self.abox_smoother.smooth(np.reshape(align_bbox, (-1, 2))).reshape(-1)

            # 重新warp 图片，保持scale 不变
            w, h = 256, 256
            rt, s = decompose_tfm(m)
            s_x, s_y = s[0][0], s[1][1]
            m = rt
            align_frame = cv2.warpAffine(frame, m, (math.ceil(w / s_x), math.ceil(h / s_y)), flags=cv2.INTER_CUBIC)
            inv_m = cv2.invertAffineTransform(m)

            face = copy.deepcopy(align_frame)
            h, w, c = align_frame.shape
            bbox = align_bbox
            bbox[0] *= (w - 1) / 255
            bbox[1] *= (h - 1) / 255
            bbox[2] *= (w - 1) / 255
            bbox[3] *= (h - 1) / 255

            rect = [round(f) for f in bbox[:4]]
            pady1, pady2, padx1, padx2 = self.pads
            y1 = max(0, rect[1] - pady1)
            y2 = min(h, rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(w, rect[2] + padx2)

            coords = (y1, y2, x1, x2)
            face = face[y1:y2, x1:x2]

            face = cv2.resize(face, self.img_size)
            
            face_imgs_path = f"{self.save_dir}/face_imgs"
            align_imgs_path = f"{self.save_dir}/align_imgs"
            cv2.imwrite(f"{align_imgs_path}/{file_name_without_extension}.png", align_frame)
            cv2.imwrite(f"{face_imgs_path}/{file_name_without_extension}.png", face)
            
            imginfo.append({
                    #'img': face,
                    #'frame': frame,
                    'coords': coords,
                    #'align_frame': align_frame,
                    'm': m,
                    'inv_m': inv_m,
            })
        
        end_pro = time.time()
        print('视频预处理耗时(s):', end_pro - start_pro)
        return imginfo

    def osmakedirs(self,path_list):
        for path in path_list:
            os.makedirs(path) if not os.path.exists(path) else None

    def run(self):
        import pickle
        frame_info_list=self.get_input_imginfo(f"{self.save_dir}/full_imgs")
        first_frame=f"{self.save_dir}/full_imgs/00000000.png"
        if not os.path.exists(first_frame):
            print('视频处理失败，请检查视频，比如每一帧都要有人脸，视频名字不能有中文等')
            return
        frame=cv2.imread(first_frame)
        frame_h,frame_w = frame.shape[:2]
        
        data = {
            'fps': self.fps,
            'frame_num': len(frame_info_list),
            'frame_h': frame_h,
            'frame_w': frame_w,
            'frame_info_list': frame_info_list
        }
        
        avatar_info_file = os.path.join(f"{self.save_dir}/{self.avatar_name}.pkl")
        with open(avatar_info_file, 'wb') as f:
            pickle.dump(data, f)

def main(video_path,avatar_name,save_dir,pads=0):
    parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')
    parser.add_argument('--face', type=str, help='Filepath of video/image that contains faces to use', default=video_path)
    parser.add_argument('--avatar_name', type=str, help='avatar name', default=avatar_name)
    parser.add_argument('--save_dir', type=str, default=save_dir)
    parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', default=25., required=False)
    pad_area=[pads*5, pads*5, pads*5, pads*5]
    parser.add_argument('--pads', nargs='+', type=int, default=pad_area, help='Padding (top, bottom, left, right). Please adjust to include chin at least')

    parser.add_argument('--pretrained_model_dir', type=str, default='weights', help='')

    args = parser.parse_args()
    runner = Runner(args)
    runner.run()

