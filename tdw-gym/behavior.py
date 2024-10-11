from mmaction.apis import init_recognizer, inference_recognizer
from pycocotools import mask
import cv2
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"
detector = None
class tdw_behavior:
    def __init__(self):
        self.inferencer = init_recognizer(
            config = "detection_pipeline/config_behavior.py",
            checkpoint = "detection_pipeline/best_acc_top1_epoch_82.pth",
            device = "cuda"
        )
        name_list = [
            "pick up, success",
            "pick up, fail",
            "put on, success",
            "put on, fail",
            "put down, success",
            "put down, fail",
            "None",
            "walk, success",
        ]
        self.name_map = {i: name_list[i] for i in range(len(name_list))}

    def cls_to_name_map(self, cls_id):
        return self.name_map[cls_id]

    def images_to_video(self, images, video_name, fps=30):
        frame = cv2.imread(images[0])
        height, width, layers = frame.shape
        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for image in images:
            video.write(cv2.imread(image))

        cv2.destroyAllWindows()
        video.release()
        return video_name

    def __call__(self, video):
        # input can be a path or rgb image
        if type(video) == list:
            video = self.images_to_video(video, "temp.mp4")
        result = inference_recognizer(self.inferencer, video)
        label = result.get("pred_label").item()
        score = result.get("pred_score")[label].item()
        return label, self.cls_to_name_map(label), score, result.get("pred_score")

def init_behavior():
    global detector
    if detector == None:
        detector = tdw_behavior()
    return detector

def main():
    tdw = tdw_behavior()
    result = tdw("detection_pipeline/demo.mp4")
    print(result)
    #image_root = "results/high_goalplace_task_follow_helper/behavior_train/8/Images/1/"
    #images = []
    #for i in range(0, 50):
    #    images.append(image_root + "{:04}.png".format(i))
    #result = tdw(images)
    #print(result)
    
if __name__ == '__main__':
    main()