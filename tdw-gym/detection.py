from mmdet.apis import DetInferencer
from pycocotools import mask
import cv2
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"
detector = None
class tdw_detection:
    def __init__(self):
        self.inferencer = DetInferencer(
            model = "detection_pipeline/config.py",
            weights = "detection_pipeline/epoch_4.pth",
            device = "cuda"
        )

        name_list = [
                'b04_bowl_smooth',
                'plate06',
                'teatray',
                'basket_18inx18inx12iin_plastic_lattice',
                'basket_18inx18inx12iin_wicker',
                'basket_18inx18inx12iin_wood_mesh',
                'basket_18inx18inx12iin_bamboo',
                'bread',
                'b03_burger',
                'b03_loafbread',
                'apple',
                'b04_banana',
                'b04_orange_00',
                'f10_apple_iphone_4',
                'b05_executive_pen',
                'key_brass',
                'apple_ipod_touch_yellow_vray',
                'b04_lighter',
                'small_purse',
                'b05_calculator',
                'pencil_all',
                'mouse_02_vray',
                'b04_backpack',
                '102_pepsi_can_12_fl_oz_vray',
                'b03_cocacola_can_cage',
                '104_sprite_can_12_fl_oz_vray',
                'fanta_orange_can_12_fl_oz_vray',
                'b01_croissant',
                'b03_pink_donuts_mesh',
                'b03_banan_001',
                'b04_red_grapes',
                '4ft_wood_shelving',
                'appliance-ge-profile-microwave3',
                'arflex_strips_sofa',
                'b03_grandpiano2014',
                'cabinet_24_door_drawer_wood_beach_honey',
                'dining_room_table',
                'dishwasher_4',
                'emeco_navy_chair',
                'hp_printer',
                'kettle_2',
                'sm_tv',
                'truck',
                'fire hydrant',
                'huffy_nel_lusso_womens_cruiser_bike_2011vray',
                'b01_tent',
                'vase_05',
                'bed',
                'cabinet',
                'refrigerator',
                'yellow_side_chair',
                'trashbin',
                'b05_fire_extinguisher',
        ]
        self.name_map = {i: name_list[i] for i in range(len(name_list))}

    def cls_to_name_map(self, cls_id):
        return self.name_map[cls_id]

    def __call__(self, img_path, decode = True, no_save_pred = True, out_dir = ''):
        # input can be a path or rgb image
        result = self.inferencer(img_path, no_save_pred = no_save_pred, out_dir = out_dir)
        if decode and result['predictions'][0]['masks'] != []:
            rle_format = result['predictions'][0]['masks']
            # print('rle:', rle_format)
            mask_format = mask.decode(rle_format)
            # print('mask:', mask_format)
            result['predictions'][0]['masks'] = mask_format
        return result

def init_detection():
    global detector
    if detector == None:
        detector = tdw_detection()
    return detector

from PIL import Image
def main():
    tdw = tdw_detection()
    result = tdw("detection_pipeline/example.png", decode = False, no_save_pred = False, out_dir = 'outputs')
    print(result)
    img = Image.open("detection_pipeline/example.png")
    img = np.array(img)[..., [2, 1, 0]]
    # need to change channel here.
    result = tdw(img, decode = True)
    print(result)
    print(result['predictions'][0]['masks'].shape)
    
if __name__ == '__main__':
    main()