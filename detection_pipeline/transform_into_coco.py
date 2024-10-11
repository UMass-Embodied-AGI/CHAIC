import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from skimage import measure
from PIL import Image
import numpy as np
import os
import re
import json
import math
import shutil
import random
import multiprocessing
from joblib import Parallel, delayed
from pycocotools.coco import COCO
import argparse
print("Warning: be careful when using this ugly code :) Many values are hard-coded.")


def _load_annotation_json(json_path):
    with open(json_path, "r") as f:
        annotation_dict = json.load(f)

    cate = annotation_dict["categories"] # [{"id", "name"}]
    imgs = annotation_dict["images"] # [{"file_name", "id"}]
    anns = annotation_dict["annotations"] # [{"segmentation", 'image_id', 'category_id', 'area'}]
    return cate, imgs, anns


def _create_sub_masks(mask_image):
    width, height = mask_image.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))[:3]

            # If the pixel is not black...
            if pixel != (0, 0, 0):
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                   # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x+1, y+1), 1)
    return sub_masks


def _create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd=0):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(np.array(sub_mask), 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        if poly.geom_type == "Polygon":
            polygons.append(poly)
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            if len(segmentation):
                segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }
    return annotation


def _get_annotations(dir_path, categories, resolution=(512, 512), img_fold="0"):
    images = []
    annotations = []
    category_dict = {x["name"]: x["id"] for x in categories}
    ann_id = 0
    with open(os.path.join(dir_path, "color_obj_map.json")) as json_file:
        color_obj_map = json.load(json_file)

    # Remove unwanted objects
    # Hard code to remove tent and shelve in shopping
    removed_key = []
    for i in color_obj_map:
        if color_obj_map[i] == "b01_tent":
            removed_key.append(i)
            continue
        if color_obj_map[i] == "4ft_wood_shelving" and "shopping" in dir_path:
            removed_key.append(i)
    for i in removed_key:
        del color_obj_map[i]

    file_list = os.listdir(os.path.join(dir_path, img_fold))
    for i in file_list:
        img_id = int(re.findall(r"\d+", i)[0])
        # Images
        if i.startswith("img_"):
            images.append({
                "file_name": os.path.join(dir_path, img_fold, i),
                "height": resolution[0],
                "width": resolution[1],
                "id": img_id
            })
        # Annotations
        if i.startswith("id_"):
            seg_img = Image.open(os.path.join(dir_path, img_fold, i))
            sub_masks = _create_sub_masks(seg_img)
            for color, sub_mask in sub_masks.items():
                _key = "[%3s %3s %3s]" % tuple(color[1:-1].split(", "))
                if _key in color_obj_map:
                    ann_id += 1
                    category_id = category_dict[color_obj_map[_key]]
                    annotation = _create_sub_mask_annotation(sub_mask, img_id, category_id, ann_id)
                    annotations.append(annotation)
    return images, annotations, dir_path

# In COCO format
def create_annotation_json(name_map_path, base_path, json_name):
    cate = _get_categories(name_map_path)
    paths = [os.path.join(base_path, i) for i in os.listdir(base_path)]
    n_jobs = multiprocessing.cpu_count() - 2
    parallel_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_get_annotations)(_path, cate) for _path in paths
    )

    for _imgs, _anns, _dir in parallel_results:
        annotation_json = {"images": _imgs, "annotations": _anns, "categories": cate}
        print(os.path.join(_dir, json_name))
        with open(os.path.join(_dir, json_name), "w") as f:
            f.write(json.dumps(annotation_json))


def _get_categories(name_map_path):
    with open(name_map_path) as json_file:
        perception_obj_list = json.load(json_file).keys()
    categories = []
    for i in perception_obj_list: # Id 0 is for empty class
        categories.append({"id": len(categories) + 1, "name": i})
    return categories


def remove_repeating_imgs(dir_path, resolution=(512, 512), img_fold="0"):
    # Caused by fail to move
    # Extract seq ids
    imgs = os.listdir(os.path.join(dir_path, img_fold))
    seq = re.findall(r"\d+", str(imgs))
    seq = [int(x) for x in seq]
    seq = sorted(set(seq))
    print("total imgs:", len(seq))

    # Find repeating imgs
    remove_ids = []
    stats = []
    pre_img = np.array(Image.open(os.path.join(dir_path, img_fold, "img_0000.jpg")))
    for i in seq[1:]:
        cur_img = np.array(Image.open(os.path.join(dir_path, img_fold, "img_%04d.jpg" % i)))
        stats.append(np.sum(abs(pre_img - cur_img)))
        if np.sum(abs(pre_img - cur_img)) < (64 * resolution[0] * resolution[1] * 3): # for RGB images
            remove_ids.append(i)
        pre_img = cur_img
    # plt.hist(stats, bins=30)
    # plt.show()

    # Delete files
    img_types = ["img_%04d.jpg", "depth_%04d.png", "id_%04d.png"]
    for i in remove_ids:
        for j in img_types:
            os.remove(os.path.join(dir_path, img_fold, j % i))
    print("remain imgs:", len(os.listdir(os.path.join(dir_path, img_fold))) / 3)


def dataset_stats(dir_path, json_name):
    cate, imgs, anns = _load_annotation_json(os.path.join(dir_path, json_name))

    # Show category stats
    cate_name = {x["id"]: x["name"] for x in cate}
    cate_count = {x["id"]: 0 for x in cate}
    area_stats = {x["id"]: [] for x in cate}
    for i in anns:
        if len(i["segmentation"]):
            cate_count[i["category_id"]] += 1
            area_stats[i["category_id"]].append(i["area"])
    print("Obj Stats:")
    for i in cate_count:
        print(cate_name[i], cate_count[i])

    # save stats
    with open(os.path.join(dir_path, f"{json_name.split('.')[0]}Stats.json"), "w") as f:
        for i in cate_count:
            f.write(f"{cate_name[i]}: {cate_count[i]}\n")

    # Show bbox area hist
    # for i in area_stats:
    #     _stats = area_stats[i]
    #     plt.title(cate_name[i])
    #     _max = math.ceil(max(_stats)) if len(_stats) else 100
    #     _min = math.floor(min(_stats)) if len(_stats) else 0
    #     bins = range(_min, _max+1)
    #     counts, _, _ = plt.hist(_stats, bins=bins)
    #     plt.show()
    #
    #     max_idx = np.argmax(counts)
    #     print(max(counts), bins[max_idx], bins[max_idx + 1], i, cate_name[i])

    # Show blank images
    img_obj_count = {x["id"]: 0 for x in imgs}
    for i in anns:
        img_obj_count[i["image_id"]] += 1
    blank_img_ids = [x for x in img_obj_count if img_obj_count[x] == 0]
    print(f"Image without objects: {len(blank_img_ids)}/{len(imgs)}")


def visualization(dir_path, json_name):
    coco_annotation = COCO(annotation_file=os.path.join(dir_path, json_name))

    cat_ids = coco_annotation.getCatIds()
    cat_names = coco_annotation.loadCats(cat_ids)
    img_ids = coco_annotation.getImgIds()

    img_id = img_ids[1000]
    img_info = coco_annotation.loadImgs([img_id])[0]
    img_file_name = img_info["file_name"]

    # Get all the annotations for the specified image.
    ann_ids = coco_annotation.getAnnIds(imgIds=[img_id], iscrowd=None)
    anns = coco_annotation.loadAnns(ann_ids)

    # count = 0
    # for j in coco_annotation.anns.values():
    #     if ((j["category_id"] == 6) and (1540 < j["area"] < 1586)) or ((j["category_id"] == 1) and (160 < j["area"] < 215)):
    #         count += 1
    #         img_info = coco_annotation.loadImgs([j["image_id"]])[0]
    #         img_file_name = img_info["file_name"].split("/")
    #         plt.title(str(j["area"]))
    #         plt.axis("off")
    #         plt.imshow(np.array(Image.open(os.path.join("captured_imgs", img_file_name[0], "0", img_file_name[1]))))
    #         coco_annotation.showAnns([j], draw_bbox=False)
    #         plt.savefig(f"./show_1/{j['category_id']}_{count}.png")
    #         plt.clf()
    #         count += 1

    # Show images and annotations
    plt.axis("off")
    plt.imshow(np.array(Image.open(os.path.join(dir_path, img_file_name))))
    coco_annotation.showAnns(anns, draw_bbox=True)
    plt.show()
    plt.savefig("dataset")


def _modify_json(dir_path, json_name, img_fold="0"):
    cate, imgs, anns = _load_annotation_json(os.path.join(dir_path, json_name))

    # Filter small objs
    thres_dict = {x["id"]: 35 for x in cate}
    anns = list(filter(lambda x: len(x["segmentation"]) and (x["area"] >= thres_dict[x["category_id"]]), anns)) # Filter small objs and empty anns

    img_obj_count = {x["id"]: 0 for x in imgs}
    for i in anns:
        img_obj_count[i["image_id"]] += 1

    new_imgs = list(filter(lambda x: img_obj_count[x["id"]], imgs))
    new_anns = list(filter(lambda x: img_obj_count[x["image_id"]], anns))

    for i in new_imgs:
        _paths = i["file_name"].replace("\\", "/").replace("/" + img_fold, "").split("/")
        i["file_name"] = os.path.join(_paths[-2], _paths[-1]).replace("\\", "/")
    annotation_json = {"images": new_imgs, "annotations": new_anns, "categories": cate}
    return annotation_json


def _merge_jsons(jsons):
    max_img_id = 0
    max_ann_id = 0
    imgs = []
    annotations = []
    for _json in jsons:
        _imgs = _json["images"]  # [{"file_name", "id"}]
        _anns = _json["annotations"]  # [{"segmentation", 'image_id', 'category_id', 'area'}]
        for i in _imgs: i["id"] += max_img_id
        for i in _anns: i["id"] += max_ann_id
        for i in _anns: i["image_id"] += max_img_id

        max_img_id = max([x["id"] for x in _imgs]) + 1 if len(_imgs) else max_img_id + 1
        max_ann_id = max([x["id"] for x in _anns]) + 1 if len(_anns) else max_img_id + 1
        imgs.extend(_imgs)
        annotations.extend(_anns)
    annotation_json = {"images": imgs, "annotations": annotations, "categories": jsons[0]["categories"]}
    return annotation_json


# Move all necessary files into new folder to reduce total size.
def construct_dataset(source_dir, json_name, target_dir, img_fold="0"):
    modified_jsons = [
        _modify_json(os.path.join(source_dir, x), json_name, img_fold)
        for x in os.listdir(source_dir)
        if os.path.exists(os.path.join(source_dir, x, json_name))
    ]
    # + [
    #     _modify_json(os.path.join("captured_new", x), json_name, img_fold)
    #     for x in os.listdir("captured_new")
    #     if os.path.exists(os.path.join("captured_new", x, json_name))
    # ]

    print("dump json")
    os.makedirs(target_dir, exist_ok=True)
    _json = _merge_jsons(modified_jsons)
    with open(os.path.join(target_dir, json_name), "w") as f:
        f.write(json.dumps(_json))

    print("copy files")
    for _json in modified_jsons:
        _imgs = [x["file_name"] for x in _json["images"]]
        for i in _imgs:
            _paths = i.split("/")
            i = os.path.join(source_dir, _paths[0], img_fold, _paths[1])
            target_sub_dir = os.path.join(target_dir, _paths[0])
            os.makedirs(target_sub_dir, exist_ok=True)
            target_path = os.path.join(target_sub_dir, _paths[1])
            shutil.copy(i, target_path)
            shutil.copy(i.replace('img_', 'depth_').replace(".jpg", ".png"),
                        target_path.replace('img_', 'depth_').replace(".jpg", ".png"))


def balance_dataset(dir_path, json_name, thres=2000):
    cate, imgs, anns = _load_annotation_json(os.path.join(dir_path, json_name))
    cate2anns = {x["id"]: [] for x in cate}
    for i in anns:
        cate2anns[i["category_id"]].append(i)

    balanced_anns = []
    for k, v in cate2anns.items():
        if len(v) > thres:
            balanced_anns.extend(random.sample(v, k=thres))
        else:
            balanced_anns.extend(v)
    imgs_ids = list(x["image_id"] for x in balanced_anns)
    new_imgs = list(filter(lambda x: x["id"] in imgs_ids, imgs))
    new_anns = list(filter(lambda x: x["image_id"] in imgs_ids, anns))

    annotation_json = {"images": new_imgs, "annotations": new_anns, "categories": cate}
    with open(os.path.join(dir_path, f"{json_name.split('.')[0]}Balance{thres}.json"), "w") as f:
        f.write(json.dumps(annotation_json))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="tdw_helper_perception_dataset")
    parser.add_argument("--train_image_dir", type=str, default="captured_imgs_helper/train")
    parser.add_argument("--test_image_dir", type=str, default="captured_imgs_helper/test")
    parser.add_argument("--name_map_path", type=str, default="dataset/name_map.json")

    # Process
    parser.add_argument("--no_remove_repeating_imgs", action="store_true", default=False)
    parser.add_argument("--no_create_annotation_json", action="store_true", default=False)
    parser.add_argument("--no_construct_dataset", action="store_true", default=False)
    parser.add_argument("--no_balance_dataset", action="store_true", default=False)
    parser.add_argument("--no_dataset_stats", action="store_true", default=False)
    parser.add_argument("--no_visualization", action="store_true", default=False)
    args = parser.parse_args()

    if not args.no_remove_repeating_imgs:
        for i in os.listdir(args.train_image_dir):
            print(i)
            remove_repeating_imgs(os.path.join(args.train_image_dir, i))

        for i in os.listdir(args.test_image_dir):
            print(i)
            remove_repeating_imgs(os.path.join(args.test_image_dir, i))

    if not args.no_create_annotation_json:
        create_annotation_json(args.name_map_path, args.train_image_dir, "train.json")
        create_annotation_json(args.name_map_path, args.test_image_dir, "test.json")

    if not args.no_construct_dataset:
        construct_dataset(args.train_image_dir, "train.json", args.dataset_dir)
        construct_dataset(args.test_image_dir, "test.json", args.dataset_dir)

    if not args.no_balance_dataset:
        balance_dataset(args.dataset_dir, "train.json")
        balance_dataset(args.dataset_dir, "test.json")

    if not args.no_dataset_stats:
        dataset_stats(args.dataset_dir, "train.json")
        dataset_stats(args.dataset_dir, "test.json")
    
    if not args.no_visualization:
        visualization(args.dataset_dir, "train.json")