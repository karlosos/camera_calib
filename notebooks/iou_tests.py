"""
Launch from notebooks/ directory

Calculating IoU based on visible warped template.
"""


import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

from camera_calib.utils.vizualization import visualize
from camera_calib.utils.image import denormalize
from camera_calib.utils.masks import _points_from_mask
from camera_calib.utils.homography import get_perspective_transform
from camera_calib.utils.homography import warp_image
from camera_calib.utils.homography import get_four_corners
from camera_calib.utils.homography import compute_homography
from camera_calib.utils.image import (
    torch_img_to_np_img,
    np_img_to_torch_img,
    denormalize,
)
from camera_calib.utils.utils import to_torch
from camera_calib.utils.vizualization import merge_template
from camera_calib.models.keras_models import KeypointDetectorModel
from camera_calib.utils.vizualization import rgb_template_to_coord_conv_template


def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def calculate_keypoints_mask(image, kp_model):
    pr_mask = kp_model(image)
    # visualize(
    #         image=denormalize(image.squeeze()),
    #         pr_mask=pr_mask[..., -1].squeeze(),
    #     )
    return pr_mask


def homography_viz(image, pr_mask, template):
    src, dst = _points_from_mask(pr_mask[0])
    pred_homo = get_perspective_transform(dst, src)
    pred_warp = warp_image(
        cv2.resize(template, (320, 320)), pred_homo, out_shape=(320, 320)
    )
    visualize(
        image=denormalize(image.squeeze()),
        warped_homography=pred_warp,
    )

    test = merge_template(
        image / 255.0, cv2.resize(pred_warp, (image.shape[1], image.shape[0]))
    )
    visualize(image=test)
    return pred_homo


def main():
    iou_scores = []

    # Load model
    kp_model = KeypointDetectorModel(
        backbone="efficientnetb3",
        num_classes=29,
        input_shape=(320, 320),
    )

    WEIGHTS_NAME = "../models/FPN_efficientnetb3_0.0001_8_427.h5"
    kp_model.load_weights(WEIGHTS_NAME)

    # Files
    filenames = os.listdir("../data/homography/test_img/")
    filenames = [str(filename).replace(".jpg", "") for filename in filenames]

    for file_name in filenames:
        print(file_name)
        image = load_image(f"../data/homography/test_img/{file_name}.jpg")

        # Load homography for image
        homo = np.load(f"../data/homography/test_homo/{file_name}_homo.npy")

        # Load template
        template = cv2.imread("../resources/world_cup_template.png")
        template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
        template = cv2.resize(template, (1280, 720)) / 255.0
        template = rgb_template_to_coord_conv_template(template)

        # Warp template image
        warp = warp_image(np_img_to_torch_img(template), to_torch(homo), method="torch")
        warp = torch_img_to_np_img(warp[0])

        # Transform template on image
        # visualize(
        #     warp=warp,
        #     template=template,
        #     image=image,
        # )

        # Find keypoints and homography
        pr_mask = kp_model(image)
        src, dst = _points_from_mask(pr_mask[0])
        try:
            pred_homo = get_perspective_transform(dst, src)
            print("pred_homo", pred_homo)

            # homography_viz(image, pr_mask, template)

            # Transform back transformed template
            pred_warp = warp_image(
                cv2.resize(template, (320, 320)), pred_homo, out_shape=(320, 320)
            )
            # Warped back to model coordinates for comparing in model coordinates
            pred_warp_model = warp_image(
                pred_warp, np.linalg.inv(pred_homo), out_shape=(320, 320)
            )
            # Visualize two warped templates
            # visualize(ground_truth=cv2.resize(warp, (320, 320)), prediction=pred_warp)

            # Calculating IoU
            warp_resized = cv2.resize(warp, (320, 320))
            intersection = np.logical_and(warp_resized, pred_warp)
            union = np.logical_or(warp_resized, pred_warp)
            iou_score = np.sum(intersection) / np.sum(union)
            print("IoU", iou_score)
            iou_scores.append(iou_score)
            print("Mean IoU", np.mean(np.array(iou_scores)))
        except:
            # If there were less than 4 points
            print("IoU", 0)
            iou_scores.append(0)
            print("Mean IoU", np.mean(np.array(iou_scores)))


if __name__ == "__main__":
    main()
