import numpy as np
import cv2
from matplotlib import pyplot as plt

from camera_calib.utils.vizualization import visualize
from camera_calib.utils.image import denormalize
from camera_calib.utils.masks import _points_from_mask
from camera_calib.utils.homography import get_perspective_transform
from camera_calib.utils.homography import warp_image
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
    # Load model
    kp_model = KeypointDetectorModel(
        backbone="efficientnetb3",
        num_classes=29,
        input_shape=(320, 320),
    )

    WEIGHTS_NAME = "../models/FPN_efficientnetb3_0.0001_8.h5"
    kp_model.load_weights(WEIGHTS_NAME)

    # Load template image
    template = cv2.imread("../resources/world_cup_template.png")
    template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    template = cv2.resize(template, (1280, 720)) / 255.0

    template = rgb_template_to_coord_conv_template(template)

    # Load image
    image = load_image("../data/images/BaltykSwit (2).jpg")
    pr_mask = kp_model(image)
    homography_viz(image, pr_mask, template)


if __name__ == "__main__":
    main()
