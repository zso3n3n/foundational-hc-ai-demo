import json
import base64
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image
import urllib.request


def read_image(image_path):
    """Reads an image from a file path and returns the image as a byte array."""
    with open(image_path, "rb") as f:
        return f.read()


def score_image(inference_config, frontal_path, lateral_path=None, indication="", technique="", comparison="None"):
    """Scores frontal and lateral images using the deployed model."""

    # Prepare the request payload
    url = f"{inference_config['endpoint']}/score"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {inference_config['api_key']}",
    }
    deployment = inference_config.get("azureml_model_deployment", None)
    if deployment:
        headers["azureml-model-deployment"] = deployment
    
    if lateral_path:
        input_data = {
            "frontal_image": base64.encodebytes(read_image(frontal_path)).decode("utf-8"),
            "lateral_image": base64.encodebytes(read_image(lateral_path)).decode("utf-8"),
            "indication": indication,
            "technique": technique,
            "comparison": comparison,
        }
    else:
        input_data = {
            "frontal_image": base64.encodebytes(read_image(frontal_path)).decode("utf-8"),
            "indication": indication,
            "technique": technique,
            "comparison": comparison,
        }

    data = {
        "input_data": {
            "columns": list(input_data.keys()),
            "index": [0],
            "data": [
                list(input_data.values()),
            ],
        },
        "params": {},
    }

    body = str.encode(json.dumps(data))

    # Send the request and handle response
    req = urllib.request.Request(url, body, headers)
    try:
        response = urllib.request.urlopen(req)
        findings = response.read().decode("utf-8")
        
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

    return findings


def adjust_box_for_original_image_size(norm_box, width: int, height: int):
    """
    Assuming we did a centre crop to the shortest size, adjust the box coordinates back to the original shape of the image.
    :param norm_box: A normalized box to rescale.
    :param width: Original width of the image, in pixels.
    :param height: Original height of the image, in pixels.
    :return: The box normalized relative to the original size of the image.
    """
    crop_width = crop_height = min(width, height)
    x_offset = (width - crop_width) // 2
    y_offset = (height - crop_height) // 2
    norm_x_min, norm_y_min, norm_x_max, norm_y_max = norm_box
    abs_x_min = int(norm_x_min * crop_width + x_offset)
    abs_x_max = int(norm_x_max * crop_width + x_offset)
    abs_y_min = int(norm_y_min * crop_height + y_offset)
    abs_y_max = int(norm_y_max * crop_height + y_offset)
    adjusted_norm_x_min = abs_x_min / width
    adjusted_norm_x_max = abs_x_max / width
    adjusted_norm_y_min = abs_y_min / height
    adjusted_norm_y_max = abs_y_max / height
    return (
        adjusted_norm_x_min,
        adjusted_norm_y_min,
        adjusted_norm_x_max,
        adjusted_norm_y_max,
    )


def show_image_with_bbox(path_frontal, findings, path_lateral=None):
    """Displays frontal and lateral images with bounding boxes around the findings."""
    image_frontal = Image.open(path_frontal)
    width_frontal, height_frontal = image_frontal.size

    if path_lateral:
        image_lateral = Image.open(path_lateral)
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].imshow(image_frontal, cmap="gray")
        axes[1].imshow(image_lateral, cmap="gray")
    else:
        fig, axes = plt.subplots(figsize=(10, 10))
        axes.imshow(image_frontal, cmap="gray")
        axes = [axes]

    findings_str = []
    for idx, (finding, boxes) in enumerate(findings):
        findings_str.append(f"{idx}. {finding}{' * ' if boxes else ' '}")
        if boxes:
            for box in boxes:
                box = adjust_box_for_original_image_size(
                    box, width_frontal, height_frontal
                )
                x_min, y_min, x_max, y_max = (
                    box[0] * width_frontal,
                    box[1] * height_frontal,
                    box[2] * width_frontal,
                    box[3] * height_frontal,
                )

                rect = plt.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    edgecolor="red",
                    facecolor="none",
                    linewidth=2,
                )
                axes[0].add_patch(rect)
                axes[0].text(
                    x_min + 3,
                    y_min + 3,
                    f"Finding ID: {idx}",
                    color="yellow",
                    fontsize=10,
                    verticalalignment="top",
                )

    for ax in axes:
        ax.axis("off")  # Hide the axes

    return fig, findings_str