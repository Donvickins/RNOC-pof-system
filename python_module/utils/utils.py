import pytesseract
import logging
import cv2
import sys
import torch
import numpy as np
from scipy.spatial.distance import cdist
from typing import Union
from pathlib import Path
from fuzzywuzzy import fuzz
from python_module.utils.exception_handler import InvalidImageException

logger = logging.getLogger(__name__)

# --- Constants and Mappings ---
TYPE_MAP = {'ATN': [1, 0, 0, 0], 'RTN': [0, 1, 0, 0], 'Router': [0, 0, 1, 0], 'Switch': [0, 0, 0, 1]}
COLOR_MAP = {'Red': [1, 0, 0, 0, 0, 0], 'Green': [0, 1, 0, 0, 0, 0], 'Blue': [0, 0, 1, 0, 0, 0],
             'Yellow': [0, 0, 0, 1, 0, 0], 'Orange': [0, 0, 0, 0, 1, 0], 'Gray': [0, 0, 0, 0, 0, 1]}
MAX_DIST_THRESH = 50

def extract_text(img) -> str | None:
    """
    Performs OCR using Tesseract

    We use a custom configuration to specify character set and OCR mode

    *custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'*
    Args:
        img (cv2.Mat): OpenCV Image

    Returns:
        str | None: A string containing the extracted data or None if OCR fails.
    """
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'

    try:
        text = pytesseract.image_to_string(img, config=custom_config, lang='pof_ocr')
        # Clean up the output by stripping whitespace and newlines
        text_s = text.strip()
        idx = text_s.find('_')
        if idx == -1:
            idx = text_s.find('-')
            if idx == -1:
                id = text_s
            else:
                id = text_s[:idx]
        else:
            id = text_s[:idx]

        if id:
            return id
        else:
            return 'invalid'
    except pytesseract.TesseractNotFoundError:
        logger.error("Tesseract is not installed or not in your PATH. Please install it.")
        return None


def site_id_2_binary(img) -> cv2.Mat:
    """
    This converts an image into black and white color, processes the image to be used for OCR

    :param
        img: This is an OpenCV image
    :return:
        cv2.Mat: This is the binary form of the input image
    """
    if img is None:
        logger.error("Image is empty")
        raise InvalidImageException("Image is empty")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 38, 120])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.resize(mask, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
    return mask

def get_site_id_from_image(image_path: Union[str, Path, cv2.Mat], node_bbox):
    """
    Crops the site ID from a node image and performs OCR.
    Args:
        image_path (str): The path to the full topology image.
        node_bbox (list): Bounding box of the detected node in the format [x_min, y_min, x_max, y_max].

    Returns:
        str | Image | None : The Site ID from the image or String 'out of bounds' if its out of bounds of image, or None if image does not exist or is invalid.
    """
    # Load the image
    img = None

    if isinstance(image_path, (str, Path)):
        img = cv2.imread(str(image_path))
    elif isinstance(image_path, np.ndarray):
        img = image_path

    if img is None:
        logger.error(f"Error: Could not load image from {image_path}")
        return None

    x_min, y_min, x_max, y_max = [int(coord) for coord in node_bbox]

    if y_min < y_max and x_min < x_max:
        detected_image = img[y_min:y_max, x_min:x_max]
    else:
        logger.warning("Detected object region is out of bounds or invalid.")
        return None

    row_start_raw = y_max + 1
    row_end_raw = row_start_raw + 22
    col_start_raw = x_min - 30
    col_end_raw = x_max - 1

    row_start = row_start_raw
    row_end = row_end_raw
    col_start = col_start_raw
    col_end = col_end_raw

    img_height, img_width = img.shape[:2]
    if (0 <= row_start < img_height and 0 <= row_end <= img_height and 0 <= col_start < img_width and 0 <= col_end <= img_width and
    row_start < row_end and col_start < col_end):
        site_id_image = img[row_start:row_end, col_start:col_end]
    else:
        logger.warning("Site ID region is out of bounds or invalid.")
        return 'out of bounds'

    if site_id_image.size == 0:
        logger.warning("Cropped image is empty. Bounding box may be invalid.")
        return 'out of bounds'

    return site_id_image

def get_site_id_from_node(image_path: Union[str, Path, cv2.Mat], node_bbox) -> str | None:
    """
    Crops the site ID from a node image and performs OCR.
    Args:
        image_path (str): The path to the full topology image.
        node_bbox (list): Bounding box of the detected node in the format [x_min, y_min, x_max, y_max].

    Returns:
        str | None: The extracted site ID, or None if OCR fails.
    """
    site_id_image = get_site_id_from_image(image_path, node_bbox)

    # Convert the cropped image to grayscale for better OCR performance
    site_id_binary_image = site_id_2_binary(site_id_image)
    txt = extract_text(site_id_binary_image)

    if txt is None:
        raise InvalidImageException('Invalid image from OCR extraction')
    return txt

def get_class_name(result, c_id) -> str | None:
    for i, c_name in result.names.items():
        if int(c_id) == i:
            return c_name
    return None


def create_node_tensor(nodes: list, down_id: str = None) -> dict:
    if not nodes:
        logger.info('No nodes found, cannot create a graph.')
        raise InvalidImageException('No nodes found in image')

    node_features = []
    node_centers = []
    node_ids = []
    for node in nodes:
        feature = TYPE_MAP.get(node['type']) + COLOR_MAP.get(node['color'])
        is_down = 1 if down_id and fuzz.ratio(down_id, node['id']) >= 80 else 0
        feature.append(is_down)
        node_centers.append(node['center'])
        node_ids.append(node['id'])
        node_features.append(feature)

    x = torch.tensor(node_features, dtype=torch.float)
    node_centers = np.array(node_centers)

    return {
        'x': x,
        'node_centers': node_centers,
        'node_ids': node_ids
    }


def create_edges_tensor(edges: list, node_centers: list) -> dict:
    if len(node_centers) == 0:
        logger.warning("No node centers provided, cannot create edges.")
        return {
            'edge_index': torch.empty((2, 0), dtype=torch.long).contiguous(),
            'edge_attr': torch.empty((0, 6), dtype=torch.float)
        }

    if len(edges) == 0:
        logger.warning("No edges provided, returning empty edge tensors.")
        return {
            'edge_index': torch.empty((2, 0), dtype=torch.long).contiguous(),
            'edge_attr': torch.empty((0, 6), dtype=torch.float)
        }

    edge_list = []
    edge_attributes = []
    discarded_edges = 0

    for edge in edges:
        ep1, ep2 = edge['endpoints']
        dist1 = cdist([tuple(ep1)], node_centers, 'euclidean')
        dist2 = cdist([tuple(ep2)], node_centers, 'euclidean')
        src_idx = np.argmin(dist1)
        tgt_idx = np.argmin(dist2)

        if dist1[0, src_idx] <= MAX_DIST_THRESH and dist2[0, tgt_idx] <= MAX_DIST_THRESH and src_idx != tgt_idx:
            edge_list.extend([[src_idx, tgt_idx], [tgt_idx, src_idx]])
            attr = COLOR_MAP.get(edge['color'])
            edge_attributes.extend([attr, attr])
        else:
            discarded_edges += 1
            reason = (
                f"Distance to nearest nodes ({dist1[0, src_idx]:.2f}, {dist2[0, tgt_idx]:.2f}) exceeds threshold "
                f"{MAX_DIST_THRESH} or connects same node (src_idx={src_idx}, tgt_idx={tgt_idx})"
            )
            logger.debug(f"Edge discarded: {reason}")

    if discarded_edges > 0:
        logger.warning(f"Discarded {discarded_edges} edges due to invalid connections.")

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long).contiguous()
    edge_attr = torch.tensor(edge_attributes, dtype=torch.float) if edge_attributes else torch.empty((0, 6), dtype=torch.float)

    return {
        'edge_index': edge_index,
        'edge_attr': edge_attr
    }

def extract_data_from_YOLO(result: list, img: Union[str, Path]) -> list:

    if len(result) == 0 and not img:
        logger.info('Result and image cannot be empty')
        raise InvalidImageException('Image does not contain any valid nodes')

    boxes = result[0].boxes
    nodes = []
    edges = []

    for key, item in enumerate(boxes.data):
        class_id = item[-1]
        class_name = get_class_name(result=result[0], c_id=class_id)
        bbox = boxes.data[key][:4]
        center = boxes.xywh[key][0:2].cpu().detach().numpy().tolist()
        color = class_name.split('_')[-1]

        if class_name.startswith('Link'):
            x_min, y_min, x_max, y_max = boxes.xyxy[key].cpu().detach().numpy().tolist()
            edge = {'color': color, 'endpoints': [(x_min, y_min), (x_max, y_max)]}
            edges.append(edge)
        elif any(class_name.startswith(p) for p in ['ATN', 'RTN', 'Router', 'Switch']):
            node_type = class_name.split('_')[0]
            site_id = get_site_id_from_node(image_path=img, node_bbox=bbox)
            node = {'id': site_id, 'type': node_type, 'color': color, 'center': center}
            nodes.append(node)

    return [nodes, edges]

if __name__ == '__main__':
    sys.exit(0)