"""
Author: Victor Chukwujekwu vwx1423235

This contains the business logic of the application. This is where the 3 models work together to achieve POF prediction
"""

import logging
import sys
import cv2
import torch
import numpy as np
from fuzzywuzzy import fuzz
from ultralytics import YOLO
from core.utils import helpers as utils
from core.pof.GNN.GModel import GNN
from core.utils.exception_handler import InvalidImageException, SiteIdNotFoundInImage

# Configure logging
logger = logging.getLogger(__name__)

NUM_NODE_FEATURES = 12  # 5 (type) + 6 (color)  + 1 (down_id)

def prep_models(yolo_model_path, gnn_model_path):
    """
    Prepares the YOLO and GNN models for use.
    """
    if not yolo_model_path.exists() and not gnn_model_path.exists():
        logger.error(f'Either YOLO or GNN model does not exist in: {yolo_model_path} and {gnn_model_path}')
        raise FileNotFoundError("Models does not exist in path")

    # Load models
    try:
        yolo_model = YOLO(yolo_model_path)
        yolo_model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Explicit device
        yolo_model.eval()

        logger.info(f'YOLO model loaded from: {yolo_model_path}')

        gnn_model = load_gnn_model( # from .pof.GNN.GModel import GNN
            model_path=gnn_model_path,
            in_channels=NUM_NODE_FEATURES,
            hidden_channels=128,
            num_edge_features=6  # Edge color features
        )

        gnn_model.to('cuda' if torch.cuda.is_available() else 'cpu')
        gnn_model.eval()
        logger.info(f'GNN model loaded from: {gnn_model_path}')

    except Exception as err:
        logger.error(f'Failed to load models: {err}')
        raise

    return yolo_model, gnn_model


def load_gnn_model(model_path, in_channels, hidden_channels, num_edge_features):
    """
    Loads the trained GNN model and sets it to evaluation mode.
    """
    if not model_path.exists():
        logger.error('GNN model path does not exist')
        raise FileNotFoundError(f'GNN model path does not exist in: {model_path}')

    model = GNN( # from .pof.GNN.GModel import GNN
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_edge_features=num_edge_features
    )

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def interpret_pof_predictions(pof_logits, has_pof_logit, node_ids):
    """
    Interprets the POF prediction output, considering indeterminate cases.
    """
    has_pof_prob = torch.sigmoid(has_pof_logit).item()
    if has_pof_prob < 0.5:
        return 'indeterminate', has_pof_prob
    probs = torch.sigmoid(pof_logits)
    pred_idx = torch.argmax(probs).item()
    pred_site_id = node_ids[pred_idx]
    pred_prob = probs[pred_idx].item()
    return pred_site_id, pred_prob

def pof(image, down_id: str, yolo_model, gnn_model):
    """
    Main function to process a network topology image and predict the point of failure.
    """
    # Check if image exists
    if image is None:
        logger.error('Invalid image')
        raise InvalidImageException('Image invalid')

    if not down_id:
        logger.error('Empty or invalid site ID')
        raise InvalidImageException('Empty or invalid site ID')

    image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)

    if image is None or image.size == 0:
        logger.error('Invalid image')
        raise InvalidImageException('Image is not valid')

    # Run YOLO model on the image
    yolo_data = yolo_model.predict(source=image, save=False, verbose=False, imgsz=800,device='cuda' if torch.cuda.is_available() else 'cpu')
    result = yolo_data[0]

    # b) class-wise counts
    cls_names = result.names                # {0: "RTN_Green", 1: "RTN_Red", ...}
    cls_ids   = result.boxes.cls.int().cpu().tolist()
    counts = {}
    for cid in cls_ids:
        name = cls_names[cid]
        counts[name] = counts.get(name, 0) + 1

    count_str = ", ".join(f"{v} {k}" for k, v in counts.items())
    log_line = (
        f"Detections: {count_str}"
    )
    logger.info(log_line)
    # Process results to extract nodes and edges
    nodes, edges = utils.extract_data_from_YOLO(yolo_data, image)
    if len(nodes) == 0:
        logger.error('No nodes found in image or YOLO failed to process image')
        raise InvalidImageException('No nodes found in image')

    # Create graph tensors
    node_data = utils.create_node_tensor(nodes, down_id)
    edge_data = utils.create_edges_tensor(edges, node_data['node_centers'])

    # Validate graph
    if len(node_data['node_ids']) == 0:
        logger.error('No valid nodes extracted from image')
        raise ValueError('No valid nodes extracted from image')

    if not isinstance(edge_data['edge_index'], torch.Tensor) or edge_data['edge_index'].shape[0] != 2:
        logger.error(f'Invalid edge_index: {edge_data["edge_index"]}')
        raise ValueError('Invalid edge index')

    if edge_data['edge_index'].size(1) == 0:
        logger.warning('No valid edges detected in the graph, proceeding with isolated nodes.')

    # Fuzzy matching for down_id
    matches = []
    for id in node_data['node_ids']:
        score = fuzz.ratio(down_id, id)
        if score >= 70:
            matches.append({'score': score, 'id': id})

    # logger.info(f'Detected Nodes: {node_data["node_ids"]}')
    # logger.info(f'Matches for down_id "{down_id}": {matches}')

    if len(matches) == 0:
        logger.error(f'Site down with ID: "{down_id}" not found in image')
        raise SiteIdNotFoundInImage(f'Site down with ID: "{down_id}" not found in image')

    match = max(matches, key=lambda x: x['score'])
    closest_match = match['id']

    if match['score'] < 70:
        logger.error(f'Site down with ID "{down_id}" closest match "{closest_match}" has low similarity ({match["score"]}%)')
        raise SiteIdNotFoundInImage(f'Site down with ID "{down_id}" closest match "{closest_match}" has low similarity ({match["score"]}%)')

    # Feed extracted nodes and edges to the GNN
    with torch.no_grad():
        pof_logits, has_pof_logit = gnn_model(node_data['x'], edge_data['edge_index'], edge_data['edge_attr'])

    # Interpret predictions
    return interpret_pof_predictions(pof_logits, has_pof_logit, node_data['node_ids'])

if __name__ == "__main__":
    sys.exit(0)