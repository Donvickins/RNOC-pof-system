import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

logger.info('Loading modules...')
import sys
import cv2
import torch
import argparse
from fuzzywuzzy import fuzz
from ultralytics import YOLO
from pathlib import Path
from pof.Utils import utils
from pof.GNN.GModel import GNN
logger.info('Modules loaded successfully...')

def load_gnn_model(model_path, in_channels, hidden_channels, num_edge_features):
    """
    Loads the trained GNN model and sets it to evaluation mode.
    """
    model = GNN(
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


def main():
    """
    Main function to process a network topology image and predict the point of failure.
    """
    # Check if image exists
    if image is None or image.size == 0:
        logger.error(f'Image not found or invalid at {image_path}')
        sys.exit(1)

    # Load models
    try:
        yolo_model = YOLO(yolo_model_path)
        logger.info(f'YOLO model loaded from: {yolo_model_path}')

        gnn_model = load_gnn_model(
            model_path=gnn_model_path,
            in_channels=11,  # 4 (type) + 6 (color + down_id)
            hidden_channels=128,
            num_edge_features=6  # Edge color features
        )
        logger.info(f'GNN model loaded from: {gnn_model_path}')

    except Exception as err:
        logger.error(f'Failed to load models: {err}')
        sys.exit(1)

    # Run YOLO model on the image
    yolo_data = yolo_model.predict(source=image, save=False, conf=0.5)

    # Process results to extract nodes and edges
    nodes, edges = utils.extract_data_from_YOLO(yolo_data, image)
    if len(nodes) == 0:
        logger.error('No nodes found in image or YOLO failed to process image')
        sys.exit(1)

    # Create graph tensors
    node_data = utils.create_node_tensor(nodes, down_id)
    edge_data = utils.create_edges_tensor(edges, node_data['node_centers'])

    # Validate graph
    if len(node_data['node_ids']) == 0:
        logger.error('No valid nodes extracted from image')
        sys.exit(1)

    if not isinstance(edge_data['edge_index'], torch.Tensor) or edge_data['edge_index'].shape[0] != 2:
        logger.error(f'Invalid edge_index: {edge_data["edge_index"]}')
        sys.exit(1)
    if edge_data['edge_index'].size(1) == 0:
        logger.warning('No valid edges detected in the graph, proceeding with isolated nodes.')

    # Fuzzy matching for down_id
    matches = []
    for id in node_data['node_ids']:
        score = fuzz.ratio(down_id, id)
        if score >= 70:
            matches.append({'score': score, 'id': id})

    logger.info(f'Detected Nodes: {node_data["node_ids"]}')
    logger.info(f'Matches for down_id "{down_id}": {matches}')

    if len(matches) == 0:
        logger.error(f'Site down with ID "{down_id}" not found in image')
        sys.exit(1)

    match = max(matches, key=lambda x: x['score'])
    closest_match = match['id']

    if match['score'] < 80:
        logger.error(f'Site down with ID "{down_id}" closest match "{closest_match}" has low similarity ({match["score"]}%)')
        sys.exit(1)

    # Feed extracted nodes and edges to the GNN
    with torch.no_grad():
        pof_logits, has_pof_logit = gnn_model(node_data['x'], edge_data['edge_index'], edge_data['edge_attr'])

    # Interpret predictions
    pred_result, pred_prob = interpret_pof_predictions(pof_logits, has_pof_logit, node_data['node_ids'])
    if pred_result == 'indeterminate':
        logger.info(f'Predicted Point of Failure: indeterminate')
        logger.info(f'Based on provided image, the probability of Site: {down_id} having no POF is: {(1-pred_prob) * 100:.2f}%')
    else:
        logger.info(f'Predicted Point of Failure, Site: {pred_result}')
        logger.info(f'Accuracy: {pred_prob * 100:.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Network topology image to be analysed', required=True)
    parser.add_argument('--site', help='The site Id of the down site', required=True)
    args = parser.parse_args()

    image_path = args.image.strip("'\"")
    down_site = args.site

    # Define image path
    if not Path(image_path).exists():
        logger.error('Image does not exist in path')
        sys.exit(1)

    if down_site is None:
        logger.error('No site down ID given')
        sys.exit(1)

    image = cv2.imread(str(image_path))
    down_id = down_site
    # Define model paths
    yolo_model_path = Path('models/YOLO/best.pt')
    gnn_model_path = Path('models/GNN/best.pt')

    main()