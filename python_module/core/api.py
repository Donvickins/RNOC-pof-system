import sys
import base64
import binascii
import logging
from fastapi import FastAPI, status, Request
from core.utils.schema import Request as pofRequest, Response as pofResponse
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from core.utils.pof import pof, prep_models
from pathlib import Path
from core.pof.GNN.GModel import GNN
from ultralytics import YOLO
from core.utils.exception_handler import InvalidImageException, SiteIdNotFoundInImage

logger = logging.getLogger(__name__)

app = FastAPI()

# Define model paths
yolo_model_path = Path().cwd() /'models/YOLO/best.pt'
gnn_model_path = Path().cwd() / 'models/GNN/best.pt'

yolo_model, gnn_model = prep_models(yolo_model_path, gnn_model_path)
if not isinstance(yolo_model, YOLO) or not isinstance(gnn_model, GNN):
    logger.error('[ERROR]: Failed to load models')
    sys.exit(1)

save_dir = Path('workspace/received_images')
save_dir.mkdir(exist_ok=True, parents=True)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()}
    )

@app.exception_handler(InvalidImageException)
async def image_exception_handler(request: Request, exc: InvalidImageException):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"message": "Invalid image provided"}
    )

@app.exception_handler(SiteIdNotFoundInImage)
async def site_id_exception_handler(request: Request, exc: SiteIdNotFoundInImage):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"message": "Site ID not found in image"}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": "An unexpected error occurred on the server."},
    )

@app.post('/pof')
async def check_pof(request: pofRequest):
    try:
        image_bytes = base64.b64decode(request.image_base64)
    except binascii.Error:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"message": "Invalid base64 string"})

    try:
        predicted_pof, accuracy = pof(image_bytes,request.site_id,yolo_model, gnn_model)
        accuracy = accuracy * 100
        accuracy = round(accuracy, 2)
    except InvalidImageException:
        raise
    except SiteIdNotFoundInImage as e:
        raise
    except Exception as e:
        logger.error(f'Unexpected error occurred while processing order id: {request.order_id}. Reason: {e}')
        raise

    try:
        image_path = save_dir / f'{request.order_id}.png'
        with open(image_path, 'wb') as file:
            file.write(image_bytes)
    except Exception as e:
        logger.error(f'Failed to save image received from OWS. Reason: {e}')

    return pofResponse(site_id=request.site_id, pof=predicted_pof, certainty=accuracy, order_id=request.order_id)

@app.get('/health')
async def health_check(request: Request):
    return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "OK OWS"})

if __name__ == '__main__':
    sys.exit(0)