from fastapi import APIRouter, Response, UploadFile
import numpy as np
import cv2
from ..utils import feature_extraction
from ..utils import models

router = APIRouter(prefix="/api/v1")


def convert_to_bw(img):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, bwImg = cv2.threshold(grayImg, 127, 255, cv2.THRESH_BINARY)
    return bwImg

def slice_roi(img):
    # convert 0 to 1 , 1 to 0 to find border
    img2 = np.where(img==0, 255, 0)
    
    #axis=0 refer to column, axis=1 refer to row
    
    #find column with value >0 to find left/right border
    img2_col = np.sum(img2, axis = 0)
    col_with_value = np.where(img2_col > 0)[0]
    
    #find row with value >0 to find top/bottom border
    img2_row = np.sum(img2, axis = 1)
    row_with_value = np.where(img2_row > 0)[0]
    
    #get left/right border
    start_col = min(col_with_value)
    end_col = max(col_with_value)
    
    #get top/bottom border
    start_row = min(row_with_value)
    end_row = max(row_with_value)
    
    #get ROI width & height
    roi_width = end_col - start_col
    roi_height = end_row - start_row
    
    #get ROI area
    img = img[start_row:end_row, start_col:end_col]
    
    if roi_width > roi_height:
        diff = roi_width - roi_height
        extra_pad = int(diff/2)
        img = np.pad(img, ((extra_pad, extra_pad), (0,0)), 'maximum')
    else:
        diff = roi_height - roi_width
        extra_pad = int(diff/2)
        img = np.pad(img, ((0,0), (extra_pad, extra_pad)), 'maximum')
    return img


async def loadImage(file: UploadFile):
    img_str = await file.read()
    file.close()
    # CV2
    nparr = np.fromstring(img_str, np.uint8)
    # cv2.IMREAD_COLOR in OpenCV 3.1
    img_np = cv2.imdecode(nparr, flags=1)

    bwImg = convert_to_bw(img_np)
    croped = slice_roi(bwImg)
    croped = cv2.cvtColor(croped, cv2.COLOR_GRAY2BGR)

    image = cv2.resize(croped, (64, 64), interpolation=cv2.INTER_AREA)
    # tresholding binary
    ret, tresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # show to loaded image
    return tresh


async def predictImage(file: UploadFile, target: int):
    image = await loadImage(file)
    print(image)
    features = feature_extraction.ExtractFeature(image)
    accr, predResult = models.mlModel.forward([features])
    for i in range(0, len(accr[0])):
        print(accr[0][i])
    return models.y_labels[int(predResult[0])], float(1 - (models.mlModel.loss_func(accr, models.torch.tensor([target]))))


@router.post("/predict/:id")
async def predictImageHandler(
    res: Response,
    id: int,
    file: UploadFile
):
    print(id)
    result, result1 = await predictImage(file, id)
    return {
        "message": "success",
        "predictResult": result,
        "expectedResult": models.y_labels[id],
        "predictPercentage": result1
    }
