"""FastAPI backend exposing endpoints for license plate recognition using OpenCV.

Endpoints:
 - POST /api/process-image  -> upload image and process using OpenCV-based recognition
"""
from __future__ import annotations

import os
import uuid
import shutil
import logging
from pathlib import Path
from typing import Any

from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse

from .config import (
    INPUT_IMAGES_DIR,
    PROCESSED_IMAGES_DIR,
)
from .opencv_processor import process_image

logger = logging.getLogger(__name__)
app = FastAPI(title='LPR Backend with OpenCV')


# ensure input/output dirs exist
Path(INPUT_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
Path(PROCESSED_IMAGES_DIR).mkdir(parents=True, exist_ok=True)

# 挂载前端静态目录
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# 配置根路径返回首页（HTML）
@app.get("/")
async def read_index():
    return FileResponse("frontend/index.html")


@app.post('/api/process-image')
async def process_image_endpoint(file: UploadFile = File(...)):
    # Save uploaded file to INPUT_IMAGES_DIR
    raw_name = file.filename or 'uploaded'
    safe_name = os.path.basename(raw_name)
    filename = f"{uuid.uuid4().hex}_{safe_name}"
    dest_path = os.path.join(INPUT_IMAGES_DIR, filename)

    try:
        with open(dest_path, 'wb') as out_f:
            shutil.copyfileobj(file.file, out_f)
    except Exception as e:
        logger.exception('Failed to save uploaded file')
        raise HTTPException(status_code=500, detail='Failed to save uploaded file')
    finally:
        file.file.close()

    # 使用OpenCV处理器处理图像
    try:
        # 调用OpenCV处理器
        plate_path, plate_number = process_image(dest_path)
        
        # 校验结果有效性
        if not plate_path or plate_path == '':
            plate_path = '未生成车牌裁剪图'
        if not plate_number or plate_number == '':
            plate_number = '识别失败'

        return {
            'plate_number': plate_number,          # 识别的车牌号码
            'uploaded_image': dest_path,           # 上传的原始图像路径
            'plate_image_path': plate_path,        # 裁剪后的车牌图像路径
            'status': 'success'
        }
    except Exception as e:
        logger.exception('Image processing failed')
        raise HTTPException(status_code=500, detail=f'Processing error: {str(e)}')


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('backend.main:app', host='0.0.0.0', port=8000, reload=True)