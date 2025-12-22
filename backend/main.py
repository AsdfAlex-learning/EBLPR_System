"""FastAPI backend exposing endpoints that bridge to MATLAB via matlab_client.

Endpoints:
 - GET  /api/matlab/status  -> returns whether matlab.engine is installed and running
 - POST /api/matlab/start   -> start MATLAB engine (sync)
 - POST /api/matlab/stop    -> stop MATLAB engine
 - POST /api/matlab/call    -> call arbitrary MATLAB function (JSON body)
 - POST /api/process-image  -> upload image and call matlab_core/process_image
"""
from __future__ import annotations

import os
import uuid
import shutil
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from .config import (
    INPUT_IMAGES_DIR,
    PROCESSED_IMAGES_DIR,
)
from . import matlab_client

logger = logging.getLogger(__name__)
app = FastAPI(title='LPR Backend with MATLAB bridge')


# ensure input/output dirs exist
Path(INPUT_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
Path(PROCESSED_IMAGES_DIR).mkdir(parents=True, exist_ok=True)


@app.get('/api/matlab/status')
async def matlab_status():
    has_engine = getattr(matlab_client, '_HAS_MATLAB', False)
    running = matlab_client.client.is_running() if has_engine else False
    return {'engine_installed': has_engine, 'engine_running': running}


@app.post('/api/matlab/start')
async def matlab_start():
    if not getattr(matlab_client, '_HAS_MATLAB', False):
        raise HTTPException(status_code=503, detail='matlab.engine not installed')
    try:
        matlab_client.client.start()
        return {'status': 'started'}
    except Exception as e:
        logger.exception('Failed to start MATLAB engine')
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/matlab/stop')
async def matlab_stop():
    try:
        matlab_client.client.stop()
        return {'status': 'stopped'}
    except Exception as e:
        logger.exception('Failed to stop MATLAB engine')
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/matlab/call')
async def matlab_call(payload: dict) -> Any:
    """Call arbitrary MATLAB function. JSON payload should be: {"func": "name", "args": [..], "nargout": 1}
    Arguments are passed as-is; complex types may need conversion in client code.
    """
    func = payload.get('func')
    args = payload.get('args', [])
    nargout = int(payload.get('nargout', 1))

    if not func:
        raise HTTPException(status_code=400, detail='Missing func in payload')

    if not getattr(matlab_client, '_HAS_MATLAB', False):
        raise HTTPException(status_code=503, detail='matlab.engine not installed')

    try:
        result = matlab_client.client.call_function(func, *args, nargout=nargout)
        return {'result': str(result)}
    except Exception as e:
        logger.exception('MATLAB call failed')
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/process-image')
async def process_image(file: UploadFile = File(...)):
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

    # If matlab not installed, return uploaded path for frontend preview
    if not getattr(matlab_client, '_HAS_MATLAB', False):
        return JSONResponse(status_code=503, content={'error': 'matlab.engine not installed', 'uploaded': dest_path})

    # ========== 核心修改：调用MATLAB的process_image，获取两个返回值（车牌路径+识别号码） ==========
    try:
        # MATLAB的process_image返回两个输出参数：plate_path 和 plate_number
        # nargout=2 表示获取2个输出参数
        matlab_result = matlab_client.client.call_function(
            'process_image', dest_path, nargout=2
        )
        
        # 解析MATLAB返回的结果（matlab.engine返回的是元组，顺序对应MATLAB函数的输出）
        plate_path_matlab = str(matlab_result[0])  # 裁剪后的车牌图像路径
        plate_number = str(matlab_result[1])       # MATLAB识别出的车牌号码

        # 校验结果有效性
        if not plate_path_matlab or plate_path_matlab == '':
            plate_path_matlab = '未生成车牌裁剪图'
        if not plate_number or plate_number == '':
            plate_number = 'MATLAB识别失败'

        return {
            'plate_number': plate_number,          # 新增：MATLAB识别的车牌号码
            'uploaded_image': dest_path,           # 上传的原始图像路径
            'plate_image_path': plate_path_matlab, # MATLAB裁剪后的车牌图像路径
            'status': 'success'
        }
    except Exception as e:
        logger.exception('MATLAB processing failed')
        raise HTTPException(status_code=500, detail=f'MATLAB error: {str(e)}')


if __name__ == '__main__':
    import uvicorn

    # Optionally preload MATLAB if env var set to true
    preload = os.environ.get('PRELOAD_MATLAB', 'false').lower() in ('1', 'true', 'yes')
    if preload and getattr(matlab_client, '_HAS_MATLAB', False):
        try:
            matlab_client.client.start()
        except Exception:
            logger.exception('Preload matlab failed')

    uvicorn.run('backend.main:app', host='0.0.0.0', port=8000, reload=True)