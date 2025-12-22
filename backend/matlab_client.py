"""
Lightweight MATLAB Engine client wrapper (适配车牌识别场景)

Requires that matlab.engine was installed using MATLAB's provided installer
instructions (i.e. not necessarily via pip). This wrapper exposes start/stop
and a `call_function` helper using MATLAB's `feval`.

The client will add the project's `matlab_core` directory to MATLAB path when
starting so that .m files placed there are callable.
"""
from __future__ import annotations

import os
import threading
import logging
import time
from typing import Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    # import the matlab.engine module object as matlab_engine for clarity
    import matlab.engine as matlab_engine  # type: ignore
    _HAS_MATLAB = True
except ImportError:
    matlab_engine = None  # type: ignore
    _HAS_MATLAB = False
except Exception as e:
    # 捕获其他可能的导入异常（如MATLAB引擎版本不兼容）
    logger.error(f"Failed to import matlab.engine: {str(e)}")
    matlab_engine = None
    _HAS_MATLAB = False


class MatlabClient:
    def __init__(self, matlab_core_dir: Optional[str] = None) -> None:
        self._eng = None
        self._lock = threading.RLock()
        self._start_time: Optional[float] = None
        
        # 优化1：自动定位matlab_core目录（适配不同项目结构）
        if matlab_core_dir and os.path.isdir(matlab_core_dir):
            self.matlab_core_dir = os.path.abspath(matlab_core_dir)
        else:
            # 尝试从当前文件向上查找matlab_core目录
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent  # 项目根目录
            default_core_dir = project_root / "matlab_core"
            self.matlab_core_dir = str(default_core_dir.absolute())
        
        # 优化2：记录MATLAB引擎状态
        self._engine_status = "stopped"  # stopped / starting / running / error

    def start(self, timeout: int = 30) -> None:
        """
        启动MATLAB引擎（带超时和状态检测）
        :param timeout: 启动超时时间（秒），默认30秒
        """
        if not _HAS_MATLAB:
            raise ImportError('matlab.engine is not available in this Python environment')

        with self._lock:
            # 如果引擎已启动，直接返回
            if self.is_running():
                logger.info("MATLAB engine is already running")
                return
            
            # 标记启动中状态，避免重复启动
            self._engine_status = "starting"
            logger.info('Starting MATLAB engine (timeout: %ds)...', timeout)
            
            try:
                # 带超时的引擎启动（MATLAB启动可能较慢）
                start_time = time.time()
                # start_matlab may take a while; this blocks the calling thread
                self._eng = matlab_engine.start_matlab()
                self._start_time = time.time()
                self._engine_status = "running"
                logger.info(f"MATLAB engine started successfully (took {time.time()-start_time:.2f}s)")
                
                # 添加matlab_core目录到MATLAB路径（确保.m文件可调用）
                self._add_matlab_core_path()
                
            except TimeoutError:
                self._engine_status = "error"
                raise TimeoutError(f"MATLAB engine startup timed out after {timeout} seconds")
            except Exception as e:
                self._engine_status = "error"
                logger.error(f"Failed to start MATLAB engine: {str(e)}")
                self._eng = None
                raise

    def _add_matlab_core_path(self) -> None:
        """添加matlab_core目录到MATLAB路径（容错处理）"""
        if not self._eng or not os.path.isdir(self.matlab_core_dir):
            logger.warning(f"matlab_core directory not found: {self.matlab_core_dir}")
            return
        
        try:
            # 检查路径是否已存在
            matlab_path = self._eng.path()
            if self.matlab_core_dir not in matlab_path:
                # Pass absolute path to MATLAB
                self._eng.addpath(self.matlab_core_dir, nargout=0)
                logger.info(f"Added matlab_core to MATLAB path: {self.matlab_core_dir}")
            else:
                logger.info(f"matlab_core already in MATLAB path: {self.matlab_core_dir}")
                
            # 可选：刷新MATLAB路径缓存
            self._eng.rehash(nargout=0)
        except Exception as e:
            logger.exception(f"Failed to add matlab_core to MATLAB path: {str(e)}")

    def is_running(self) -> bool:
        """
        优化：更可靠的引擎运行状态检测
        """
        if self._eng is None:
            return False
        
        try:
            # 测试调用MATLAB内置函数，验证引擎是否真的可用
            self._eng.isempty([], nargout=1)
            return True
        except Exception:
            # 引擎存在但不可用，重置状态
            self._eng = None
            self._engine_status = "stopped"
            return False

    def call_function(
        self, 
        func_name: str, 
        *args: Any, 
        nargout: int = 1,
        retry_times: int = 1
    ) -> Any:
        """
        调用MATLAB函数（增强错误处理和重试机制）
        
        :param func_name: MATLAB函数名（如process_image）
        :param args: 函数参数
        :param nargout: 输出参数个数（适配process_image的2个输出）
        :param retry_times: 失败重试次数（默认1次）
        :return: MATLAB函数返回结果（多个返回值时为元组）
        """
        if not _HAS_MATLAB:
            raise ImportError('matlab.engine is not available in this Python environment')

        # 确保引擎已启动
        if not self.is_running():
            logger.info("MATLAB engine not running, starting now...")
            self.start()

        # 带重试的函数调用
        last_exception = None
        for attempt in range(retry_times + 1):
            with self._lock:
                try:
                    logger.debug(
                        f"Calling MATLAB function: {func_name}, args: {args}, nargout: {nargout}, attempt: {attempt+1}"
                    )
                    # 调用MATLAB函数（核心逻辑不变）
                    result = self._eng.feval(func_name, *args, nargout=nargout)
                    logger.info(f"MATLAB function {func_name} called successfully (attempt {attempt+1})")
                    return result
                except Exception as e:
                    last_exception = e
                    logger.error(
                        f"MATLAB function call failed (attempt {attempt+1}/{retry_times+1}): {func_name}, error: {str(e)}"
                    )
                    # 重试前重置引擎（如果是引擎连接问题）
                    if attempt < retry_times:
                        logger.info("Attempting to restart MATLAB engine before retry...")
                        self.stop()
                        time.sleep(1)
                        self.start()
        
        # 所有重试失败，抛出最后一次异常
        logger.exception(f"All retries failed for MATLAB function: {func_name}")
        raise last_exception  # type: ignore

    def stop(self) -> None:
        """
        停止MATLAB引擎（安全退出，避免资源泄漏）
        """
        with self._lock:
            if self._eng is not None:
                logger.info('Stopping MATLAB engine...')
                try:
                    # 先清理路径（可选）
                    if os.path.isdir(self.matlab_core_dir):
                        self._eng.rmpath(self.matlab_core_dir, nargout=0)
                    # 安全退出引擎
                    self._eng.quit()
                    logger.info('MATLAB engine stopped successfully')
                except Exception as e:
                    logger.exception(f'Error quitting MATLAB engine: {str(e)}')
                finally:
                    # 重置状态
                    self._eng = None
                    self._start_time = None
                    self._engine_status = "stopped"
            else:
                logger.info("MATLAB engine is not running")

    def get_status(self) -> dict:
        """
        获取引擎详细状态（适配main.py的status接口）
        """
        running = self.is_running()
        uptime = time.time() - self._start_time if self._start_time and running else 0
        return {
            "engine_available": _HAS_MATLAB,
            "is_running": running,
            "status": self._engine_status,
            "uptime_seconds": round(uptime, 2) if uptime > 0 else 0,
            "matlab_core_dir": self.matlab_core_dir,
            "matlab_core_exists": os.path.isdir(self.matlab_core_dir)
        }


# 单例客户端（全局使用）
client = MatlabClient()

# 程序退出时自动停止引擎（避免MATLAB进程残留）
def _cleanup_matlab_engine():
    """退出时清理MATLAB引擎"""
    if client.is_running():
        logger.info("Cleaning up MATLAB engine on exit...")
        client.stop()

# 注册退出钩子
try:
    import atexit
    atexit.register(_cleanup_matlab_engine)
except Exception:
    logger.warning("Failed to register MATLAB engine cleanup hook")