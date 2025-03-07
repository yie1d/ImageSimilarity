import os
import re
import time
import datetime
import logging
import typing as t
from stat import ST_MTIME
from pathlib import Path
from logging.handlers import BaseRotatingHandler


class CustomTimedRotatingFileHandler(BaseRotatingHandler):
    """自定义日志记录器，每天凌晨清空旧日志"""
    def __init__(self, custom_log_name, filepath, backup_count=0, encoding=None, delay=False):
        BaseRotatingHandler.__init__(self, filepath, 'a', encoding, delay)
        self.backup_count = backup_count

        # 每天更新
        self.interval = 60 * 60 * 24
        self.suffix = "%Y-%m-%d"
        self.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}(\.\w+)?$", re.ASCII)

        filename = self.baseFilename
        if os.path.exists(filename):
            _t = os.stat(filename)[ST_MTIME]
        else:
            _t = int(time.time())
        self.rolloverAt = self.computeRollover(_t)
        self.custom_log_name = custom_log_name

        # 每次执行时都需要扫一次 清理历史文件
        for s in self.getFilesToDelete():
            os.remove(s)

    def computeRollover(self, current_time: int):
        """
        计算每天文件需要变更的时间
        从每天00:00:00变更
        """
        _t = time.localtime(current_time)
        current_hour, current_minute, current_second = _t[3:6]

        r = self.interval - ((current_hour * 60 + current_minute) * 60 + current_second)
        if r < 0:
            r += self.interval
        return current_time + r

    def shouldRollover(self, record):
        """
        判断是否需要变更记录的文件
        """
        _t = int(time.time())
        if _t >= 1:
            return 1
        return 0

    def doRollover(self):
        """执行变更"""
        if self.stream:
            self.stream.close()
            self.stream = None

        currentTime = int(time.time())
        dstNow = time.localtime(currentTime)[-1]

        if self.backup_count > 0:
            for s in self.getFilesToDelete():
                os.remove(s)
        self.baseFilename = f'{Path(self.baseFilename).parent}/{(self.custom_log_name + "_") if self.custom_log_name else ""}{datetime.datetime.now().strftime("%Y-%m-%d")}.log'
        if not self.delay:
            self.stream = self._open()

        newRolloverAt = self.computeRollover(currentTime)
        while newRolloverAt <= currentTime:
            newRolloverAt = newRolloverAt + self.interval

        dstAtRollover = time.localtime(newRolloverAt)[-1]
        if dstNow != dstAtRollover:
            if not dstNow:
                addend = -3600
            else:
                addend = 3600
            newRolloverAt += addend

        self.rolloverAt = newRolloverAt

    def getFilesToDelete(self):
        """
        获取需要删除的文件路径
        """
        file_path = Path(self.baseFilename)

        file_path_list = []
        for log_file_path in file_path.parent.glob('*.log'):
            if log_file_path.is_file() and self.extMatch.match(log_file_path.name) and log_file_path.name.startswith(self.custom_log_name):
                file_path_list.append(str(log_file_path))
        if len(file_path_list) < self.backup_count:
            file_path_list = []
        else:
            file_path_list.sort()
            file_path_list = file_path_list[:len(file_path_list) - self.backup_count]

        return file_path_list


def get_logger(log_name: str = '', log_dir_path: t.Union[None, str, Path] = None) -> logging.Logger:
    """获取logger，如果log_name同名，则 ***以第一次设置的log_dir_path为准，后续设置的log_dir_path无效！！！*** """
    my_logger = logging.getLogger(log_name)

    if not my_logger.handlers:
        my_logger.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] - [%(message)s]')
        log_dir_path = Path(log_dir_path) if log_dir_path else Path(Path(__file__).resolve().parent.joinpath('log'))
        log_dir_path.mkdir(parents=True, exist_ok=True)
        log_filename = log_dir_path.joinpath(f'{(log_name + "_") if log_name else ""}{datetime.datetime.now().strftime("%Y-%m-%d")}.log')
        file_handler = CustomTimedRotatingFileHandler(log_name, str(log_filename), backup_count=4, encoding='utf8')
        file_handler.setFormatter(formatter)
        my_logger.addHandler(file_handler)

    return my_logger
