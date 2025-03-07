import typing as t


class BasicException(Exception):
    """
    基本的异常类
    """
    description: t.Optional[str] = '基本异常错误'

    def __init__(self, description: t.Optional[str] = None, error_info: t.Any = None) -> None:
        super().__init__()
        if description is not None:
            self.description = description
        self.error_info = error_info

    def __repr__(self):
        return self.description

    def __str__(self):
        return self.description


class RequestParamsException(BasicException):
    pass


class ModelNotFoundException(BasicException):
    pass
