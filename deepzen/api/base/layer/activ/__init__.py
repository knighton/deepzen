from .map import BaseMapAPI
from .relative import BaseRelativeAPI


class BaseActivAPI(BaseMapAPI, BaseRelativeAPI):
    pass
