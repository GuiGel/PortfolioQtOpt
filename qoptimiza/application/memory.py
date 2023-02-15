from typing import Optional
from loguru import logger

class TokenApi:
    _value: Optional[str] = None

    def __init__(self, value: Optional[str] = None) -> None:
        self._value = value

    @property
    def value(self) -> Optional[str]:
        logger.debug(f"get token api with val {self._value}")
        return self._value

    @value.setter
    def value(self, value) -> None:
        logger.debug(f"set token api to {value}")
        self._value = value

    @value.deleter
    def value(self) -> None:
        logger.debug(f"delete token api {self._value} -> None")
        self._value = None

    def __str__(self) -> str:
        return f"TokenApi(value={self.value})"

_token_api = TokenApi()

if __name__ == "__main__":
    val = "abcd"
    _token_api.value = val
    print(_token_api.value)