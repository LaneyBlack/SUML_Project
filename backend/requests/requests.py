from enum import Enum

from pydantic import BaseModel
from typing import Optional


class Label(str, Enum):
    REAL = 0
    FAKE = 1

    def __str__(self):
        return self.name


class Prediction(BaseModel):
    title: str
    text: str
    label: Optional[Label]
