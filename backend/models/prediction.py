from enum import Enum

from pydantic import BaseModel
from typing import Optional


class Label(str, Enum):
    """
    Custom enum for labeling the data
    """
    REAL = 0
    FAKE = 1

    def __str__(self):
        return self.name


class Prediction(BaseModel):
    """
    Prediction ml_model is a class mostly used for transferring and storing data
    @param
        title(str) - Title of the News
        text(str) - Text of the News
        label(Label-enum) - Label of the News
    """
    title: str
    text: str
    label: Optional[Label] = None
