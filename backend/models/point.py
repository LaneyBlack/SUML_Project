from pydantic import BaseModel
from typing import Optional


class Point(BaseModel):
    x: int
    y: Optional[float]