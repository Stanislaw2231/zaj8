from pydantic import BaseModel
from typing import Optional

class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: Optional[float]
    SibSp: int
    Parch: int
    Fare: Optional[float]
    Embarked: Optional[str]