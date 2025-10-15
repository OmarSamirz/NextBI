from langchain.agents import tool

from typing import Union
from datetime import datetime

@tool
def current_datetime() -> str:
    """
    Returns the current date and time.
    Use this when the user asks about the current date or time.
    """
    now = datetime.now()
    return now.strftime("Today is %Y-%m-%d and the current time is %H:%M:%S")

@tool
def multiply(first: Union[float, int], second: Union[float, int]) -> Union[float, int]:
    """Multiply two float or integer numbers together."""
    return first * second

@tool
def add(first: Union[float, int], second: Union[float, int]) -> Union[float, int]:
    "Add two float or integer numbers."
    return first + second

@tool
def subtract(first: Union[float, int], second: Union[float, int]) -> Union[float, int]:
    "Subtract two float or integer numbers."
    return first - second

@tool
def divide(first: Union[float, int], second: Union[float, int]) -> Union[float, int]:
    "Divide two float or integer numbers."
    if second == 0:
        return 0

    return first / second 