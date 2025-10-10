from langchain.agents import tool

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
def multiply(first: float, second: float) -> float:
    """Multiply two float numbers together."""
    return first * second

@tool
def add(first: float, second: float) -> float:
    "Add two float numbers."
    return first + second

@tool
def subtract(first: float, second: float) -> float:
    "Subtract two float numbers."
    return first - second

@tool
def divide(first: float, second: float) -> float:
    "Divide two float numbers."
    if second == 0:
        return 0

    return first / second 