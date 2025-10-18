from langchain.agents import tool
from langchain_community.tools import DuckDuckGoSearchResults

from datetime import datetime

@tool
def current_dollar_pound_exchange_rate(current_date: str) -> str:
    """
    Returns the current USD to EGP exhange rate according to the provided current date.
    Use this when you need to know the current USD to EGP exhange rate.
    """
    query = f"What is the USD to EGP exhange rate according to {current_date}"
    search = DuckDuckGoSearchResults()
    return search.invoke(query)

@tool
def current_datetime() -> str:
    """
    Returns the current date and time.
    Use this when the user asks about the current date or time.
    """
    now = datetime.now()
    return now.strftime("Today is %Y-%m-%d and the current time is %H:%M:%S")