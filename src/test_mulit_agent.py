
from utils import get_multi_agent
from modules.event_loop_thread import EventLoopThread

async def test():
    multi_agent = await get_multi_agent()

    query1 = "What are the tables we have?"
    response1 = await multi_agent.run(query1)
    print(response1["response"])

    query2 = "Can you plot them in a pie chart equally"
    response2 = await multi_agent.run(query2)
    print(response2["response"])

async def main():
    await test()


if __name__ == "__main__":
    loop = EventLoopThread()
    loop.run_coroutine(main())