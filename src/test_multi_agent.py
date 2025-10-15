
from utils import get_multi_agent
from modules.event_loop_thread import EventLoopThread

async def test():
    multi_agent = await get_multi_agent()

    query_1 = "How are you!"
    response = await multi_agent.run(query_1)
    print(response)

    query_2 = "What are the tables do we have?"
    response = await multi_agent.run(query_2)
    print(response)


async def main():
    await test()


if __name__ == "__main__":
    loop = EventLoopThread()
    loop.run_coroutine(main())