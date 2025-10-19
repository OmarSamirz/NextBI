from utils import get_multi_agent
from modules.event_loop_thread import EventLoopThread

async def test():
    multi_agent = await get_multi_agent()
    await multi_agent.run("What are the tables we have?")

async def main():
    await test()


if __name__ == "__main__":
    loop = EventLoopThread()
    loop.run_coroutine(main())