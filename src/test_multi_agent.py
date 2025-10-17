import asyncio

from utils import get_multi_agent, get_ai


async def test():
    # multi_agent = await get_multi_agent()
    td_agent = await get_ai()

    answer = await td_agent.generate_reply([{"role": "user", "content": "How many tables do we have?"}])
    print(answer)
    # query_1 = "How are you!"
    # response1 = await multi_agent.run(query_1)
    # print(response1["response"])

    # query_2 = "What is the database name you should use?"
    # response2 = await multi_agent.run(query_2)
    # print(response2["response"])

    # query_3 = "Can you plot it in a pie chart"
    # response3 = await multi_agent.run(query_3)
    # print(response3["response"])

async def main():
    await test()


if __name__ == "__main__":
    asyncio.run(main())