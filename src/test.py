
import asyncio

from ai_modules.gpt import AIGPT

async def test_gpt():
    agent = AIGPT()

    response = await agent.generate_reply([{
        "role": "user",
        "content": "what is the sum of all deposits of all the branches in Ohio?"
    }])
    print(f"response: {response}")

async def main():
    await test_gpt()

if __name__ == "__main__":
    asyncio.run(main())