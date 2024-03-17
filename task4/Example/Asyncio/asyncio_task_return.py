import asyncio
import time


async def say_after(delay, what):
    await asyncio.sleep(delay)
    return what


async def main():
    task1 = asyncio.create_task(
        say_after(3, 'hello'))

    task2 = asyncio.create_task(
        say_after(3, 'world'))

    print(f"started at {time.strftime('%X')}")

    # Wait until both tasks are completed (should take
    # around 3 seconds.)
    await task1
    await task2

    print(task1.result())
    print(task2.result())

    print(f"finished at {time.strftime('%X')}")


if __name__ == "__main__": 
    asyncio.run(main())
    