import asyncio
import time


async def io_operation():
    await asyncio.sleep(1)  # Отдаем управление обратно в Event loop пока ждём


async def starter():
    print('Выполняем IO операции')
    await asyncio.gather(*[io_operation() for i in range(10000)])


if __name__ == "__main__":
    start = time.monotonic() # точка отсчета времени
    asyncio.run(starter())
    duration = time.monotonic() - start
    print(f'Duration: {round(duration, 4)} seconds')
