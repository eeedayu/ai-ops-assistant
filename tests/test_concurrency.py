import asyncio
import time


async def fake_llm(
    task_id: int,
):

    await asyncio.sleep(1)

    return f"task-{task_id}"


async def main():

    start = time.perf_counter()

    results = await asyncio.gather(
        fake_llm(1),
        fake_llm(2),
        fake_llm(3),
        fake_llm(4),
        fake_llm(5),
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(results)

    print(
        f"elapsed={elapsed:.2f}s"
    )


if __name__ == "__main__":
    asyncio.run(main())