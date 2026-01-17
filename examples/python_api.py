from detllm import check, run


def main() -> None:
    run(
        backend="hf",
        model="distilgpt2",
        prompts=["Hello"],
        tier=1,
        out_dir="artifacts/run1",
    )

    report = check(
        backend="hf",
        model="distilgpt2",
        prompts=["Hello"],
        runs=3,
        batch_size=1,
        out_dir="artifacts/check1",
    )

    print(report.status, report.category)


if __name__ == "__main__":
    main()
