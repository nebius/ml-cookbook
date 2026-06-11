import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common import poll_research, tavily_post, write_json, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a research report with Tavily Research.")
    parser.add_argument("--input", required=True, help="Research prompt or question")
    parser.add_argument(
        "--output-dir", default="outputs/research", help="Directory for report artifacts"
    )
    parser.add_argument("--model", default="mini", choices=["mini", "pro", "auto"])
    parser.add_argument("--poll-interval", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queued = tavily_post(
        "research",
        {
            "input": args.input,
            "model": args.model,
        },
    )
    request_id = queued["request_id"]
    completed = poll_research(
        request_id=request_id,
        poll_interval=args.poll_interval,
        timeout_seconds=args.timeout_seconds,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(str(output_dir / "response.json"), completed)
    write_json(str(output_dir / "sources.json"), {"sources": completed.get("sources", [])})
    content = completed.get("content", "")
    if isinstance(content, str):
        write_text(str(output_dir / "report.md"), content)
    else:
        write_json(str(output_dir / "report_structured.json"), content)
    print(f"Research task {request_id} completed. Wrote artifacts to {output_dir}")


if __name__ == "__main__":
    main()
