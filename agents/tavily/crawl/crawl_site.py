import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common import tavily_post, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl a site with Tavily Crawl.")
    parser.add_argument("--url", required=True, help="Root URL to crawl")
    parser.add_argument("--output", default="outputs/crawl/corpus.json", help="Output JSON path")
    parser.add_argument(
        "--instructions", default="Find the most useful pages for an engineering onboarding corpus."
    )
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-breadth", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    response = tavily_post(
        "crawl",
        {
            "url": args.url,
            "instructions": args.instructions,
            "limit": args.limit,
            "max_depth": args.max_depth,
            "max_breadth": args.max_breadth,
            "include_images": False,
            "include_favicon": True,
            "include_usage": True,
        },
    )
    write_json(args.output, response)
    print(f"Wrote crawl results to {args.output}")


if __name__ == "__main__":
    main()
