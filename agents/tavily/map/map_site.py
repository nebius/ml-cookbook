import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common import tavily_post, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map a site with Tavily Map.")
    parser.add_argument("--url", required=True, help="Root URL to map")
    parser.add_argument("--output", default="outputs/map/site_map.json", help="Output JSON path")
    parser.add_argument("--instructions", default=None, help="Optional mapping instructions")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-breadth", type=int, default=20)
    parser.add_argument("--limit", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {
        "url": args.url,
        "max_depth": args.max_depth,
        "max_breadth": args.max_breadth,
        "limit": args.limit,
        "include_usage": True,
    }
    if args.instructions:
        payload["instructions"] = args.instructions

    response = tavily_post("map", payload)
    write_json(args.output, response)
    print(f"Wrote mapped site results to {args.output}")


if __name__ == "__main__":
    main()
