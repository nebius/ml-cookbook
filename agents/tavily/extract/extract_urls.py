import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common import tavily_post, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract content from URLs with Tavily Extract.")
    parser.add_argument("--urls-file", required=True, help="Path to a newline-delimited list of URLs")
    parser.add_argument("--output", default="outputs/extract/results.json", help="Output JSON path")
    parser.add_argument("--query", default=None, help="Optional reranking intent")
    parser.add_argument("--extract-depth", default="advanced", choices=["basic", "advanced"])
    parser.add_argument("--format", default="markdown", choices=["markdown", "text"])
    parser.add_argument("--chunks-per-source", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    urls = [line.strip() for line in Path(args.urls_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = {
        "urls": urls,
        "extract_depth": args.extract_depth,
        "format": args.format,
        "include_favicon": True,
        "include_usage": True,
    }
    if args.query:
        payload["query"] = args.query
        payload["chunks_per_source"] = args.chunks_per_source

    response = tavily_post("extract", payload)
    write_json(args.output, response)
    print(f"Wrote extraction results for {len(urls)} URLs to {args.output}")


if __name__ == "__main__":
    main()
