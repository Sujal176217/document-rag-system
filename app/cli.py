"""Command line helper.

  python -m app.cli ingest [path]     index a file or folder (default: data/docs)
  python -m app.cli ask "question"    ask a question
  python -m app.cli providers         show active providers / which keys are set
"""
import argparse
from pathlib import Path

from .config import settings
from .providers import ConfigError, describe_providers
from .service import build_service


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("path", nargs="?", default=str(settings.docs_dir))
    p_ask = sub.add_parser("ask")
    p_ask.add_argument("question")
    sub.add_parser("providers", help="show active providers and which keys are set")
    args = parser.parse_args()

    if args.cmd == "providers":
        info = describe_providers()
        print("Active:", info["active"])
        for p in info["providers"]:
            print(f"  {p['name']:<10} key var: {p['key_env']:<18} keys set: {p['keys_configured']}  "
                  f"embeddings: {p['embeddings']}  llm: {p['llm']}")
        return

    try:
        service = build_service()
    except ConfigError as e:
        raise SystemExit(f"Setup problem: {e}")
    if args.cmd == "ingest":
        path = Path(args.path)
        n = service.ingest_directory(path) if path.is_dir() else service.ingest_file(path)
        print(f"Indexed {n} chunks from {path}")
    else:
        result = service.ask(args.question)
        print(f"\nAnswer: {result.answer}\n")
        for i, s in enumerate(result.sources, 1):
            print(f"  [{i}] {s.source}")


if __name__ == "__main__":
    main()
