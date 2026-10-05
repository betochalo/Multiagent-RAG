"""Multi-agent solver with GraphRAG (Taller 03 v2).

    uv run python -m investigation_agent <statement.pdf> [output_dir]
"""

import json
import sys
from pathlib import Path


def main() -> None:
    from investigation_agent.graph.orchestrator import Solver

    if len(sys.argv) < 2:
        sys.exit("usage: python -m investigation_agent <statement.pdf> [output_dir]")
    pdf = Path(sys.argv[1])
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("corridas") / pdf.stem
    result = Solver().solve(str(pdf), str(output))
    print(json.dumps(result, ensure_ascii=False, indent=2))
