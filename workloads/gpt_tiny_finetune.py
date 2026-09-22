"""Convenience entrypoint: gpt-tiny finetune (vertical-slice smoke)."""
from __future__ import annotations

import sys

from workloads.llm_workloads import main as llm_main


def main() -> None:
    # Default argv for vertical slice when invoked without flags.
    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                "--mode",
                "finetune",
                "--preset",
                "tiny",
                "--probe",
            ]
        )
    else:
        # Normalize --duration -> --max-seconds for orchestrator compatibility.
        argv = []
        i = 1
        while i < len(sys.argv):
            if sys.argv[i] == "--duration" and i + 1 < len(sys.argv):
                argv.extend(["--max-seconds", sys.argv[i + 1]])
                i += 2
                continue
            if sys.argv[i] == "--mode" and i + 1 < len(sys.argv) and sys.argv[i + 1] == "finetune":
                argv.extend(["--mode", "finetune", "--preset", "tiny"])
                i += 2
                continue
            argv.append(sys.argv[i])
            i += 1
        if "--preset" not in argv:
            argv.extend(["--preset", "tiny"])
        if "--mode" not in argv:
            argv.extend(["--mode", "finetune"])
        sys.argv = [sys.argv[0], *argv]
    llm_main()


if __name__ == "__main__":
    main()
