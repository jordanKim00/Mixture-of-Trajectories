from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Docker, GPU, and HF cache wiring.")
    parser.add_argument(
        "--model_name_or_path",
        default="deepseek-ai/deepseek-moe-16b-chat",
        help="HF model id expected in the local cache.",
    )
    args = parser.parse_args()

    import torch
    import transformers

    print(f"[torch] {torch.__version__} cuda={torch.cuda.is_available()} n_gpu={torch.cuda.device_count()}")
    for idx in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(idx)
        print(f"  GPU{idx}: {prop.name} {prop.total_memory / 1024**3:.1f} GiB")
    print(f"[transformers] {transformers.__version__}")

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    print(f"[HF_HOME] {hf_home}")
    cache_key = "models--" + args.model_name_or_path.replace("/", "--")
    snapshot_glob = str(hf_home / "hub" / cache_key / "snapshots" / "*")
    snapshots = sorted(glob.glob(snapshot_glob))
    if snapshots:
        print("[cache] found model snapshots:")
        for snapshot in snapshots:
            print(f"  {snapshot}")
    else:
        raise SystemExit(f"[cache] no local snapshots found for {args.model_name_or_path}")

    remote_module_glob = str(
        hf_home
        / "modules"
        / "transformers_modules"
        / args.model_name_or_path.split("/")[0]
        / args.model_name_or_path.split("/")[1]
        / "*"
        / "modeling_deepseek.py"
    )
    remote_modules = sorted(glob.glob(remote_module_glob))
    if remote_modules:
        print("[remote_code] found cached modeling_deepseek.py")
    else:
        print("[remote_code] cached remote code not found yet; model load may populate it")


if __name__ == "__main__":
    main()

