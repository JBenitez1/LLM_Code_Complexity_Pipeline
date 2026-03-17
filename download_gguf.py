#!/usr/bin/env python3
"""
Download GGUF models from Hugging Face into a local directory.

Examples:
  python3 download_gguf.py --repo-id Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
  python3 download_gguf.py --repo-id bartowski/Qwen2.5-Coder-7B-Instruct-GGUF --filename "*Q4_K_M.gguf"
  python3 download_gguf.py --repo-id repo/name --filename model.gguf --local-dir models
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


def list_gguf_files(repo_id: str, revision: str = "main", token: str | None = None) -> List[str]:
    """Return GGUF filenames available in a repo."""
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="model", revision=revision)
    return sorted(path for path in files if path.lower().endswith(".gguf"))


def load_model(
    repo_id: str,
    filename: str | None = None,
    local_dir: str = "models",
    revision: str = "main",
    token: str | None = None,
) -> List[str]:
    """
    Download GGUF model files from Hugging Face.

    If `filename` is omitted, all `.gguf` files in the repo are downloaded.
    `filename` may be an exact filename or a glob pattern like `*Q4_K_M.gguf`.
    Returns local filesystem paths for downloaded files.
    """
    destination = Path(local_dir)
    destination.mkdir(parents=True, exist_ok=True)

    if filename:
        has_glob = any(ch in filename for ch in "*?[]")
        if has_glob:
            snapshot_path = snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
                allow_patterns=[filename],
                local_dir=str(destination),
                token=token,
                local_dir_use_symlinks=False,
            )
            matches = sorted(str(path) for path in Path(snapshot_path).rglob("*.gguf"))
            if not matches:
                raise FileNotFoundError(
                    f"No GGUF files matched pattern {filename!r} in repo {repo_id!r}."
                )
            return matches

        downloaded = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            filename=filename,
            local_dir=str(destination),
            token=token,
            local_dir_use_symlinks=False,
        )
        if not downloaded.lower().endswith(".gguf"):
            raise ValueError(f"Requested file is not a GGUF file: {filename}")
        return [downloaded]

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        allow_patterns=["*.gguf"],
        local_dir=str(destination),
        token=token,
        local_dir_use_symlinks=False,
    )
    matches = sorted(str(path) for path in Path(snapshot_path).rglob("*.gguf"))
    if not matches:
        raise FileNotFoundError(f"No GGUF files found in repo {repo_id!r}.")
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GGUF model files from Hugging Face.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face model repo, e.g. bartowski/model-name-GGUF")
    parser.add_argument("--filename", help="Exact GGUF filename or glob pattern, e.g. '*Q4_K_M.gguf'")
    parser.add_argument("--local-dir", default="models", help="Directory where files are downloaded")
    parser.add_argument("--revision", default="main", help="Repo revision, branch, or commit")
    parser.add_argument("--token", default=None, help="HF token. If omitted, huggingface_hub uses local auth if available")
    parser.add_argument("--list-files", action="store_true", help="List GGUF files in the repo and exit")
    args = parser.parse_args()

    if args.list_files:
        files = list_gguf_files(repo_id=args.repo_id, revision=args.revision, token=args.token)
        if not files:
            print("No GGUF files found.")
            return
        for file_name in files:
            print(file_name)
        return

    downloaded = load_model(
        repo_id=args.repo_id,
        filename=args.filename,
        local_dir=args.local_dir,
        revision=args.revision,
        token=args.token,
    )
    for path in downloaded:
        print(path)


if __name__ == "__main__":
    main()
