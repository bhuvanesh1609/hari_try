import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def resolve_token(token_env: str) -> str:
    token = os.getenv(token_env)
    if not token:
        raise RuntimeError(
            f"Missing Hugging Face token in environment variable {token_env}. "
            "Set it before running this script."
        )
    return token


def main() -> None:
    parser = argparse.ArgumentParser(description="Push a local model folder to Hugging Face Hub")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, e.g. user/my-model")
    parser.add_argument("--local-path", required=True, help="Local model directory to upload")
    parser.add_argument("--private", action="store_true", help="Create repo as private")
    parser.add_argument("--token-env", default="HF_TOKEN", help="Environment variable that contains HF token")
    parser.add_argument(
        "--commit-message",
        default="Upload model artifacts",
        help="Commit message for Hub upload",
    )
    args = parser.parse_args()

    local_path = Path(args.local_path)
    if not local_path.exists() or not local_path.is_dir():
        raise FileNotFoundError(f"Model folder not found: {local_path}")

    token = resolve_token(args.token_env)
    api = HfApi(token=token)

    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(local_path),
        commit_message=args.commit_message,
    )

    print(f"Uploaded model folder {local_path} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
