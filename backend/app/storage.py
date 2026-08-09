"""Video file storage - local disk by default, real S3-compatible object
storage (AWS S3, Cloudflare R2, etc.) when configured via env vars.

BLOCKER, flagged explicitly: provisioning an actual bucket and getting
credentials requires signing up for a third-party service (AWS or
Cloudflare) and generating API keys in their dashboard - genuinely not
something scriptable from this environment (no account, no CLI session,
no API key to create one with). This module is written so that flipping
storage over needs zero code changes once that's done - just set these
env vars on the host:
    S3_BUCKET               - bucket name
    AWS_ACCESS_KEY_ID       - access key
    AWS_SECRET_ACCESS_KEY   - secret key
    S3_ENDPOINT_URL         - optional; set this for Cloudflare R2 (an
                               S3-compatible API) or any non-AWS provider,
                               omit it for real AWS S3
    AWS_REGION              - optional, defaults to "auto" (R2's convention)
Until those are set, everything transparently uses local disk exactly as
before - no behavior change for local dev or an unconfigured deploy.

Analyzed-session videos still always land on local disk first regardless
of configuration: the pose-analysis pipeline (pipeline/pose_estimation.py)
reads frames via OpenCV, which needs a real local file to seek through -
there's no way around a local working copy for that step. What object
storage below adds is a *persistence* layer on top: once a video is
uploaded, `persist()` also pushes a durable copy to the bucket (when
configured) so it survives a redeploy wiping local disk (see
render.yaml's docstring on the free tier's ephemeral disk), independent
of whether the pipeline has finished reading it locally.
"""
import os
from pathlib import Path


def is_configured() -> bool:
    return bool(os.environ.get("S3_BUCKET") and os.environ.get("AWS_ACCESS_KEY_ID"))


def _client():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_REGION", "auto"),
    )


def persist(local_path: Path, key: str) -> None:
    """Best-effort: uploads a durable copy of an already-locally-saved file
    to object storage, if configured. Never raises - a failed persist just
    means the video only lives on (ephemeral) local disk, same as today;
    it shouldn't turn a successful upload/analysis into a user-facing error.
    """
    if not is_configured():
        return
    try:
        _client().upload_file(str(local_path), os.environ["S3_BUCKET"], key)
    except Exception:  # noqa: BLE001 - see docstring: never let this break the request
        pass


def signed_url(key: str, expires_in: int = 3600) -> str:
    """A temporary, directly-fetchable URL for a stored object - used to
    redirect a video-serving route to object storage instead of streaming
    from local disk, when configured.
    """
    return _client().generate_presigned_url(
        "get_object",
        Params={"Bucket": os.environ["S3_BUCKET"], "Key": key},
        ExpiresIn=expires_in,
    )
