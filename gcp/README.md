# gcp/ — Cloud TPU lifecycle

Scripts to run experiments on a Google Cloud TPU **without keeping one
running 24/7**. The TPU is disposable compute; durable state lives in a GCS
bucket in your own project. All scripts run on your **laptop** and load
config from `.env` via `lib.sh`.

```
laptop  ──gcloud──►  TPU VM (compute, ephemeral)  ──gs://──►  bucket (durable)
  ▲                                                              │
  └──────────────────── pull artifacts ─────────────────────────┘
```

## Projects & storage

| | project | role |
|---|---|---|
| compute | `dis-2026-tpu-zx332` (course-managed; account `xulucy317@gmail.com`) | rents the v6e chip; **no storage here** |
| storage | `myloop-2026` (yours, account `wdk0082@gmail.com`) | bucket `gs://dis-2026-zx332-tpu-store`, prefix `nla-metrics/` (`us-east5`) |

Same region for both → no egress cost. The TPU reaches the cross-project
bucket via the auth wired by `setup_storage.sh`.

## One-time setup

```bash
gcp/setup_storage.sh     # grant the TPU's service account access to your bucket
```

(The bucket already exists; `setup_storage.sh` only does the IAM grant, and
asks first. It also prints an SA-key fallback if the keyless grant doesn't work.)

Also set in `.env`: `GIT_REMOTE` (the repo URL the VM clones) once you've
pushed, and your `CRSID`.

## Per-session loop (adopted long-lived node)

This project ADOPTED the pre-existing node `dis-2026-zx332-tpu` (v6e-1, on-demand, not a
queued resource). Its disk persists across stop/start, so the venv and the ~41 GB HF model
cache survive; only the chip is billed while READY.

```bash
gcp/start.sh                              # STOPPED -> READY (billing starts)
gcp/bootstrap.sh                          # first time / after dependency changes: uv sync + torch_xla + .env
git commit -am wip && git push            # the VM runs committed code (GIT_REF, default main)
gcp/launch_bg.sh experiments/001_x.py …   # detached run; log path printed (~/scratch/logs on the VM)
gcp/ssh.sh 'tail -f ~/scratch/logs/<log>' # follow it
gcp/pull.sh exp_001                       # sync gs://…/nla-metrics/artifacts/exp_001 -> ./artifacts/exp_001
gcp/stop.sh                               # READY -> STOPPED (disk kept)
```

Helpers: `gcp/status.sh` (node state, running jobs, disk, bucket), `gcp/ssh.sh [cmd]`,
`gcp/launch.sh` (streaming variant; dies with the ssh session). `create.sh` / `teardown.sh`
are kept for the disposable queued-resource workflow and refuse to touch the adopted node.

## Conventions

- **Spot by default** (`TPU_SPOT=1`): cheap and preemptible. Make training
  resumable — checkpoint to `$CKPT_DIR` (GCS) every ~15–30 min and restore
  the latest on start. Set `TPU_SPOT=0` for on-demand.
- **Queued resources** are the provisioning unit (`QR_NAME`); deleting the
  queued resource deletes the node.
- **Local-pull mode**: leave `GCS_BUCKET` empty in `.env` to skip the bucket
  entirely — `launch.sh`/`teardown.sh` then `scp` artifacts to your laptop
  before deleting. Good for quick on-demand runs; weaker under preemption.
- **Never commit credentials.** SA keys live under `gcp/keys/` (gitignored).
- `WORKER=0` by default (single-host `v6e-1`); set `WORKER=all` for
  multi-host slices.
- `FORCE=1` skips confirmation prompts (for scripting teardown).

## Notes / TODO

- `launch.sh` passes experiment args positionally; quote complex args.
- Code reaches the VM via `git clone`/`pull` (push before launching). For
  fast local iteration without pushing, `scp` the changed files with
  `gcp/ssh.sh` or add an rsync helper.
- The on-VM checkpoint-to-GCS write + SIGTERM-on-preemption handler is
  framework-specific — add it to `src/<pkg>/` once the framework is chosen.
