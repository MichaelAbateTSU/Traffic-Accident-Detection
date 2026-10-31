# Isolated live-camera experiment

This branch-only experiment is not intended for merging or deploying. Manually
dispatch the existing Native ARM CPU runtime workflow on
`michaelabatetsu-redesigned-spork` to run the amd64 live smoke. The live job cannot
run on main, another branch, a push, or a pull request. Existing push/PR runtime
checks remain unchanged outside this experimental branch.

The image, locked dependencies, API, source validation, YOLO11n, DeepSORT and
scorer must match production revision
`0408066230734c8bf88eac0aeeed313d7c01e950`. Only the fixed existing R2_061 HTTPS
source is approved; there are no URL inputs. The authenticated API runs on
container loopback inside an internal-only Docker network; the production
exact-host/public-IP-pinned proxy is the only internet route. No ports are
published. The operator token is generated in memory and passed through the
process environment, never printed or written to a file.

One job requests 12 frames, `save_frames=false`, a 120-second production execution
limit, two CPUs and 4 GiB without swap. The API root filesystem and artifact mount
are read-only. Only bounded RAM-backed job metadata is writable. Image/video
writers are fail-closed and Python writes outside the numerical metrics/SQLite
allowlist are denied. Measurement wrappers call the unchanged model, tracker and
scorer and count their returned results; no tensors, embeddings, coordinates or
images are reported. Container logging and core dumps are disabled. No artifact
upload or media download-to-file step exists. Live HLS bytes and decoded images
exist only in decoder/process memory.

The probe checks authentication, rejected unapproved sources, direct egress
isolation, proxy rejection of an unapproved host, terminal job status, published
frame count, the empty artifact mount and metadata-only writable data directory.
Its safe JSON summary is retained only in Actions logs. Container stdout/stderr
and raw API errors are suppressed. The controller stops/removes both services
and networks in `finally`, with an additional always-run workflow cleanup.
Removing the API destroys RAM-only job data.

A failed stream-open/decode or inference is reported as failure, not inferred
success from playlist HTTP headers. Successful processing is only a transient
runner smoke, not a deployed backend, an accuracy evaluation, or evidence about
other cameras. The small sample and unchanged heuristic score do not establish
accident detection quality. Read-only mounts, write guards, empty artifacts and
disabled logs provide bounded no-retention evidence, not an audit of all
third-party native code or the runner provider's memory handling.
