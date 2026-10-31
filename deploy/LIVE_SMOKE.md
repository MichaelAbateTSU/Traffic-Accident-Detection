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
are read-only. PyTorch's generated Python import metadata is initialized in a
private 16 MiB RAM directory before any camera access, then its files/directories
are made non-writable. A before/after hash inventory must remain unchanged.
Only bounded RAM-backed job metadata is writable during inference. Image/video
writers are fail-closed and Python writes outside the numerical metrics/SQLite
allowlist are denied. Measurement wrappers call the unchanged model, tracker and
scorer and count their returned results; no tensors, embeddings, coordinates or
images are reported. Container logging and core dumps are disabled. No artifact
upload or media download-to-file step exists. Live HLS bytes and decoded images
exist only in decoder/process memory.

The probe checks authentication, rejected unapproved sources, direct egress
isolation, proxy rejection of an unapproved host, terminal job status, published
frame count, the empty artifact mount and metadata-only writable data directory.
Denied metadata-write attempts are explicitly counted, not permitted; a caught
library metadata-write denial does not fail an otherwise unchanged inventory.
Any image/video-writer attempt fails the smoke.
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

## Verified result, 2026-10-31

[Actions run 41229037035](https://github.com/MichaelAbateTSU/Traffic-Accident-Detection/actions/runs/41229037035)
**passed** on the ephemeral `ubuntu-24.04` amd64 runner. Its exact experiment
revision is `f178f950a63a322c772cfc31558117b4c0359653`; production code matched
current main `0408066230734c8bf88eac0aeeed313d7c01e950`, with the new embedding
batch option left at its unchanged default of 16. Image/proxy builds, the
authenticated live probe and always-run cleanup all passed. The offline ARM
job was deliberately skipped on this experimental branch.

The one fixed R2_061 API job was **complete**, with **12/12 frames processed**.
Actual YOLO11n, DeepSORT tracker, MobileNetV2 re-ID embedder and AccidentScorer
each executed 12 frame calls. Counters recorded **33 vehicle detection
observations**, **33 embedded crops**, **30 confirmed-track observations** and
at most **3 simultaneously confirmed tracks**. These are repeated per-frame
observations, not unique vehicle counts. The peak heuristic score was
**0.2206**, with **0 accident events** and `accident_detected=false`; the score
is not a calibrated probability.

The production image verified YOLO11n weight SHA-256
`0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
There was no source substitution, fixture input or mocked model/tracker in
the Actions experiment. Model/tracker/scorer measurement wrappers only counted
unchanged return values.

No-image evidence: **0 published frames, 0 artifact files, 0 attempted image/video
writes and 0 uploaded Actions artifacts**. The writable data directory contained
only numerical/SQLite/startup-inventory metadata. The single generated startup
Python metadata file's hash inventory was unchanged after inference. One
Ultralytics event/callback temporary-metadata write was **denied** by the same
write guard; it was not permitted or silently omitted from the evidence.
Container logs, core dumps and swap were disabled; no ports or persistent data
volumes existed. Both temporary services were stopped/removed, and RAM-only
job metadata was destroyed. Authentication and approved-source/proxy/direct-egress
checks passed. Only safe JSON numerical summaries and provenance remain in
Actions logs; no footage, segments, screenshots or embeddings were uploaded.

The diagnostic history is retained transparently:

| Actions run | Exact experiment revision | Outcome |
| --- | --- | --- |
| [41228053587](https://github.com/MichaelAbateTSU/Traffic-Accident-Detection/actions/runs/41228053587) | `e7347072fc4dcc172d8402a8db57b601f4f78bc5` | Failed before safe job counters were available; no inference claim. Production base was `ab28ded1e3eaf3824321843afe458958b1a0e676`. |
| [41228314043](https://github.com/MichaelAbateTSU/Traffic-Accident-Detection/actions/runs/41228314043) | `b282b7b099055498bea16fb0bb8872b8e7e1386e` | API job failed, 0 frames and 0 completed model/tracker/scorer calls; four startup metadata writes blocked. |
| [41228538927](https://github.com/MichaelAbateTSU/Traffic-Accident-Detection/actions/runs/41228538927) | `6a79a811b8faaa770eda70f314e8ec5371807e02` | API job failed, 0 frames; identified PyTorch/torchvision import-time tempfile metadata as the blocker. |
| [41228797486](https://github.com/MichaelAbateTSU/Traffic-Accident-Detection/actions/runs/41228797486) | `1ad986f196aa8a4d345d0fdf9597bae4aaf4df2c` | API job completed 12 frames after startup-only metadata initialization/freezing, but harness failed because it incorrectly treated a prevented optional metadata write as retention. Artifacts remained empty and startup metadata unchanged. |
| [41229037035](https://github.com/MichaelAbateTSU/Traffic-Accident-Detection/actions/runs/41229037035) | `f178f950a63a322c772cfc31558117b4c0359653` | Passed: API complete, 12 frames, genuine inference/re-ID/tracking/scoring, unchanged metadata and no imagery retained. Write/authentication/egress controls were not relaxed. |

All diagnostic runs after the first used production main
`0408066230734c8bf88eac0aeeed313d7c01e950`; their cleanup steps completed.
Local execution denial was not retried or bypassed. Local validation used only
syntax/YAML checks and mocked guard tests, never a camera connection.

This proves bounded live decoding/inference on this runner for this camera at
this time, **not** a deployed/public backend, long-running reliability or accuracy.
HLS may buffer additional bytes/decoded frames in memory beyond the 12 frames
actually inferred. No precision/recall or accident ground truth was measured.
