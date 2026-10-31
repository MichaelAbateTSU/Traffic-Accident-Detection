# Free CPU production backend

This deployment runs remotely with Docker Compose: Caddy serves HTTPS, a
single-worker CPU API runs on an **internal-only network**, and a bounded
HTTPS CONNECT proxy is its only route to approved public cameras. Jobs,
incident status changes and images use a persistent named volume.

**Deployment status:** no Oracle account/VM authorization, SSH target or backend
DNS name is configured in this workspace. No cloud resources have been created,
and no public backend URL is live. The local container verification does not
change that. Supply an authorized existing free VM, its SSH access and a DNS name
before deploying. An account may exist elsewhere; absent configuration is not
evidence that the user has no account.

Verified on this workspace's **amd64** Docker host: 198 combined evaluator/hosting tests with the real
locked ML dependencies, certificate-verified HTTPS inference over the isolated
proxy, authenticated JPEG retrieval, private/metadata egress rejection, and
preserved results/images after API restart and recreation, including a
300-frame worker regression with 900 saved JPEGs and durable results, plus a
nine-job HTTPS inference batch whose first/last images survive restart.
The ARM64 production image subsequently **built and passed native runtime
verification** on GitHub's standard `ubuntu-24.04-arm` runner:
[run 41220324104](https://github.com/MichaelAbateTSU/Traffic-Accident-Detection/actions/runs/41220324104),
tested commit `9d9b764342a8d6f5d540f174c4d5d6cba7936da1`.
The native `aarch64` image initialized actual YOLO11n and DeepSORT/scorer,
processed two three-frame synthetic jobs, wrote indexed JPEG triplets, and
observed **two CPU threads in all six model forwards**, including first
predictor setup and a repeated job. Serving ran non-root, network-disabled,
read-only, with two CPUs/4 GB and bounded tmpfs; no footage/model artifacts were
uploaded and no secrets were used. This validates architecture/runtime
compatibility, **not Oracle deployment, remote camera decoding or production
performance**. The earlier local ARM emulation attempt failed because QEMU
was unavailable; that separate failure does not invalidate the native result.
Public ACME issuance, VM reboot and live-cloud availability remain unverified.

Experimental FPS scoring is **disabled by default**: leave
`EXPERIMENTAL_SCORER_SOURCE_FPS` blank/unset in `deploy/.env`. Compose forwards
an explicit value, but enable it only when every admitted source has that
independently trusted constant rate. Nominal capture FPS is not timing proof;
mixed rates, VFR and unreported drops are unsupported. This experiment is
validated with synthetic trajectories, not calibrated accident accuracy or a
new real-model deployment measurement. See the
[timing contract and quantitative regressions](../README.md#experimental-trusted-constant-fps-scorer).

## Render Free feasibility

**The current backend cannot safely run on Render Free.** No Render backend service
was provisioned, and no public deployment is implied by this assessment.
The static frontend is hosted at
`https://traffic-accident-detection-frontend.onrender.com`; this does not mean
an inference backend is live or that the frontend can run remote detection yet.
[Render's compute plans](https://render.com/docs/compute-plans) specify
0.1 CPU / 512 MB for free web services.
[Render's Free documentation](https://render.com/docs/free) describes idle
spin-down after 15 minutes, ephemeral local files, no free persistent disks,
and Free Postgres expiry after 30 days.
These limits were supplied from an official-document review; this workspace
could not independently refresh that page because `render.com` DNS resolution
failed. Recheck the provider's current limits before making any purchase.

On 2026-10-31, the locked **amd64** production image was tested locally with
`--cpus 0.1 --memory 512m --memory-swap 512m` (no swap), retaining YOLO11n,
DeepSORT, two inference threads and the real scorer. The existing offline smoke
attempts two three-frame jobs, using a tiny 64x64 synthetic decoding fixture but
actual inference and indexed JPEG writes. The first run was OOM-killed after
approximately 65 seconds, before completing even the first job. A repeat with
Docker health checks disabled loaded YOLO around 24 seconds, initialized
DeepSORT, then was OOM-killed after **48.2 seconds**: exit code **137** and
Docker `OOMKilled=true`. The last two-second cgroup sample recorded a peak of
425,107,456 bytes at 44.48 seconds; this is a sampled lower bound, **not** the
terminal peak. No inference completion time or successful frame count was
obtained. This is a representative local resource-limit experiment, not a
measurement on Render hardware; real camera decoding/resolution can need more
resources, not less.

### Bounded memory-reduction follow-up

Further local experiments retained the same YOLO11n weights, genuine
MobileNetV2 DeepSORT embedder, scorer and preprocessing, under the same
0.1 CPU / 512 MiB / no-swap limits. These were disposable process-level
experiments, **not enabled production settings**:

| Experiment | Actual result |
|---|---|
| Disable PyTorch oneDNN/MKLDNN workspaces; retain two inference threads | One three-frame 720x1280 synthetic-source job completed at 60.7 seconds including startup; the second job OOM-killed the process at 80.7 seconds. Recorded cgroup peak reached the 536,870,912-byte limit. |
| Also use one intra-op/inter-op/OpenCV/BLAS thread, `MALLOC_ARENA_MAX=1`, and `torch.inference_mode()` around the worker | Two three-frame 720x1280 synthetic-source jobs completed by 64.0 seconds including startup; indexed JPEGs decoded successfully, and genuine re-ID inference on two crops passed. Recorded cgroup high-water was 479,576,064 bytes, leaving little spare RAM. |
| Same reductions, actual decoded ACCIDENT `single (3)` preview, 120 requested frames and saved images on a disposable disk-backed volume | No OOM, but the unchanged 300-second execution deadline explicitly failed the job; process exited with the expected validation failure after 338.7 seconds including startup/shutdown. Recorded cgroup high-water reached 496,181,248 bytes (about 473 MiB). Completion, full artifacts and a successful 120-frame score were not obtained. |
| Same reductions, first 12 decoded preview frames, followed by a separate existing-size 16-crop genuine re-ID batch | Twelve frames completed at 111.3 seconds including startup, with 12 saved image triplets; the last raw/tracked/score-overlay JPEGs decoded at 1280-pixel longest side. Peak heuristic score was 0.1103, not a probability. Recorded high-water before the separate crop stress was 469,708,800 bytes. The 16-crop embedder stress then reached the 512 MiB cap and OOM-killed the process at 119.8 seconds. |

The video was an explicitly opted-in research preview, SHA-256
`ce308189f4f3b6803e67e767a52247e80e1de415381f7e6f3ccb15f357689a41`;
its attribution and **CC BY-NC-SA 4.0 noncommercial restriction** are described
in the labeled-video section below. It was downloaded only for the experiment,
not bundled or made a default source. Local capture was substituted for remote
opening to isolate decoded-frame resource costs; this was **not** an
authenticated Render/egress HTTP deployment test. The separate 16-crop stress
used repeated 224x224 synthetic crops to exercise the embedder's existing batch
capacity, not a claim that this preview contained 16 simultaneous detections.

Allocator/thread/autograd reductions help a narrow workload, but have not
established reliable memory headroom, full-batch completion, numerical
equivalence or CPU throughput for production. No optimization was promoted into
the runtime: silently shrinking re-ID batches, disabling the embedder,
changing models or lifting deadlines would not demonstrate readiness. The
independent durability, egress isolation and idle-lifecycle blockers below
remain even if an individual short inference succeeds.

### Opt-in DeepSORT forward batch limit

`DEEPSORT_EMBEDDER_BATCH_SIZE` accepts integers **1-16**, default **16**.
Set it explicitly to `2` in the deployment environment to reduce genuine
MobileNetV2 forward batches without changing weights, re-identification,
detection count/order, tracking thresholds or the default configuration.
Standalone pipeline users can pass
`DeepSortTracker(embedder_batch_size=2)`; existing no-argument callers retain
the original behavior. Invalid settings fail validation rather than silently
falling back. This is a useful **forward-batch control**, not a Render deployment
configuration or a hard process-memory guarantee.

An isolated comparison with identical experimental one-thread/allocator/
inference-mode/oneDNN reductions and **19 varied crops** measured batch 2
completing two successive embedder calls by 49.7 seconds including startup,
with terminal cgroup high-water **294,174,720 bytes** (about 280.5 MiB).
Batch 16 completed its first call at 44.9 seconds, high-water 464,355,328 bytes,
but was OOM-killed during the second call. This establishes a forward-batching
memory benefit on that fixture, not a bound for the combined YOLO/API process.

The pinned `deep_sort_realtime==1.3.2` constructs its MobileNet embedder with
16, then exposes `max_batch_size` as a mutable inference limit. Its `predict`
method **preprocesses every supplied crop before splitting model forwards**.
Changing the limit bounds forward activation batches, **not the total
preprocessing/crop memory as detection count grows**; all crops and their
ordered embeddings are retained. No detection cap or pruning was introduced.
Real FP32 tests compare 19 varied crops at batch 16 versus 2, in both ordinary
autograd and inference-mode execution: absolute tolerance `1e-5`, relative
tolerance `1e-4`, and cosine discrepancy below `1e-6`. Those tolerances permit
FP32 accumulation-order differences, not arbitrary feature changes. Tests
also preserve all 19 track IDs/classes/boxes over four updates and identical
rendered track overlays. These fixtures do not prove assignment equivalence
for every near-threshold scene or across other architectures.

At local Render-like limits, batch 2 **combined with the earlier experimental
one-thread/allocator/inference-mode/oneDNN reductions** completed 12 decoded
preview frames with 12 saved triplets at 101.2 seconds including startup.
A subsequent genuine 19-crop batch also completed; process exit was successful
after 127.7 seconds. The recorded cgroup high-water was 474,492,928 bytes
(about 452.5 MiB), leaving roughly 59.5 MiB below the 512 MiB cap. This is a
small, hardware-specific margin, not a supported production envelope.
The 120-frame clip still **failed the unchanged 300-second execution deadline**:
51 frames and triplets persisted before failure, peak partial heuristic score
0.2483, recorded high-water 490,635,264 bytes, process exit after 335.4 seconds
including startup/shutdown. No full-clip score is claimed.

Setting batch 2 **alone** does not eliminate the prior default-runtime OOM:
both batch 2 and batch 16 were OOM-killed during MobileNet warmup in isolated
tests retaining original oneDNN/two-thread/autograd behavior. The other
process-level reductions remain experimental and are not enabled by this
setting. Storage, enforced egress isolation, spin-down and provider
authorization blockers are unchanged; no Render backend is provisioned.

Reproduce the basic failure after building the production image:

```sh
docker build -t traffic-detection-cpu:render-limit-check .
docker create --name traffic-render-limit-check --no-healthcheck \
  --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges --cpus 0.1 \
  --memory 512m --memory-swap 512m --pids-limit 128 \
  --tmpfs /tmp:size=64m,mode=1777 --tmpfs /data:size=32m,mode=1777 \
  --mount "type=bind,source=$PWD/deploy/verify_cpu_runtime.py,target=/app/verify_cpu_runtime.py,readonly" \
  traffic-detection-cpu:render-limit-check \
  python verify_cpu_runtime.py --expected-machine x86_64
docker start -a traffic-render-limit-check
docker inspect --format '{{json .State}}' traffic-render-limit-check
docker rm traffic-render-limit-check
```

There are additional independent blockers even if memory were sufficient:

- **Durability:** `/data/jobs.sqlite3` and saved frames require persistent
  storage. Render's ephemeral filesystem loses them on replacement/redeploy,
  preventing accepted-job restart recovery and breaking protected image links.
  A 30-day Free Postgres database is not a durable replacement, and changing
  database engines would not preserve the images. Remote database/object
  storage would need authorization, bounded quotas, expiry/retention and
  validated failure/recovery behavior.
- **SSRF isolation:** this stack relies on Docker Compose's internal-only API
  network and a separate dual-homed proxy as the sole outbound route. A single
  Render web container does not reproduce that topology. Merely setting proxy
  environment variables, or running a proxy beside an Internet-connected API,
  does not enforce the existing egress boundary. A different enforceable
  deployment architecture must be designed and tested, not silently removed.
- **Ingress and lifecycle:** Render supplies HTTPS and expects binding to
  `0.0.0.0:$PORT` (see [web services](https://render.com/docs/web-services)).
  The current command and health check use port 8000 and trust only the Compose
  Caddy address. Port/proxy adaptation is possible but does not solve memory,
  storage or network isolation. Free idle spin-down also precludes an
  always-running service and requires explicit cold-start/job interruption
  handling; keep-alive traffic is not an availability guarantee.

**Render remains the user's preferred provider**; no alternative has been
selected. An **authorized Oracle Always Free VM** is only an unselected free
candidate with potentially sufficient CPU, RAM and persistent storage for the
existing Compose stack, subject to capacity/reclamation limitations below.
No Oracle VM deployment has been verified. Keeping the frontend on GitHub Pages
is independent of this backend choice. Deploying this backend on Render requires
a separately authorized **paid, adequately sized** runtime and
durable disk or database/object storage, plus an enforceable egress design.
The existing two-CPU/4-GB runtime envelope is tested; no smaller paid tier is
certified here. Paying for RAM alone does not fix the other blockers. No paid
plan, external account, model downgrade, lost persistence guarantee or relaxed
authentication/source protection has been selected or provisioned.

## Free-host requirements

An Oracle **Always Free** `VM.Standard.A1.Flex` instance in the account's home
region is a CPU-capable candidate. Use at most **2 OCPUs / 6 GB RAM** for this
stack, subject to the account's **current remaining** free allocation and free
boot/storage quota. Do not use trial credits, a paid shape, paid storage, an
account upgrade or a paid load balancer. This repository does not provision
anything or assume an account exists.

Check [Oracle's current Always Free documentation](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm)
and the console's Always Free labels before creating any resource. Free capacity
can be unavailable and idle instances can be reclaimed. Neither Oracle nor this
configuration guarantees availability; do not manufacture load to evade reclaim
rules. A free static frontend (for example GitHub Pages) also does not make the
inference server free or always available.

Use an existing DNS name or an appropriately authorized free dynamic-DNS name;
buying a domain is not required by the runtime. Point its A record at the VM and
remove an AAAA record unless IPv6 actually reaches this stack.
An optional no-account DNS alternative is an IP-embedded hostname such as
`YOUR-PUBLIC-IP-WITH-HYPHENS.sslip.io`; [sslip.io](https://sslip.io/) documents
Caddy/HTTP-01 use. This is a third-party DNS dependency with no uptime guarantee,
not a provisioned backend. The VM still needs a stable public IP and reachable
ports 80/443; ACME issuance on the eventual VM remains unverified.
Docker Engine with Compose v2 and outbound DNS/HTTPS are required. Allow inbound TCP 80/443
in the VM firewall and cloud security rules; restrict SSH to operator IPs.
Do **not** open 8000 or 3128. Leave Docker's isolation/firewall rules enabled.
Enable Docker's system service on the VM so `restart: unless-stopped` also works
after a host reboot.

## Deploy on the authorized VM

Clone the repository on the VM and check out the reviewed deployment commit.
Run from the repository root:

```sh
cp deploy/.env.example deploy/.env
chmod 600 deploy/.env
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

Privately put the generated token in `API_TOKEN` in `deploy/.env`, then configure:

- `API_DOMAIN`: DNS hostname pointing at this VM, without scheme/path.
- `ACME_EMAIL`: certificate renewal contact.
- `CORS_ORIGINS`: exact frontend origins; defaults to `https://michaelabatetsu.github.io`.
- `APPROVED_STREAM_URLS`: JSON array of exact operator-approved **HTTPS** camera
  manifest/video URLs; no wildcard paths, URL credentials or arbitrary user input.
- `STREAM_HOSTS`: comma-separated exact hostnames for those sources and **every**
  legitimate redirect, variant manifest, segment and key CDN host. HTTPS-only
  sources and segments are intentional; private-camera/RTSP/HTTP streams are
  not supported by this public production deployment.

For the hosted Render frontend, explicitly set
`CORS_ORIGINS=https://traffic-accident-detection-frontend.onrender.com` on the
eventual authorized backend. To retain the GitHub Pages frontend too, set
`CORS_ORIGINS=https://traffic-accident-detection-frontend.onrender.com,https://michaelabatetsu.github.io`.
These are exact origins, without paths or trailing slashes; do not use a
wildcard. This is operator deployment configuration, not a runtime default
change or a claim that a backend exists. Configure the frontend's API URL only
after a genuine HTTPS backend is deployed and verified; keep the bearer token
out of public static assets and acquire it privately at runtime.

The example registry matches the frontend's R2_061 through R2_069 sources at
`https://mcleansfs3.us-east-1.skyvdn.com/rtplive/<camera>/playlist.m3u8`.
Copying an example is **not** permission to use those streams: the operator must
review the source/usage rights and explicitly approve each exact URL and host.
The same hostname was observed for one master/media manifest, but CDN redirects
and segment/key hosts may change. Inspect them before adding any hostname;
never broaden to a wildcard or disable the private-IP/DNS checks. DNS for this
camera CDN failed in this workspace, so these live cameras are not verified
here and source outages must be surfaced as failed jobs, not success.

```sh
docker compose --env-file deploy/.env -f deploy/compose.yaml config --quiet
docker compose --env-file deploy/.env -f deploy/compose.yaml up -d --build
docker compose --env-file deploy/.env -f deploy/compose.yaml ps
curl --fail https://YOUR_API_DOMAIN/health
```

Caddy obtains and renews a public certificate and redirects HTTP to HTTPS.
Only Caddy's fixed internal IP is trusted for forwarded host/scheme headers,
so returned artifact URLs use HTTPS. If the `172.30.0.0/24` subnet conflicts
with an existing network, change **all** private IPs, IPAM subnet, the API's
`egress` host entry and `--forwarded-allow-ips` together.

Models and dependencies are installed **at build time**, not at server start:
Python 3.11, CPU-only PyTorch/torchvision, hash-locked architecture-specific
dependencies and a checksum-verified `yolo11n.pt`. Docker selects the amd64 or
arm64 lock on native builds. The small nano model is an intentional resource
tradeoff, not an accuracy-equivalent replacement for the local `yolo11x.pt`.
Labeled evaluation must validate the chosen model/thresholds before relying on
incident predictions. Replacing the baked model requires rebuilding the image;
never let a public request choose or download weights.

Ultralytics distributes YOLO under [AGPL-3.0 or a separate Enterprise license](https://www.ultralytics.com/license).
Production operators must assess and meet the applicable obligations; a free VM
does not waive licensing. No ACCIDENT research dataset is bundled here or
represented as commercially cleared: that dataset's CC BY-NC-SA 4.0 license
restricts commercial use.

## Frontend authentication contract

Configure the frontend with `https://YOUR_API_DOMAIN`, never `localhost`.
All routes, including image artifacts, need
`Authorization: Bearer <operator token>`, except `GET /health` and CORS
preflight. The frontend must obtain the token from its authorized operator at
runtime; **never** embed it in GitHub Pages assets, build variables or commits.
This is a single shared operator credential, not public sign-up or per-user
roles. Anyone with the token can read data, submit approved jobs and update
incidents. Rotate it by editing the private env file and recreating the API.

CORS permits GET/POST/PATCH, `Authorization`/`Content-Type` headers and exposes
`Content-Disposition` for CSV downloads. Fetch images with the authorization
header and render blob URLs; a plain `<img src>` cannot supply it. No bearer
tokens in query strings. CORS is not an authorization mechanism.

An exact URL allowlist alone does **not** prevent HLS SSRF. Here the API has no
internet gateway, FFmpeg's protocols exclude local files, and every HTTPS CONNECT
target is checked against `STREAM_HOSTS`. FFmpeg verifies upstream TLS certificates.
The proxy rejects private, loopback,
link-local, shared-address, multicast and metadata destinations, validates **all**
DNS answers, and connects to the already-validated numeric IP (no re-resolution).
Redirects and segment URLs must pass this same boundary. Do not attach the API
to an internet network or publish/bypass the proxy. An unapproved URL returns
403; an unreachable/unapproved downstream host fails the job explicitly.

## Opt-in labeled research video

For **noncommercial research only**, an authorized operator can explicitly add
the ACCIDENT authors' individual public previews to the private deployment env.
These are not included in the default production camera registry and are not
bundled in this repository/image. Attribute **Picek et al., ACCIDENT (2026)**,
the [paper](https://arxiv.org/abs/2604.09819),
[author preview repository](https://github.com/accidentbench/accidentbench.github.io)
and [official Kaggle dataset](https://www.kaggle.com/datasets/picekl/accident).
The dataset is **CC BY-NC-SA 4.0**: review current terms, provide attribution,
respect the noncommercial restriction and applicable share-alike obligations.
Do not treat this opt-in as commercial deployment clearance.

For a research-only deployment with these two sources, use these exact values;
to retain the live cameras, append the URLs/hostname to their existing lists:

```ini
APPROVED_STREAM_URLS=["https://media.githubusercontent.com/media/accidentbench/accidentbench.github.io/main/assets/videos/real-world/tbone%20(3).mp4","https://media.githubusercontent.com/media/accidentbench/accidentbench.github.io/main/assets/videos/real-world/single%20(3).mp4"]
STREAM_HOSTS=media.githubusercontent.com
```

Recreate the API **and egress proxy** after editing the private env file, without
deleting data volumes:

```sh
docker compose --env-file deploy/.env -f deploy/compose.yaml up -d
python3 deploy/smoke.py https://YOUR_API_DOMAIN \
  --stream-url 'https://media.githubusercontent.com/media/accidentbench/accidentbench.github.io/main/assets/videos/real-world/single%20(3).mp4' \
  --max-frames 149
```

The API now accepts a strict integer `max_frames` of **1-300**, default **5**.
For `single (3).mp4` / `eQvvIoDMXzY_00.mp4`, request **at least 149 frames**
to include zero-based index 148, the CSV's recorded onset frame. Five or 120
frames do not reach that frame. The CSV is onset-only metadata, not a complete
frame-level accident interval; retain its indexing/time fields without
inventing accident-end labels. For `tbone (3).mp4` / `e17Wq4Hpqlk_02.mp4`,
the CSV says head-on with onset frame 18, despite the preview filename/category.
Process the full ordered sequence; do not feed ground-truth labels or
pre-annotated boxes into YOLO.

`save_frames: true` preserves indexed triplets: raw `capture`, tracked
`capture_annotated`, and score-overlaid `pipeline_output`. The smoke retrieves
all three at the first and last saved indexes using the operator bearer token,
and prints `peak_confidence` and `accident_detected`.
The peak is a **heuristic**, not a calibrated accident probability; YOLO
detects vehicles, and temporal tracking/scoring estimates accident-like motion.
An author-labeled collision can produce no alarm. Overlay annotations are
model outputs, not ground-truth accident labels. The API event timestamp is
processing elapsed time, not media time; compare labels using frame index/FPS
in the separate offline evaluator. Saved images remain subject to retention
and the dataset's restrictions; do not redistribute them as cleared examples.

On 2026-10-31, the actual isolated amd64 Docker stack opened both author-hosted
HTTPS videos through the allowlisted proxy, authenticated each request, ran
the baked `yolo11n.pt` plus DeepSORT/scorer with 1280-side `INTER_LINEAR`
preprocessing and two CPU threads, and retrieved all three first/last JPEG
variants. Each job saved **149 indexed triplets (447 JPEGs), indexes 0-148**.
Decoded `pipeline_output/000148.jpg` differed from its tracking-only counterpart,
confirming the score-overlay output was produced.

| Opt-in author preview | Frames | Peak heuristic score | Alarm | Observed elapsed |
|---|---:|---:|---|---:|
| `single (3).mp4` | 149 | 0.2500 | No | 38.7 s |
| `tbone (3).mp4` | 149 | 0.6990 | Yes; first index 58 | 48.6 s |

Elapsed time is job creation to completion on this workspace host, including
decode/tracking/image writes; it is not Oracle ARM performance. The single
preview's prefix only reaches its CSV onset frame at the end, so no alarm is
not proof of a normal clip. The t-bone/head-on preview is one positive sample,
not a sensitivity/false-positive benchmark. The score is not calibrated and
depends on model/preprocessing/runtime/scene; do not substitute it for a safety
decision. These authenticated local-stack checks are **not a public deployment**.

## Bounds, persistence and verification

The production shared model executes **one job at a time** with FIFO admission
for up to ten outstanding jobs (one worker plus nine waiting slots). The existing
nine-camera batch is accepted without bypassing authentication or source checks.
Accepted jobs are durably `pending` until their turn, then `running`; overload
returns 503 with `Retry-After`. Development keeps its previous ten-worker default
and an outstanding-job cap of ten.
The API container has 2 CPUs, 4 GB RAM, 128 PIDs, a read-only root filesystem,
a non-root UID and no Linux capabilities. Uvicorn admits at most 32 simultaneous
HTTP tasks/connections, and Caddy caps request bodies at 16 KB.
Production registers an `on_predict_start` thread-budget callback because
Ultralytics resets PyTorch CPU threads during first predictor device selection.
The configured two-thread count is reapplied before first/repeated prediction
inference; development models are not modified by this production-only callback.
Video open/read timeout is 10 seconds;
the proxy uses the same `JOB_TIMEOUT_SECONDS` setting with a 30-second
read/inference margin (330 seconds by default, bounded at 3630 seconds).
Its ten-second idle timeout and 16-connection limit remain independent
protections. Changing the deadline recreates both API and egress services;
do not configure different values or leave an older proxy running.
the production 300-second deadline starts **when execution begins**, not while waiting in
the queue, and is checked between frames, so a single read/inference
can overshoot it. Frames larger than 1280 pixels on their longest edge are
downscaled **in production only**, which can affect pixel-based scoring.
The evaluation layer owns the API frame-count contract.
The 120-frame frontend default is **not** a guaranteed sub-300-second run:
CPU contention, scene density, decoding and tracking can make it exceed that
deadline. An isolated nano run (1280 downscale, two threads)
processed 120 labeled frames in **31.5 seconds on the evaluation host**, whereas
a contended raw-resolution run took 319.1 seconds. These are different
environments, not an accuracy-equivalent model comparison or a production VM
benchmark. That isolated run used `INTER_AREA` resizing, while production uses
OpenCV's default `INTER_LINEAR`; its scores and timing are not exact
production-preprocessing comparisons. A corrected evaluation must use
`INTER_LINEAR` with the same target dimensions and thread settings.
The deployment retains a configurable 300-second execution cap with
explicit terminal timeout failures, not complete partial results. Actual native
ARM camera/VM timing remains unmeasured and must be checked before claiming throughput.

At most 300 frame triplets are saved per job, with a 2 MB limit per JPEG. The
most recent **nine** completed/failed saved jobs plus the active job retain
artifacts, preserving a full nine-camera batch at completion. Active jobs are
never pruned. Worst-case image storage is approximately **18 GB**
(10 jobs x 300 frames x 3 JPEGs x 2 MB); ensure at least 20 GB of free disk
space beyond OS/image/database needs, within the account's free storage quota.
Histories are capped at 100 jobs and expire after
24 hours. A subsequent batch can evict older terminal images before their
history expires; image URLs can then return 404. Production startup rejects an
artifact retention count smaller than nine. Failed
partial writes are cleaned; cleanup failures are logged and require operator
attention. Docker logs and temporary storage are also bounded.

SQLite uses WAL/full synchronization. Executor failures and unexpected worker
exits record terminal failures, rather than leaving accepted jobs orphaned.
Graceful shutdown fails waiting jobs explicitly and cancels their queued tasks;
an already-running job may finish before the process exits.
Restarting retains completed job results,
incident overrides and saved images; interrupted pending/running jobs become
explicit failures rather than being silently replayed. Expired/evicted jobs
and their images are removed. Named volumes are durable across container
recreation, **not** backups or protection against VM reclamation. Stop the API
before taking a consistent backup of the data volume; copy backups to separate
authorized storage within its free quota. Do not use `down --volumes` on real
data.

For a real deployed smoke, set `API_TOKEN` privately in your shell (do not paste
it into shared command logs), then:

```sh
python3 deploy/smoke.py https://YOUR_API_DOMAIN \
  --stream-url 'YOUR_EXACT_APPROVED_HTTPS_STREAM_URL'
# Exercise nine submissions and fetch every image only after the batch finishes.
python3 deploy/smoke.py https://YOUR_API_DOMAIN \
  --stream-url 'YOUR_EXACT_APPROVED_HTTPS_STREAM_URL' --batch-size 9
# Save the job_id from the successful response, then restart without deleting volumes.
docker compose --env-file deploy/.env -f deploy/compose.yaml restart api
python3 deploy/smoke.py https://YOUR_API_DOMAIN --existing-job YOUR_SAVED_JOB_ID
```

The smoke verifies HTTPS/certificate trust, readiness, unauthorized/wrong-token
rejection, CORS, metadata-URL rejection, real inference and protected JPEG
retrieval; the second invocation verifies persistent results/images after restart.
There is no skip or success-shaped fallback for a failed inference job.
At the measured evaluation-host rate, nine 120-frame jobs would take roughly
five minutes if that rate held; this is not a production promise.
At the default 300-second cap, ten outstanding jobs can require roughly
50 minutes plus read/inference overshoot. The smoke allows one hour for an
occupied queue by default; frontend polling must also allow queue wait plus
execution time, not just the per-job limit. If the operator changes the execution
cap or queue capacity, set `--timeout-seconds` to cover
`(MAX_PENDING_JOBS + 1) * JOB_TIMEOUT_SECONDS` plus read/inference/network margin,
adjust frontend polling accordingly, and remeasure the actual VM.

Dependency locks can be refreshed deliberately with `uv pip compile`, Python
3.11, `--generate-hashes`, and respectively `--python-platform
x86_64-manylinux_2_28` / `aarch64-manylinux_2_28`; see each lock's generated command.
Rebuild and rerun the smoke after dependency/model/config changes. Keep a
previous reviewed image and back up persistent state before deployment updates.

The [native ARM workflow](../.github/workflows/native-arm-runtime.yml) repeats
the actual image build and offline synthetic runtime check on relevant pushes
and pull requests, or manual dispatch. It uses GitHub's standard ARM hosted
runner (free for this public repository), read-only repository permissions and
a SHA-pinned checkout with no persisted credentials. It neither provisions
cloud resources nor publishes an image/artifact. `deploy/verify_cpu_runtime.py`
also runs against an existing amd64 image with `--expected-machine x86_64`;
its synthetic decoder fixture does not stand in for the separately verified
authenticated network/video smoke.
