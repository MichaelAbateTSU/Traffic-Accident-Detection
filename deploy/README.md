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

Verified on this workspace's **amd64** Docker host: 142 tests with the real
locked ML dependencies, certificate-verified HTTPS inference over the isolated
proxy, authenticated JPEG retrieval, private/metadata egress rejection, and
preserved results/images after API restart and recreation, including a
300-frame worker regression with 900 saved JPEGs and durable results, plus a
nine-job HTTPS inference batch whose first/last images survive restart.
The arm64 lock was resolved against ARM wheels. An ARM64 build was attempted
on this amd64 host, but failed at the first shell instruction with
`exec /bin/sh: exec format error`: QEMU/ARM emulation is not installed or
advertised by its BuildKit worker. No native or emulated ARM runtime passed here.
Public ACME issuance, VM reboot and live-cloud availability remain unverified.

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
remove an AAAA record unless IPv6 actually reaches this stack. Docker Engine
with Compose v2 and outbound DNS/HTTPS are required. Allow inbound TCP 80/443
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
Video open/read timeout is 10 seconds;
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
ARM/VM timing remains unmeasured and must be checked before claiming throughput.

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
