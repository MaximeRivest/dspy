# TLS-terminating broker — design + go/hold verdict

**Status:** DESIGN, pending Maxime's decision. Today: 2026-08-12.
**Author bar (Maxime):** "Design it, but HOLD and say so clearly if we
cannot do it in a way that any amateur Python user, in any setting
(Databricks, corporate proxy, etc.), will not struggle with. If we can
make it automated, yes."

This document answers that bar first (the verdict), then gives the
design the verdict is about.

---

## TL;DR verdict: **HOLD the on-by-default version. Offer a narrow, opt-in version for one clearly-bounded case.**

A TLS-terminating broker **cannot** be made amateur-safe and automatic
in every setting. The blocker is not our code — it is that terminating
TLS means **decrypting the user's traffic to their AI provider**, and
doing that silently is exactly the move corporate security, Databricks,
and every hardened environment are built to detect and forbid. Making it
"just work" everywhere requires installing a **local certificate
authority (CA)** the machine trusts — and that step is the one that
*genuinely* breaks for amateurs and is *rightly* blocked in managed
environments.

So the honest split:

- **The full "secrets never touch the child, even for HTTPS" guarantee**
  needs TLS termination, which needs a trusted local CA. That step is
  **not** automatable safely for a general amateur audience. → **HOLD.**
- **The secret-safety we already have** (the env-var channel) plus the
  broker's **allowlist + audit log over HTTPS** (via `CONNECT`, no
  decryption) already work everywhere with zero setup. → **keep, and
  document as the default.**
- A **TLS-terminating broker as an explicit, opt-in tool** for the
  bounded case "I am running fully-untrusted optimizer-authored code and
  I accept a local CA on THIS machine" is buildable and honest — but it
  is a power-user switch, never the default, and it must refuse loudly
  rather than half-work when the CA cannot be trusted.

The rest of this doc explains *why* that is the right line, so you can
overrule it with full information if you want the opt-in tool built now.

---

## What the broker is for (recap in one paragraph)

When the optimizer scores a candidate, the candidate's code may run
isolated (D-042). It sometimes needs to reach a real AI provider. We do
**not** want the untrusted child to hold the API key — a key inside the
sandbox is a key already leaked to whatever code is in there. The broker
is a parent-owned proxy: the child talks to the broker, the broker holds
the key and attaches it on the way out, and the broker only allows a
declared list of hostnames (with a full request log). Generated code
cannot leak a key it never held, and "what did this candidate reach" is
one list to read.

## The exact thing that works today, and the exact thing that doesn't

The broker (`dspy/optim/broker.py`) has two paths:

- **Plain HTTP forward path** — the broker sees the full request, so it
  can **replace the `Authorization` header** with the real key. Secret
  injection works. This is what the localhost-stub tests prove.
- **HTTPS `CONNECT` tunnel path** — the browser/client opens an
  *encrypted* tunnel to the provider through the broker. The broker can
  **allow or deny by hostname and log it**, but it **cannot read or
  modify** the bytes — TLS is end-to-end between the child and the
  provider. So it **cannot inject the key**.

Real providers (Anthropic, OpenAI, Azure, Bedrock, Databricks model
serving) are **all HTTPS**. So today, to reach a real provider from the
scoring child, the key must travel to the child through the environment
variable channel (which we built and which is safe against *disk* leaks,
but the key does physically enter the child process — visible in that
child's `/proc/<pid>/environ`).

**TLS termination** is the only way to inject a secret into an HTTPS
request the child makes: the broker pretends to *be* the provider to the
child (presenting a certificate the child trusts), decrypts, injects the
key, re-encrypts to the real provider. This is "man-in-the-middle, but
it's your own machine doing it on purpose" — the mitmproxy / corporate
web-filter pattern.

## Why "trusted local CA" is the wall for amateurs

For the child's HTTPS client to accept the broker impersonating
`api.anthropic.com`, the broker must present a certificate for that
hostname signed by a CA the child **trusts**. There is no trusted CA on
Earth that will sign `api.anthropic.com` for you (correctly — that would
be a global security failure). So you must **create your own local CA and
add it to the trust store** the child uses. That step is where it breaks:

1. **It is genuinely fiddly for amateurs.** Python's TLS trust is a
   maze: `certifi`'s bundle, the OS trust store, `SSL_CERT_FILE`,
   `REQUESTS_CA_BUNDLE`, `httpx`/`aiohttp` each with their own knobs, and
   provider SDKs that sometimes pin or bypass. Getting a self-signed CA
   honored by *whatever HTTPS stack the candidate happens to use* is not
   a one-liner; it is the thing people file week-long support threads
   about. We can automate the common case (point the child's
   `SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE` at our CA bundle), but "any
   client in any setting" is precisely what we cannot promise.

2. **Managed environments block it by design — and should.** On
   Databricks, corporate laptops behind a proxy, locked-down CI, the
   trust store is **not yours to modify**, and a process that tries to
   MITM outbound TLS is a red flag the platform actively prevents. In a
   corporate-proxy setting there is *already* a TLS-terminating CA in the
   path (the company's). Stacking ours on top, or fighting theirs, is a
   mess an amateur has no chance of debugging — and if they somehow
   succeed, they may have just weakened their employer's security
   posture. This is the opposite of "just works."

3. **The failure mode is scary, not graceful.** When trust is
   misconfigured, the symptom is `SSLCertVerificationError` deep inside a
   provider SDK, or — far worse — a "helpful" workaround where someone
   disables verification (`verify=False`) to make the error go away,
   which silently turns off *all* TLS security for that traffic. An
   amateur-facing feature whose natural failure nudge is "just turn off
   certificate checking" is a feature that makes people **less** safe. It
   fails your bar hard.

**Conclusion for the bar:** a general, on-by-default, automatic
TLS-terminating broker for all users in all settings is **not
achievable** without asking amateurs to do CA-trust surgery that is
brittle everywhere and forbidden (rightly) in exactly the managed
settings you named. So the default answer is **HOLD**.

## What we ship instead (the honest default — no build, already true)

State this plainly in the docs so nobody believes we do more than we do:

- **Default egress = env-var credential channel + HTTPS allowlist/log via
  `CONNECT`.** The child reaches only allowlisted provider hostnames; the
  broker logs every attempt; the key reaches the child through its
  environment (safe from disk/artifact leaks, not hidden from the child's
  own process). This works on Databricks, behind a corporate proxy,
  everywhere — **zero setup**.
- We do **not** claim "the child never holds the key" for real HTTPS
  providers under the default. We claim exactly what is true: the key
  never touches disk, the artifact, the job file, or argv; egress is
  allowlisted and logged; and for the strongest "child never holds it"
  property you must either (a) use a local/plain-HTTP gateway, or (b)
  opt into the TLS-terminating tool below and accept its requirements.

This is the same honesty rule you set for `gpu=`: never let the user
believe we enforce more than we do.

## The opt-in tool (buildable, power-user only) — if you want it

If and when you want the full "child never holds the key, even for
HTTPS" guarantee on a machine you control, this is the shape. It is a
**deliberate switch**, never a default, and it **fails closed**.

**API sketch:**

```python
broker = dspy.Broker(
    allow=["api.anthropic.com"],
    inject={"api.anthropic.com": dspy.SecretFromEnv("ANTHROPIC_API_KEY")},
    tls_terminate=True,              # explicit opt-in; default False
    ca_dir="~/.dspy/broker-ca",      # where the local CA lives (this machine only)
)
```

**Mechanism:**
1. On first use with `tls_terminate=True`, generate a local CA keypair
   in `ca_dir` (0600, never leaves the machine, never in any artifact).
2. Per allowlisted host, mint a leaf cert on the fly signed by that CA.
3. The child is launched with its TLS trust pointed **only** at our CA
   bundle for the allowlisted hosts (`SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE`
   set on the child env; documented that clients bypassing these are
   unsupported).
4. The broker terminates TLS from the child, injects the real
   `Authorization`, and opens a **normally-verified** TLS connection to
   the true provider (we verify the provider with the real CA bundle —
   we MITM the child, never the provider).

**The fail-closed rules (this is what makes it honest, not the feature):**
- If `tls_terminate=True` but we **cannot** establish that the child's
  client will trust our CA — refuse to start, with a message naming
  exactly what is unmet. **Never** fall back to the env-var channel
  silently (that would mean the user asked for the strong guarantee and
  got the weak one without being told).
- **Never** disable provider-side verification. If the real provider's
  cert doesn't verify, that's a real error surfaced loudly.
- **Never** suggest `verify=False` anywhere, in any error text.
- A self-probe at startup, like the isolation self-probes: make one
  request through the terminating path to a known allowlisted host (or a
  loopback stub) and confirm injection happened and trust held; abort as
  infrastructure if it didn't.
- Refuse `tls_terminate=True` when the environment shows an existing
  corporate MITM proxy in the path that we'd be fighting (detectable:
  `HTTPS_PROXY` already set to a non-loopback address) — tell the user
  their environment already terminates TLS and this switch is not for
  them.

**Scope of the opt-in tool:** "I control this machine, I'm running
fully-untrusted optimizer-authored code, and I want the child to never
hold the key even over HTTPS." That is a real and legitimate case (a
disposable box, a security researcher's rig). It is **not** the amateur /
Databricks / corporate case, and the tool says so.

## Recommendation

1. **Ratify the HOLD** on any on-by-default TLS-terminating broker. It
   cannot meet your amateur-in-any-setting bar, for reasons that are
   structural (trust stores) not fixable by better code.
2. **Ship the documentation truth now** (no build): default egress is
   env-var channel + HTTPS allowlist/log; we do not claim key-hiding for
   real HTTPS under the default. (Small docs change; I can fold it into
   the current cleanup wave.)
3. **Decide separately** whether you want the **opt-in power-user tool**
   built. My advice: defer it until a concrete need appears (you actually
   want to run untrusted-authored code against a real paid HTTPS provider
   on a box you own). It's a clean, self-contained stage whenever you
   want it, and the fail-closed design above is ready.

The one-line answer to your bar: **we cannot make TLS termination
amateur-safe and automatic everywhere — so it stays a deliberate,
fail-closed, power-user switch, and the everyday default keeps the honest
allowlist+log+env-var story that already works in every setting.**
