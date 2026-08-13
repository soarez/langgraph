# 🦜🕸️ LangGraph A2A

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Serve a compiled LangGraph as a conformant [Agent2Agent](https://a2a-protocol.org)
1.0 agent.

A2A is how an agent delegates work to an agent it does not own — different
team, different framework, different company. This package is the protocol
layer: it turns a graph into an A2A server, and nothing else. It depends on the
official [`a2a-sdk`](https://github.com/a2aproject/a2a-python) and `langgraph`.

```bash
pip install langgraph-agent2agent
```

## Quickstart

```python
from langgraph.a2a import BaseStoreTaskStore, StateAdapter, create_a2a_app

app = create_a2a_app(
    graph,  # compiled, with a checkpointer
    name="research-assistant",
    description="Searches the web and summarises what it finds.",
    version="1.0.0",
    url="https://agents.example.com/a2a",  # where callers reach it
    task_store=BaseStoreTaskStore(store),
    security_schemes={"bearer": ...},
    context_builder=MyAuthenticatingContextBuilder(),
    state=StateAdapter(output_data=lambda s: s.get("result")),
)
```

`add_a2a_routes(existing_app, graph, ...)` mounts the same routes into a
Starlette or FastAPI application you already have. `build_a2a_server(...)`
returns the card, handler and routes for anything more unusual.
`LangGraphAgentExecutor` is available for composing the SDK directly.

## What it maps

| LangGraph | A2A |
|---|---|
| `thread_id` | `contextId` — the conversation, spanning every task in it |
| a run | part of a task — one task spans every run it takes to answer |
| `interrupt()` | `input-required`, resumed on the same `taskId` |
| `CredentialRequest` passed to `interrupt()` | `auth-required` |
| `Command(resume=...)` | the caller's next message on that task |
| what the model said | one `Artifact` — the transcript, opened by the first token and appended to |
| `output_text` / `output_parts` | that mapping's result instead, read from the final state, with no tokens streamed into it |
| `output_data` | a `data` part appended when the run ends, in either shape |
| node raises | `FAILED`, with a correlation id and no detail |
| node raises `TaskRejected` | `REJECTED`, with the reason |
| tools and subgraphs | derived `AgentSkill`s, off by default |
| `BaseStore` under `context_namespace` | memory a deployment wants outside the thread |

## The conversation is the thread

`contextId` maps to `thread_id`, and a task is one unit of work inside it. This
is the decision most of the rest follows from.

A2A makes a task terminal when it completes, so the second turn of a
conversation is a *new task carrying the same `contextId`*. Key the thread on
the task and every turn after the first begins on an empty checkpoint: the agent
answers each message perfectly and remembers nothing, which is the one thing a
checkpointer was there to prevent.

The price is concurrency, and it is not really LangGraph's price. Two turns that
overlap in time did not exist for each other when they were composed; combining
their results afterwards preserves the data and cannot repair the reading, and
every later turn reads the incoherent transcript as history. A conversation is a
sequence in which each turn is composed in light of the ones before it.

So: **one task at a time per conversation.**

What an arriving task meets, and the boundary is whether the wait is bounded:

| The conversation is | The arriving task |
|---|---|
| running another task | **waits** for it. A run is bounded by `run_timeout`, so the wait ends |
| parked on a question asked more recently than `pause_deadline` | **refused** with `UnsupportedOperationError` naming the blocking task. Nothing obliges a counterparty to answer, so queueing here queues behind an event that may never happen |
| parked on a question asked longer ago than that | **displaces** it: the parked task goes terminal with a stated reason and this one proceeds |

The deadline is measured from the moment the question was published, and
**nothing sweeps** — it is evaluated only when another task arrives. A pause
nothing is waiting behind stays parked and answerable indefinitely, which is
what an approval that legitimately waits days needs. Cancelling the blocking
task releases the conversation the same way.

Work that does not need this conversation's memory belongs in a conversation of
its own, where it runs fully in parallel — which is what the refusal tells the
caller.

Both the rule and the deadline are stated on the card as an extension, so a
caller discovers them instead of meeting them. A later per-context queue with a
bounded wait would turn the refusal into a delay, which is a compatible
relaxation.

A *parked* conversation is excluded from the task store, so a pause blocks new
turns after a restart too. Exclusion between **processes** is a seam: a store
that implements `ConditionalTaskStore`. Declare `multi_replica=True` and the
server refuses to start without one, rather than letting an in-process lock look
like exclusion — beside a second replica that lock produces no error, it
produces a lost turn both callers were told had succeeded.

The first row of that table changes with the declaration, because the wait has
to move somewhere another process can see. Under `multi_replica=True` a turn
**claims** the conversation with one conditional write of its own task at
`WORKING`; a store that can exclude refuses that write while another task is
already working the same context, and the arriving turn retries until it
succeeds. The release is the ordinary write that ends a turn — completed,
failed, or parked at a question — so nothing has to remember to let go.

Bounded by `run_timeout`, after which the arriving caller is refused with the
holder named. Nothing here clears a claim whose owner died: that is the cost of
having no lease, and `tests/test_two_replicas.py` drives the whole arrangement
as two OS processes rather than two objects in one event loop, against the
SQLite store in `tests/replicas/` — one file of SQL, and the shape of what a
deployment supplies.

## Human in the loop

`interrupt()` suspends the graph at a checkpoint. The task goes
`input-required` and the caller answers it later — after a restart, if need be.
The pause is reported twice in the same status message: as a `data` part marked
as a function call carrying the interrupt id, and as text, so a caller that
speaks neither extensions nor structured answers can still reply in prose.

```python
def approve(state):
    decision = interrupt({"question": "Ship it?", "diff": state["diff"]})
    ...
```

Answering:

```jsonc
// structured — names which pause it answers, so parallel pauses are independent
{"interrupt_id": "3f2a…", "value": true}
// or just send text, when exactly one pause is outstanding
```

```jsonc
// the pause, in the task's status message
{"interrupt_id": "3f2a…", "node": "approve", "payload": {"question": "Ship it?"}}
```

Both parts carry `metadata["langgraph_a2a_kind"]` — `interrupt` or
`credential_request` on a pause, `interrupt_response` on an answer. The
id-correlated shape is modelled on Google ADK's, which solves the same problem,
but the keys are this project's: ADK's are prefixed into its own namespace and
its credential request carries a schema this package does not implement, so a
client that recognised them would mis-handle what we send. Compatibility with a
specific client is a claim to test with that client in the loop.

Three rules keep resumption sound, each of them a defect if dropped: the pause
set is taken from what the run actually raised (the stored state over-reports an
answered pause in one direction and hides a re-raised one in the other), a resume
is addressed at the checkpoint the task paused at rather than the thread's head,
and a task marked `input-required` whose graph has nothing pending is failed
loudly rather than re-run — LangGraph accepts a resume with nothing pending, does
nothing, and returns the previous state, which would otherwise reach the caller
as a fresh answer.

Two things this costs, both deliberate and both tested:

- Resuming re-executes the paused node from its start, so side effects between
  the node's entry and its `interrupt()` call happen twice. That is LangGraph's
  semantics; this package does not hide it.
- Interrupt payloads pass through an encoder (pydantic models, dataclasses,
  bytes, enums). `interrupt()` accepts any Python object, and one that will not
  serialise becomes its `repr` rather than a protocol error.

A graph compiled **without** a checkpointer is served, without the pause: the
durable-interrupt extension is not advertised, turns do not accumulate because
there is no thread to carry them, and an `interrupt()` that happens anyway fails
the task. `input-required` means "ask me again", and a run that was never
suspended has nowhere for the answer to arrive.

## What the answer is

Two shapes, and the adapter picks which:

- **No output mapping — the artifact is the transcript.** The first token opens
  it and every visible model token appends. A graph that calls tools speaks more
  than once and no run knows which turn is its last until it ends, so what is
  assembled while tokens arrive is a record of what was said, not a verdict on
  which part was the answer.
- **`output_text` or `output_parts` — the artifact is that mapping's result**,
  read from the final state when the run ends. A result computed after the run
  cannot be streamed without lying about what is arriving, so supplying the
  mapping is what turns token streaming off. There is no second switch:
  `stream_tokens=False` changes when transcript bytes arrive, never what they
  are.

`output_data` is additive in both: a `data` part appended when the run ends.
Either way the answer is delivered once, in one object, and what a caller
reconstructs from the stream is byte-for-byte what `GetTask` returns.

## What it refuses to start as

Several configurations produce a server that looks correct and is not, so
`create_a2a_app` rejects them:

- **No authentication, no declaration.** `contextId` is chosen by the caller,
  so without an authenticated caller a second one presenting the same id is
  indistinguishable from the first. Pass a `context_builder` that
  authenticates, or say `single_tenant=True`.
- **A durable checkpointer with an in-memory task store.** The graph's state
  would survive a restart while the task waiting for an answer did not, and the
  caller's reply would arrive as a fresh turn.
- **A state key the graph does not declare**, and a required durable-interrupt
  extension on a graph with no checkpointer. Both are promises the
  configuration cannot keep, and both fail at the first request rather than at
  startup unless something checks.
- **Webhook registrations less durable, or keyed differently, than the tasks
  they belong to.** An in-memory config store beside a durable task store loses
  registrations the tasks referring to them survive; a config store keyed by the
  authenticated principal — the SDK's default — disagrees with a task store
  keyed by the subject the moment a credential rotates. Pass
  `InMemoryPushNotificationConfigStore(owner_resolver=subject_scope)`, or a
  durable store resolved the same way.

Identity is checked past the context, too. The SDK loads the task and then
passes `task=None` into its context builder, so its own `contextId`/`taskId`
agreement check never runs: a caller can present a victim's `taskId` alongside a
context it legitimately owns, pass the ownership check, and advance someone
else's task. The conversation is therefore taken from the **stored task**, a
request whose `contextId` disagrees with it is refused, and so is a message to a
task already terminal.

Beyond that: inbound `data` parts reach one declared state key and no other; a
node's exception reaches the caller as a correlation id and never as its text,
including for failures raised before the run starts; a run has a bound
(`run_timeout`, ten minutes by default) so no task sits at `WORKING` for the
life of the process; and a completed task always carries an artifact, even when
the graph produced nothing.

### Who a conversation belongs to

A `contextId` is bound on first use, and a later mismatch is reported as a
missing task — the specification does not permit distinguishing "not yours" from
"not there".

It binds to a **subject**, not to the caller's credential. A2A's caller is
normally an agent presenting one service credential on behalf of many end users,
and 1.0 has no on-behalf-of field, so binding to the credential would collapse
every user behind that peer into one owner:

```python
create_a2a_app(graph, subject_resolver=state_subject("end_user"), ...)
```

`state_subject` reads a claim the `ServerCallContextBuilder` extracted from the
verified credential. It has to arrive that way: `metadata` is a field on
`SendMessageRequest` and on none of `GetTaskRequest`, `ListTasksRequest` or
`SubscribeToTaskRequest`, so a subject carried in a message is present on write
and absent on read, and a store scoped by it hides a task from its owner or
hands it to someone else. A security scheme that carries the subject, or an
extension header, both reach the context builder — which is where this package
resolves it, once, for every RPC.

The subject is deliberately **not** part of `thread_id`: a rotated key or a
renamed principal must not fork the conversation onto an empty thread.

## Conformance

The official [A2A TCK](https://github.com/a2aproject/a2a-tck) is pinned
(`5996b79`) and run against a SUT built from this package:

```bash
make tck
```

Latest run — 74/76 MUST, 4/7 SHOULD, 4/4 MAY (16 MUST skipped: gRPC and
HTTP+JSON transports; 22 not exercised by the suite):

| Failing | Level | Owner | Why |
|---|---|---|---|
| `DM-MSG-001` | MUST | this package, deliberate | Wants a bare `Message` answer instead of a task. This package always opens a task: a task is what survives a restart, a resume and a cancellation. |
| `JSONRPC-SSE-002` | MUST | `a2a-sdk` | 1.1.2 parses the body before checking `Content-Type`, so a wrong one is `ParseError`, not `ContentTypeNotSupportedError`. |
| `DM-SERIAL-005` | SHOULD | `a2a-sdk` | 1.1.2 rejects unrecognised request fields instead of ignoring them. |
| `CORE-HIST-005/006` | SHOULD | `a2a-tck` | The suite reuses one `messageId` across turns; `a2a-sdk` 1.1.2 de-duplicates `Task.history` by `messageId`, so the repeats are dropped. Reproduced directly against the server. |

The gate is "no MUST failure outside this list" — not "everything passes", which
would be either permanently red or quietly deleted. Coverage is one of A2A's
three bindings; gRPC and HTTP+JSON are not served and their requirements are
reported skipped, not passed.

`tests/test_tck.py` gates on that list: a MUST failure outside it is a
regression, and a member of it that starts passing is a note to delete —
`PUSH-DELIVER-001` left the list that way, once this package started sending the
credentials a caller registers instead of the header the SDK sends.

Push notifications also come with a destination policy, because a registered
webhook is a URL a caller chose and the server then makes requests to: https,
and no loopback, link-local or private address unless the deployment vouches for
its own network with `allow_private_webhooks=True`. Checked when the config is
registered, so a caller learns at once rather than never hearing back.

The TCK drives an agent through scenarios keyed by a `messageId` prefix, so the
SUT (`tests/tck/sut.py`) is a graph that cooperates with that signal — the
prefix reaches it through `config_from_context`. Everything under the graph is
the shipped server.

## What this package does not do

Every guarantee here holds **within one process**. Anything that has to hold
across processes is a seam the deployment fills, with an in-memory
implementation for development and a refusal at construction when a seam that
was left empty cannot keep a promise the card makes.

| Not done here | The seam |
|---|---|
| Cross-process exclusion | `ConditionalTaskStore` |
| Webhook delivery that survives a crash | `PushNotificationSender` |
| Stream re-attachment across replicas | the SDK's `QueueManager` |
| Cancelling a run in another process | — the run belongs to the process that owns it, and the server says so |
| Anything that runs unprompted | — nothing sweeps; reconciliation happens at start |

The package does not run a server, pick a database, coordinate replicas, or
deliver webhooks itself. That is the same line `a2a-sdk` draws.

## Deployment

**One replica, unless the store you supply can exclude.** Submit-and-poll,
`SubscribeToTask` and webhooks all work — and all of them work *while the
accepting process lives*. None survives it: a task's progress is tied to the
process running it, so only a paused task is durable across a restart, because
only a pause has a checkpoint to resume from.

A `WORKING` task whose process died is not recovered. It is **failed with a
stated reason at the next start**, so a poller gets an answer instead of a task
that says `WORKING` for ever — `A2AServer.astart()` does that, and skips it when
`multi_replica=True`, where the same scan would close another replica's live
work.

The default `InMemoryContextAuthorizer` is process-local for the same reason —
use `StoreContextAuthorizer` if you run more than one process, and read what it
says about the write it cannot make atomic.

`multi_replica=True` is therefore a declaration with two consequences: the
construction-time refusal above, and reconciliation switching itself off.
Recovering the work of a dead process — a lease, or a handoff — is not something
this package does; see [what it does not do](#what-this-package-does-not-do).

## Versions

Verified against `a2a-sdk` 1.1.2, `langgraph` 1.2.11, `protobuf` 6.33.6, on
CPython 3.11 and 3.12.

**Python 3.11 or later**, which is one version above the rest of the monorepo
and not a preference. `langgraph.config.get_config()` refuses to run in an async
context below 3.11, and `interrupt()` calls it — so on 3.10 every pause fails
the task instead of parking it, and the pause is the whole point. The package
therefore declares `>=3.11` and is tested on 3.11 through 3.14 rather than on
the shared matrix. A2A's data model was rewritten at 1.0 and the SDK moves in
bursts, so the dependency range is bounded (`a2a-sdk>=1.1,<2`) and tested in
CI rather than left open.

A 0.3 counterparty is served too: the card declares both protocol versions, the
1.0 well-known path carries the legacy fields alongside the 1.0 ones, and the
0.3-shaped document is also served at `/.well-known/agent.json`
(`PREV_AGENT_CARD_WELL_KNOWN_PATH` in `a2a-sdk` 0.3). An `a2a-sdk` 0.3.26
client parses both and selects a transport; the 0.3 request path itself is the
SDK's compatibility mode and is not separately covered here.

## Example

`examples/approval_agent.py` is both halves of the exchange — a graph that pauses
for approval, and a client that answers it — in one runnable file:

```bash
uv run python -m examples.approval_agent          # the whole exchange, narrated
uv run python -m examples.approval_agent --serve  # just the server, on :8080
```

No credentials and no network, and `tests/test_example.py` runs it on every
build. An example nobody executes is a claim about the past.

## Development

```bash
uv sync
make test          # the acceptance suite, 135 checks
make tck           # conformance, clones the pinned TCK on first run
make lint format
```

The acceptance suite is driven end to end by the unmodified `a2a-sdk` client
against an app built by `create_a2a_app`:

| File | What it holds |
|---|---|
| `test_acceptance_core.py` | what any caller can rely on |
| `test_acceptance_hitl.py` | the pause: id-correlated, resumable, and its cost |
| `test_acceptance_continuity.py` | the agent remembers the conversation, across turns and across a restart |
| `test_acceptance_card.py` | nothing advertised that the configuration cannot honour |
| `test_acceptance_concurrency.py` | who may advance which task, and when |
| `test_acceptance_adversarial.py` | what a hostile or careless caller cannot do |
| `test_durability.py` | what survives the process |
| `test_two_replicas.py` | exclusion across replicas, driven as two OS processes |

The extension descriptions in `extensions/` are the documents the advertised
URIs must serve. Publishing them is a release step; until then a deployment can
point `extension_uri=` at its own copy, or pass
`durable_interrupt_extension=False`.

An agent with semantics of its own to advertise passes `extensions=[...]`, and
they are published beside those two exactly as given, `params` included. The
package withholds *its* own when the configuration cannot honour them — no
checkpointer, no pause extension — because it performs those two itself. It
cannot make that judgement about a graph, so honouring a declared extension
belongs to whoever declared it; a deployment that wants to act on one when a
caller activates it already has the request, through the context builder and
`config_from_context`.
