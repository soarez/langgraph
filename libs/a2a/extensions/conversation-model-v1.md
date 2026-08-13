# Conversation model, v1

**URI** — `https://langchain.dev/a2a/extensions/conversation-model/v1`

The text this URI must serve. See `durable-interrupt-v1.md` for why these
documents live in the repository.

## What it declares

A `contextId` is one conversation, held on one thread of state. Every task in it
reads and writes that one thread, which is what makes the second turn of a
conversation remember the first — A2A makes a task terminal when it completes,
so a follow-up is a *new task with the same `contextId`*, and it would otherwise
start from nothing.

One thread is one serial lineage of checkpoints. Two consequences follow, and
this extension states them so a caller does not have to discover them:

**One task at a time.** Two turns that overlap in time did not exist for each
other when they were composed, so combining their results afterwards preserves
the data and cannot repair the reading — the later answer is written without the
earlier question, and every later turn reads that as history. A conversation is
a sequence in which each turn is composed in the light of the ones before it.
That is what the word means, and overlapping turns cannot have it.

**What an arriving task meets**, and the boundary between waiting and being
refused is whether the wait is bounded:

| The conversation is | The arriving task |
|---|---|
| running another task | waits for it. A run is bounded by the agent's run timeout, so the wait ends and the caller gets a slower answer, not an error |
| parked on a question asked more recently than `pause_deadline_seconds` | is refused with `UnsupportedOperationError` naming the blocking task |
| parked on a question asked longer ago than that | displaces it: the parked task goes terminal with a stated reason and the arriving task proceeds |

The refusal names the blocking task so the exchange becomes a handshake: fetch
it, see whether it is running or waiting, then answer it, cancel it, or wait.
A2A has no "busy" error, so the refusal travels under a permanent-sounding code
— that is why this extension exists.

**The deadline is a release, not a lifetime.** It is measured from the moment
the question was published, and **nothing sweeps**: the only thing that ever
compares against it is a second task arriving at the same conversation. A pause
nothing is waiting behind stays parked and answerable indefinitely, which is what
an approval that legitimately waits days needs. The value is published as
`pause_deadline_seconds` in this extension's `params`.

**Cancelling the blocking task releases the conversation** the same way. The next
turn then runs against a graph still parked at the abandoned question and
discards it. That is correct there and only there: proceeding against a pause a
live task still owns would destroy the question silently, and the answer arriving
later would be accepted, change nothing, and return the previous state as though
it were fresh.

## What it costs, and what it does not

Parallelism is not lost; it moves up a level. Work that does not need this
conversation's memory belongs in a conversation of its own, where it runs fully
in parallel with no coordination at all. The unit of parallelism is the
conversation, not the task — which is why the refusal says so.

No client meets this limit by accident. It has to set `contextId` itself, and
then open a second task before the first has finished.

## What may change

A later per-context queue with a bounded wait would turn the refusal into a
delay. That is a compatible relaxation: a caller written against this document
keeps working, because a request that used to be refused would then succeed
after waiting.
