# Durable interrupt, v1

**URI** — `https://langchain.dev/a2a/extensions/durable-interrupt/v1`

An A2A extension URI is expected to resolve to a description of the extension.
This is the text that URI must serve. Until it is published, a deployment can
point `create_a2a_app(extension_uri=...)` at its own copy, and a deployment that
does not want to advertise an unresolvable URI can pass
`durable_interrupt_extension=False`.

## What it declares

`input-required` on this agent is a graph suspended at a checkpoint, not a turn
that ended with the agent announcing it was confused. The protocol cannot tell
those apart; this extension says which one you are talking to.

| | |
|---|---|
| Advertised when | the agent's graph is compiled with a checkpointer. An agent that cannot suspend does not advertise this. |
| `required` | off by default. Marking it required would refuse every conformant client that has not heard of this URI. |

## The guarantees

**The pause survives the process.** A task at `input-required` can be answered
after the server has restarted. Its state is in the checkpointer, not in memory.

**Pauses are individually addressable.** A graph can pause in several parallel
branches at once. Each pause is a `data` part carrying an `interrupt_id`, and an
answer names the id it answers, so two open questions do not collide.

**Answering re-executes the paused node from its start.** This is the cost, and
it is the half a caller is most likely to be surprised by. The graph resumes at
the node that paused, not at the line that paused, so every effect between that
node's entry and its pause happens again. The pre-pause body runs *at least
once* per answer. An agent that charges a card before asking for confirmation
will charge it twice; an agent that reads a row, asks, and then writes will read
it twice.

## The wire format

A pause, in the task's status message:

```jsonc
{
  "data": { "interrupt_id": "3f2a…", "node": "approve", "payload": { "question": "Ship it?" } },
  "metadata": { "langgraph_a2a_kind": "interrupt" }
}
```

`langgraph_a2a_kind` is `credential_request` instead when the agent is asking
for a credential, and the task state is `auth-required` rather than
`input-required`.

The answer, on the same `taskId`:

```jsonc
{
  "data": { "interrupt_id": "3f2a…", "value": true },
  "metadata": { "langgraph_a2a_kind": "interrupt_response" }
}
```

`value` is any JSON value. A client that does not implement this extension can
send plain text instead, which answers the pause when exactly one is
outstanding; with more than one outstanding, a text answer is refused, because
nothing in it says which question it answers.

The same prompt is always present as text in the status message, so a text-only
caller can render the question without understanding any of the above.
