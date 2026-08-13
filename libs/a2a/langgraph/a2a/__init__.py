"""Serve a compiled LangGraph as an Agent2Agent (A2A) 1.0 agent.

```python
from langgraph.a2a import create_a2a_app

app = create_a2a_app(
    graph,
    name="research-assistant",
    description="Searches the web and summarises what it finds.",
    version="1.0.0",
    url="https://agents.example.com/a2a",
    task_store=BaseStoreTaskStore(store),
    security_schemes={"bearer": ...},
    context_builder=MyAuthenticatingContextBuilder(),
)
```

The mapping this package implements:

| LangGraph | A2A |
|---|---|
| `thread_id` | `contextId` — the conversation, spanning every task in it |
| a run of the graph | part of a task — one task spans every run it takes to answer |
| `interrupt()` | `input-required`, resumed on the same `taskId` |
| `CredentialRequest` passed to `interrupt()` | `auth-required` |
| final state | an `Artifact`, with structured output as a `data` part |
| token stream | appended artifact chunks |
| node raises | `FAILED`, with a correlation id and no detail |
| node raises `TaskRejected` | `REJECTED`, with the reason |
| tools and subgraphs | derived `AgentSkill`s, off by default |

Requires Python 3.11 or later: `langgraph.config.get_config()` refuses to run in
an async context below that, and `interrupt()` calls it, so a pause on 3.10
fails the task rather than parking it.

One conversation is one thread, which is what makes a second turn remember the
first — A2A makes a task terminal when it completes, so a follow-up is a new
task carrying the same `contextId`. The cost is that tasks in a conversation are
serialised, and one arriving while the conversation is parked at
`input-required` is rejected rather than queued.
"""

from langgraph.a2a.authorization import (
    ContextAuthorizer,
    InMemoryContextAuthorizer,
    StoreContextAuthorizer,
    SubjectResolver,
    context_namespace,
    principal_subject,
    state_subject,
    subject_scope,
    thread_id,
)
from langgraph.a2a.card import build_agent_card, compat_card, derive_skills
from langgraph.a2a.executor import LangGraphAgentExecutor, TaskRejected
from langgraph.a2a.extension import CONVERSATION_URI as CONVERSATION_MODEL_EXTENSION_URI
from langgraph.a2a.extension import URI as DURABLE_INTERRUPT_EXTENSION_URI
from langgraph.a2a.interrupts import CredentialRequest
from langgraph.a2a.server import (
    A2AConfigurationError,
    A2AServer,
    add_a2a_routes,
    build_a2a_server,
    create_a2a_app,
)
from langgraph.a2a.state import StateAdapter, last_ai_text
from langgraph.a2a.task_store import BaseStoreTaskStore, ConditionalTaskStore

__all__ = [
    "CONVERSATION_MODEL_EXTENSION_URI",
    "DURABLE_INTERRUPT_EXTENSION_URI",
    "A2AConfigurationError",
    "A2AServer",
    "BaseStoreTaskStore",
    "ConditionalTaskStore",
    "ContextAuthorizer",
    "CredentialRequest",
    "InMemoryContextAuthorizer",
    "LangGraphAgentExecutor",
    "StateAdapter",
    "StoreContextAuthorizer",
    "SubjectResolver",
    "TaskRejected",
    "add_a2a_routes",
    "build_a2a_server",
    "build_agent_card",
    "compat_card",
    "context_namespace",
    "create_a2a_app",
    "derive_skills",
    "last_ai_text",
    "principal_subject",
    "state_subject",
    "subject_scope",
    "thread_id",
]
