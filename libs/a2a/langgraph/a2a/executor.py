"""An `AgentExecutor` that runs a compiled LangGraph behind A2A 1.0.

The three decisions worth knowing before reading the code:

**A task is not a run.** One A2A task spans as many graph runs as the agent
asks questions. Resumption is read from the task's own state in the task store,
never inferred from run status.

**A thread is a conversation, not a task.** `contextId` alone keys the thread,
because A2A makes a task terminal when it completes: the second turn of a
conversation is a new task carrying the same `contextId`, and a thread per task
would start it on an empty checkpoint with the agent remembering nothing. The
price is that one conversation is one serial lineage — tasks within it are
serialised here, and a task arriving while the conversation is parked at
`input-required` is refused rather than queued.

**A pause is a suspension.** `interrupt()` parks the graph at a checkpoint;
`input-required` reports that, and the resume continues the same task. Resuming
re-executes the paused node from its start, so side effects between the node's
entry and its `interrupt()` call happen again. That is LangGraph's semantics,
not something this package can hide.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from google.protobuf import json_format
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types.a2a_pb2 import (
    ListTasksRequest,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
)
from a2a.utils.errors import (
    A2AError,
    InvalidParamsError,
    TaskNotFoundError,
    UnsupportedOperationError,
)
from langgraph.a2a import extension, interrupts
from langgraph.a2a._encoding import encode as encode_payload
from langgraph.a2a.authorization import (
    ContextAuthorizer,
    SubjectResolver,
    context_namespace,
    principal_subject,
    subject_of,
    thread_id,
)
from langgraph.a2a.interrupts import AmbiguousResume, PendingInterrupt
from langgraph.a2a.parts import data_part, parts_text, text_part
from langgraph.a2a.state import StateAdapter
from langgraph.a2a.task_store import RUNNING_STATES, ConditionalTaskStore

if TYPE_CHECKING:
    from a2a.server.tasks import TaskStore

logger = logging.getLogger(__name__)

RESPONSE_ARTIFACT = "response"
"""Artifact id the answer is written to, streamed or not."""

STREAM_FLUSH_CHARS = 512
"""Token text buffered before a chunk is emitted."""

STREAM_FLUSH_SECONDS = 0.25
"""Longest a buffered chunk waits, so a slow model still looks alive."""

ORPHANED_INTERRUPT_MESSAGE = (
    "This task was waiting for an answer, but the graph has no pending "
    "interrupt to resume. The task cannot be continued."
)

EMPTY_RESULT_TEXT = "The agent completed without producing any output."
"""What a task reports when the graph's final state yields nothing readable."""

DEFAULT_RUN_TIMEOUT = 600.0
"""Seconds a single run may take before the task is failed.

Without a bound a task sits at `WORKING` for the life of the process, which no
caller can distinguish from an agent that is still thinking. Ten minutes is long
enough for a deep research turn and short enough to notice.
"""

DEFAULT_PAUSE_DEADLINE = 86_400.0
"""Seconds a question may go unanswered before its task is ended.

The one real cost of serialising a conversation. A pause is durable by design,
nothing in LangGraph expires a pending interrupt and nothing in A2A expires a
task, so a counterparty that simply stops replying would otherwise end the
conversation for ever. A day is long enough for a human in a different timezone.
"""

CHECKPOINT_METADATA_KEY = "langgraph_a2a_checkpoint"
"""Task metadata key holding the checkpoint a paused task must resume at."""

REQUESTED_CONTEXT_ID_KEY = "a2a_requested_context_id"
"""Call-context state key holding the `contextId` the caller actually sent."""

TERMINAL_STATES = (
    TaskState.TASK_STATE_COMPLETED,
    TaskState.TASK_STATE_CANCELED,
    TaskState.TASK_STATE_FAILED,
    TaskState.TASK_STATE_REJECTED,
)

PAUSED_STATES = (
    TaskState.TASK_STATE_INPUT_REQUIRED,
    TaskState.TASK_STATE_AUTH_REQUIRED,
)

CLAIM_POLL_INTERVAL = 0.25
"""How often a turn re-reads the store while another replica holds the thread."""


class TaskRejected(Exception):
    """Raise from a node to refuse the work outright: the task goes `REJECTED`.

    `FAILED` says the agent tried and could not; `REJECTED` says it will not.
    The distinction matters to a caller deciding whether to retry, so it is a
    separate exception rather than an error string. Unlike a failure, the
    message given here is shown to the caller — a refusal that will not say
    what was refused is not actionable.
    """


class LangGraphAgentExecutor(AgentExecutor):
    """Run a compiled graph as an A2A agent.

    Args:
        graph: a compiled graph. Compile it with a checkpointer if it uses
            `interrupt()`: `input-required` is only resumable if the graph's
            state outlives the run that produced it.
        state: how A2A messages map onto graph state and back. See `StateAdapter`.
        config_from_context: extra `RunnableConfig` per request — caller
            identity, tenant, recursion limit, timeouts. `thread_id` is derived
            here and cannot be overridden.
        context_authorizer: binds each `contextId` to its first subject.
        subject_resolver: who a request acts for. Defaults to the authenticated
            principal, which is right only when one credential means one
            subject.
        interrupt_encoder: maps an interrupt payload onto JSON-compatible data.
        stream_tokens: emit model token deltas as appended artifact chunks.
        require_extension: refuse callers that have not declared support for the
            durable-interrupt extension.
        extension_uri: the durable-interrupt URI this deployment publishes.
        run_timeout: seconds one run may take before the task is failed. `None`
            removes the bound, and with it any guarantee that a task leaves
            `WORKING`.
        pause_deadline: seconds a question may go unanswered before its task is
            ended and the conversation released. `None` lets one unanswered
            question freeze a conversation for ever.
        task_store: the store the server keeps tasks in, read to find out
            whether another task already holds this conversation.
        multi_replica: the deployment has declared that more than one copy of
            this process may be running. A turn arriving at a conversation
            another replica is working then waits on the store rather than on a
            run it cannot see.
    """

    def __init__(
        self,
        graph: Any,
        *,
        state: StateAdapter | None = None,
        config_from_context: Callable[[RequestContext], RunnableConfig] | None = None,
        context_authorizer: ContextAuthorizer | None = None,
        subject_resolver: SubjectResolver | None = None,
        interrupt_encoder: interrupts.PayloadEncoder = encode_payload,
        stream_tokens: bool = True,
        require_extension: bool = False,
        extension_uri: str = extension.URI,
        run_timeout: float | None = DEFAULT_RUN_TIMEOUT,
        pause_deadline: float | None = DEFAULT_PAUSE_DEADLINE,
        task_store: TaskStore | None = None,
        multi_replica: bool = False,
    ) -> None:
        self.graph = graph
        self.state = state or StateAdapter()
        self.config_from_context = config_from_context
        self.context_authorizer = context_authorizer
        self.subject_resolver = subject_resolver or principal_subject
        self.interrupt_encoder = interrupt_encoder
        self.stream_tokens = stream_tokens
        self.require_extension = require_extension
        self.extension_uri = extension_uri
        self.run_timeout = run_timeout
        self.pause_deadline = pause_deadline
        self.task_store = task_store
        self.multi_replica = multi_replica
        self._conversations: dict[str, asyncio.Lock] = {}

    # -- request setup -------------------------------------------------------

    def _conversation_lock(self, context_id: str) -> asyncio.Lock:
        """One lock per conversation, because one conversation is one thread.

        A thread is a serial lineage of checkpoints: two runs writing it at once
        lose each other's writes. A2A permits several live tasks in one context,
        so the serialisation has to be here. It is process-local and excludes
        nothing beyond this process — a deployment that declares itself
        multi-replica waits on the task store as well, which is the only thing
        a second replica writes where this one can read it.
        """
        lock = self._conversations.get(context_id)
        if lock is None:
            lock = self._conversations.setdefault(context_id, asyncio.Lock())
        return lock

    @staticmethod
    def _context_id(context: RequestContext) -> str:
        """The conversation this request belongs to.

        The task's own `contextId` wins over the request's. A caller resuming a
        task need only send its `taskId`, and when it does the SDK generates a
        fresh `contextId` for the request — following that would put the resume
        on a different thread from the pause it is answering.
        """
        if context.current_task is not None and context.current_task.context_id:
            return context.current_task.context_id
        return context.context_id or ""

    async def _subject(self, context: RequestContext) -> str:
        """Who this request acts for, and whether it may touch this conversation."""
        subject = subject_of(
            context.call_context, context, resolver=self.subject_resolver
        )
        context_id = self._context_id(context)
        if self.context_authorizer is not None and context_id:
            await self.context_authorizer.authorize(context_id, subject)
        return subject

    def _config(self, context: RequestContext, subject: str) -> RunnableConfig:
        config: RunnableConfig = (
            dict(self.config_from_context(context) or {})
            if self.config_from_context
            else {}
        )
        configurable = dict(config.get("configurable") or {})
        context_id = self._context_id(context)
        # Keyed on the conversation alone. The task is not in the key, or a
        # follow-up turn would start on an empty checkpoint; the subject is not
        # in the key, or a rotated credential would fork the conversation.
        configurable["thread_id"] = thread_id(context_id)
        configurable.setdefault("a2a_context_id", context_id)
        configurable.setdefault("a2a_task_id", context.task_id or "")
        configurable.setdefault("a2a_subject", subject)
        configurable.setdefault(
            "a2a_context_namespace", context_namespace(subject, context_id)
        )
        config["configurable"] = configurable
        return config

    def _check_identity(self, context: RequestContext) -> None:
        """Refuse a message whose task and conversation do not agree.

        Binding the context is necessary and not sufficient. The SDK loads the
        task and then passes `task=None` into its context builder, so its own
        `contextId`/`taskId` agreement check never runs and `RequestContext`
        carries whatever the caller sent. Without this, a caller can present a
        victim's `taskId` alongside a context it legitimately owns, pass the
        ownership check, and advance someone else's task.

        Terminal tasks are refused here too. The SDK checks that once when the
        active task starts, but a message queued before the task closed still
        reaches this point.
        """
        task = context.current_task
        if task is None:
            return

        if task.status.state in TERMINAL_STATES:
            raise InvalidParamsError(
                message=(
                    f"Task {task.id} is {TaskState.Name(task.status.state)} and cannot "
                    "take another message. Start a new task."
                )
            )

        requested = context.call_context.state.get(REQUESTED_CONTEXT_ID_KEY)
        if requested and task.context_id and requested != task.context_id:
            # Same answer as a context that does not exist: telling a caller
            # that this task lives in some other conversation would confirm it
            # exists.
            raise TaskNotFoundError

    async def _parked_tasks(
        self, context: RequestContext, context_id: str
    ) -> list[Task]:
        """Tasks in this conversation that are waiting for an answer.

        Only pauses are read. A turn arriving while another *runs* waits for it
        rather than being refused, and where it waits depends on whether this
        process can see the run: on the conversation lock when the run is here,
        on the store when it is not. Neither wait belongs in this list, which
        exists to answer a different question — whether an unanswered question
        is holding the conversation.
        """
        return await self._tasks_in(context, context_id, PAUSED_STATES)

    async def _running_tasks(
        self, context: RequestContext, context_id: str
    ) -> list[Task]:
        """Tasks in this conversation another process may be working on now."""
        return await self._tasks_in(context, context_id, RUNNING_STATES)

    async def _tasks_in(
        self, context: RequestContext, context_id: str, states: Sequence[Any]
    ) -> list[Task]:
        if self.task_store is None or not context_id:
            return []
        found: list[Task] = []
        for state in states:
            page = await self.task_store.list(
                ListTasksRequest(context_id=context_id, status=state),
                context.call_context,
            )
            found.extend(task for task in page.tasks if task.id != context.task_id)
        return found

    async def _claim(self, context: RequestContext, context_id: str) -> Task | None:
        """Take the conversation, where the conversation is not this process's.

        In one process the conversation lock is the whole answer and there is
        nothing to write. Beside a second replica the lock excludes nothing, so
        the claim is a conditional write of this task's own record at `WORKING`:
        a store that supports `multi_replica=True` refuses it while another
        running task holds the same context (see `ConditionalTaskStore`), and
        the arriving turn retries until it does not.

        The release is the ordinary write that ends a turn — completed, failed
        or parked at a question — so nothing has to remember to let go.

        Bounded by the same timeout that bounds a run, because an unbounded wait
        here is a request that never answers. Nothing clears a claim whose owner
        died: that is the stated cost of having no lease, and the deployment
        brings whatever ends the abandoned task.

        Returns:
            The record as it stood before the claim, for `_release` to put back
            if the turn never runs. `None` when nothing was claimed.
        """
        if not self.multi_replica or self.task_store is None:
            return None
        if not isinstance(self.task_store, ConditionalTaskStore):
            # Refused at construction, so this is unreachable from
            # `create_a2a_app`. An executor composed by hand can still get here.
            logger.warning("multi_replica with a store that cannot claim; not claiming")
            return None

        deadline = (
            None if self.run_timeout is None else time.monotonic() + self.run_timeout
        )
        call_context = context.call_context
        while True:
            stored, version = await self.task_store.get_versioned(
                context.task_id, call_context
            )
            claimed = Task()
            if stored is not None:
                claimed.CopyFrom(stored)
            else:
                claimed.id = context.task_id
                claimed.context_id = context_id
            claimed.status.state = TaskState.TASK_STATE_WORKING
            if await self.task_store.save_if_unchanged(claimed, version, call_context):
                return stored
            running = await self._running_tasks(context, context_id)
            if deadline is not None and time.monotonic() >= deadline:
                raise _busy(running[0] if running else claimed)
            await asyncio.sleep(CLAIM_POLL_INTERVAL)

    async def _release(self, context: RequestContext, stored: Task | None) -> None:
        """Put a claimed record back, for a turn that was refused rather than run."""
        if stored is None or not isinstance(self.task_store, ConditionalTaskStore):
            return
        _, version = await self.task_store.get_versioned(
            stored.id, context.call_context
        )
        if not await self.task_store.save_if_unchanged(
            stored, version, context.call_context
        ):
            logger.warning("could not release the claim on task %s", stored.id)

    async def _any_task_exists(self, context: RequestContext, context_id: str) -> bool:
        """Whether this conversation has any task record at all."""
        if self.task_store is None or not context_id:
            return False
        page = await self.task_store.list(
            ListTasksRequest(context_id=context_id), context.call_context
        )
        return bool(page.tasks)

    def _asked_before(self, task: Task) -> bool:
        """Was this question published longer ago than the deadline allows?

        Measured from the moment the task entered `input-required` — not from
        when it was created and not from when its run started. Nothing fires
        when the duration passes: this is only ever evaluated by a second task
        arriving at the same conversation, which is what makes an approval that
        legitimately waits days survive while a conversation nobody else wants
        stays parked and resumable for ever.
        """
        if self.pause_deadline is None:
            return False
        if task.status.state not in PAUSED_STATES or not task.status.HasField(
            "timestamp"
        ):
            return False
        asked_at = task.status.timestamp.ToDatetime(tzinfo=UTC)
        return (datetime.now(UTC) - asked_at).total_seconds() > (self.pause_deadline)

    async def _displace(self, task: Task, context: RequestContext) -> None:
        """End a pause an arriving turn is waiting behind, and take the conversation.

        Written straight to the task store: the request that published this
        question finished long ago and there is no event queue left to publish
        on. Conditional where the store supports it, so two arrivals cannot both
        believe they displaced the same pause.
        """
        displaced = Task()
        displaced.CopyFrom(task)
        displaced.status.state = TaskState.TASK_STATE_FAILED
        displaced.status.message.CopyFrom(
            Message(
                message_id=uuid.uuid4().hex,
                context_id=task.context_id,
                task_id=task.id,
                role=Role.ROLE_AGENT,
                parts=[text_part(self._displaced_text())],
            )
        )
        logger.info(
            "displacing task %s: asked more than %ss ago", task.id, self.pause_deadline
        )
        if self.task_store is None:
            return
        if isinstance(self.task_store, ConditionalTaskStore):
            _, version = await self.task_store.get_versioned(
                task.id, context.call_context
            )
            if not await self.task_store.save_if_unchanged(
                displaced, version, context.call_context
            ):
                raise _busy(task)
            return
        await self.task_store.save(displaced, context.call_context)

    def _displaced_text(self) -> str:
        return (
            "This question was asked more than "
            f"{self.pause_deadline:g}s ago and another task needed the "
            "conversation, so it was ended without an answer."
        )

    async def _blocking_task(
        self, context: RequestContext, context_id: str
    ) -> Task | None:
        """The task, if any, that stops a new turn starting in this conversation.

        Three outcomes, and the deadline picks between the last two: nothing
        parked, so proceed; a question asked more recently than the deadline, so
        refuse and name it; a question older than that, so displace it and
        proceed.
        """
        for task in await self._parked_tasks(context, context_id):
            if self._asked_before(task):
                await self._displace(task, context)
                continue
            return task
        return None

    @staticmethod
    def _is_resume(context: RequestContext) -> bool:
        """Read from the task, which is the only place that survives the run."""
        task = context.current_task
        return bool(
            task
            and task.status.state
            in (TaskState.TASK_STATE_INPUT_REQUIRED, TaskState.TASK_STATE_AUTH_REQUIRED)
        )

    async def _pending_interrupts(
        self, config: RunnableConfig
    ) -> tuple[list[PendingInterrupt], dict[str, Any]]:
        """What the stored state says is unanswered, and where it is stored.

        Read from `StateSnapshot.interrupts`, and read deliberately: neither
        source the graph offers is exact, and they fail in opposite directions.

        `tasks` whose `result is None` misses a live pause. A node that pauses,
        is answered, and pauses again keeps the same interrupt id and carries a
        result from the answer already given — so that filter calls a waiting
        graph empty, and this server would fail a task that is perfectly
        resumable.

        `interrupts` over-reports the other way: after one of two parallel pauses
        is answered it still lists both. That is the safer error here, because
        this set only decides which ids a caller may answer and whether anything
        is outstanding at all. An answer to a question already answered is
        dropped by the graph; a refusal to answer a live one ends the task. What
        the caller is *asked* never comes from here — see `_run`, which reports
        the pauses the run itself raised.
        """
        snapshot = await self.graph.aget_state(config)
        if snapshot is None:
            return [], {}
        nodes = _nodes_of(snapshot)
        pending = [
            PendingInterrupt(id=item.id, value=item.value, node=nodes.get(item.id))
            for item in getattr(snapshot, "interrupts", ()) or ()
        ]
        return pending, _checkpoint_of(snapshot)

    async def _pending_interrupts_or_none(self, config: RunnableConfig) -> bool:
        """Whether the graph is parked on a question, tolerating no checkpointer."""
        try:
            pending, _checkpoint = await self._pending_interrupts(config)
        except Exception:
            logger.debug("could not read pending interrupts from state", exc_info=True)
            return False
        return bool(pending)

    async def _describe_pause(
        self, config: RunnableConfig, raised: list[PendingInterrupt]
    ) -> tuple[list[PendingInterrupt], dict[str, Any]]:
        """Name the nodes of the pauses this run raised, and locate them.

        The set comes from the run, which is the only exact answer: it reports
        the pause a re-interrupting node just raised and it drops the one a
        partial resume just answered. The stored state contributes the node
        names and the checkpoint to resume at, and contributes nothing if the
        graph has no checkpointer.
        """
        try:
            snapshot = await self.graph.aget_state(config)
        except Exception:
            logger.debug("could not read the paused checkpoint", exc_info=True)
            return raised, {}
        if snapshot is None:
            return raised, {}
        nodes = _nodes_of(snapshot)
        described = [
            PendingInterrupt(id=item.id, value=item.value, node=nodes.get(item.id))
            for item in raised
        ]
        return described, _checkpoint_of(snapshot)

    # -- execution -----------------------------------------------------------

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Run one turn, holding the conversation's lock for its duration.

        Everything that can fail before the graph starts is funnelled through
        the same redaction as a node exception: a protocol error the caller can
        act on passes through, and anything else becomes `FAILED` with a
        correlation id. An internal error text reaching a peer is a leak whether
        it came from the graph or from the code around it.
        """
        context_id = self._context_id(context)
        updater = TaskUpdater(event_queue, context.task_id, context_id)
        async with self._conversation_lock(context_id):
            # The lock serialises this process. Where a second replica may hold
            # the thread, the store is the only thing that can say so, and
            # taking it is a write rather than a look.
            released = await self._claim(context, context_id)
            try:
                await self._execute(context, event_queue, updater, context_id)
            except (A2AError, asyncio.CancelledError):
                # A refused turn never ran, so the claim it took goes back.
                # `a2a-sdk` 1.1.2 also fails the task when the executor raises,
                # which releases it too — this makes the release ours rather
                # than borrowed, because a claim that leaks does not fail a
                # request, it freezes the conversation for everyone.
                await self._release(context, released)
                raise
            except Exception:
                await self._fail_opaquely(updater, context, "request setup failed")

    async def _execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
        updater: TaskUpdater,
        context_id: str,
    ) -> None:
        if self.require_extension:
            extension.require(context.requested_extensions, self.extension_uri)

        subject = await self._subject(context)
        self._check_identity(context)

        if context.current_task is None:
            # The first event of a new task must be the Task itself; the SDK
            # rejects a status update for a task it has never seen.
            task = Task(
                id=context.task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
            )
            if context.message is not None:
                task.history.append(context.message)
            await event_queue.enqueue_event(task)

        config = self._config(context, subject)
        parts = list(context.message.parts) if context.message else []

        if self._is_resume(context):
            # No deadline check here. The deadline releases a conversation
            # another task is waiting for; it is not a lifetime. A pause nothing
            # is waiting behind stays parked and answerable however long it
            # takes, which is the whole point of an approval that waits days.
            #
            # Read the pause from the checkpoint the task paused at, not from
            # the thread's head.
            pending, checkpoint = await self._pending_interrupts(
                _at_checkpoint(config, _paused_checkpoint(context.current_task))
            )
            if not pending:
                # The task says it is waiting; the graph says it is not. The
                # two stores have diverged — usually a durable task store
                # paired with state that did not survive. Re-running the turn
                # here would silently answer a question nobody asked.
                logger.error(
                    "task %s is at %s with no pending interrupt on its thread",
                    context.task_id,
                    TaskState.Name(context.current_task.status.state),
                )
                await updater.failed(
                    updater.new_agent_message([text_part(ORPHANED_INTERRUPT_MESSAGE)])
                )
                return
            try:
                resumes = interrupts.resume_map(parts, parts_text(parts, "\n"), pending)
            except AmbiguousResume as exc:
                raise InvalidParamsError(message=str(exc)) from exc
            if not resumes:
                raise InvalidParamsError(
                    message="This task is waiting for an answer, and the message carries none."
                )
            # Always the addressed form. A bare value is unaddressed with one
            # pause and an error with several — and a raw client dict is worse
            # than either, because LangGraph detects the map form by key shape
            # and a payload whose keys are 32-character hex would be read as a
            # map of interrupt ids.
            graph_input: Any = Command(resume=resumes)
            # The run is addressed at the checkpoint the pause happened on; the
            # conversation itself is still read at its head.
            run_config = _at_checkpoint(config, checkpoint)
        else:
            # A new turn in a conversation that a live task still holds. Running
            # it would destroy a pending question — the later answer would then
            # be accepted, change nothing, and return the previous state as
            # though it were fresh. Queueing it would park this caller behind a
            # question that may never be answered.
            blocking = await self._blocking_task(context, context_id)
            if blocking is not None:
                raise _busy(blocking)
            if await self._pending_interrupts_or_none(config) and not (
                await self._any_task_exists(context, context_id)
            ):
                # The graph is parked on a question and the conversation has no
                # task record at all to say whose it is. Proceeding would
                # destroy that question on the word of a store that has lost
                # its half of the pair.
                raise UnsupportedOperationError(
                    message=(
                        "This conversation is parked on a question whose task cannot "
                        "be read. It cannot take a new turn until the task store and "
                        "the graph agree again."
                    )
                )
            graph_input = self.state.to_graph_input(parts)
            run_config = config

        # WORKING is emitted because the run started, not because a token
        # arrived: an agent that thinks for a minute before saying anything
        # must not leave the connection silent.
        await updater.start_work()
        await self._run(updater, graph_input, run_config, config, context)

    async def _run(
        self,
        updater: TaskUpdater,
        graph_input: Any,
        run_config: RunnableConfig,
        head_config: RunnableConfig,
        context: RequestContext,
    ) -> None:
        """Drive one run, then report what it produced.

        Two configs, and the difference matters. The run is addressed at the
        checkpoint a resume must continue from; the state read afterwards is
        addressed at the conversation's head, because that is where a *new*
        pause now lives. Reading the pinned checkpoint back would find the
        question this turn just answered, record its stale coordinates on the
        task, and strand the next answer.
        """
        # A mapping reads the final state, so there is nothing to stream: the
        # answer does not exist until the run is over. Supplying one is the
        # switch, which is why `stream_tokens` cannot contradict it.
        stream = _ArtifactStream(
            updater, enabled=self.stream_tokens and not self.state.maps_output
        )
        final_state: Any = {}
        pending: list[PendingInterrupt] = []
        spoken: list[tuple[Any, str]] = []

        async def drive() -> None:
            nonlocal final_state
            async for mode, chunk in self.graph.astream(
                graph_input,
                config=run_config,
                stream_mode=["updates", "messages", "values"],
            ):
                if mode == "values":
                    final_state = chunk
                elif mode == "updates" and isinstance(chunk, dict):
                    for node_name, update in chunk.items():
                        if node_name != "__interrupt__":
                            # What the model said this run, in order. The same
                            # material the token stream carries, from a source
                            # that is there whether or not anyone streamed —
                            # the final state is the whole conversation and
                            # cannot say which turns are this run's.
                            spoken.extend(_spoken_text(update, self.state.message_key))
                            continue
                        # Accumulate. Parallel branches report their interrupts
                        # in separate updates, and keeping only the last drops
                        # every question but one.
                        for item in update or ():
                            pending.append(
                                PendingInterrupt(id=item.id, value=item.value)
                            )
                elif mode == "messages":
                    await stream.token(chunk)
            await stream.flush()

        try:
            async with _deadline(self.run_timeout):
                await drive()
        except TimeoutError:
            # A task must not sit at WORKING for the life of the process. The
            # deadline is the agent's, so the caller is told plainly rather than
            # given a correlation id for a failure that is not a defect.
            logger.warning(
                "run for task %s exceeded %ss", context.task_id, self.run_timeout
            )
            await updater.failed(
                updater.new_agent_message(
                    [text_part(f"The agent timed out after {self.run_timeout:g}s.")]
                )
            )
            return
        except asyncio.CancelledError:
            raise
        except TaskRejected as rejection:
            await updater.reject(
                updater.new_agent_message([text_part(str(rejection) or "Rejected")])
            )
            return
        except Exception:
            await self._fail_opaquely(updater, context, "graph execution failed")
            return

        if pending:
            if getattr(self.graph, "checkpointer", None) is None:
                # `input-required` means "ask me again", and there is nothing to
                # ask again: with no checkpointer the run is not suspended
                # anywhere, so an answer would arrive at a graph that never
                # paused. Serving such a graph is fine; reporting a pause it
                # cannot honour is not.
                await self._fail_unresumable_pause(updater, context)
                return
            # The checkpoint is the authority on what is outstanding, and it
            # knows which node each pause came from.
            described, checkpoint = await self._describe_pause(head_config, pending)
            await self._emit_pause(updater, described, checkpoint)
            return

        await self._emit_result(updater, final_state, stream, spoken)

    async def _fail_opaquely(
        self, updater: TaskUpdater, context: RequestContext, what: str
    ) -> None:
        """`FAILED` with a correlation id, and nothing else.

        The peer gets an id it can quote in a support request. It does not get
        the exception type, the message, or anything else this process knows —
        and that holds for failures raised before the run as well as inside it.
        """
        reference = uuid.uuid4().hex[:12]
        logger.exception("%s for task %s [ref %s]", what, context.task_id, reference)
        try:
            await updater.failed(
                updater.new_agent_message(
                    [
                        text_part(
                            f"The agent failed to complete this task. Reference: {reference}"
                        )
                    ]
                )
            )
        except Exception:
            logger.debug("could not report failure for task %s", context.task_id)

    async def _fail_unresumable_pause(
        self, updater: TaskUpdater, context: RequestContext
    ) -> None:
        """A pause from a graph that cannot suspend: fail, and say why here."""
        reference = uuid.uuid4().hex[:12]
        logger.error(
            "task %s interrupted on a graph with no checkpointer [ref %s]",
            context.task_id,
            reference,
        )
        await updater.failed(
            updater.new_agent_message(
                [
                    text_part(
                        "This agent asked a question it cannot be answered on, so "
                        f"the task was ended rather than left waiting. Reference: "
                        f"{reference}"
                    )
                ]
            )
        )

    async def _emit_pause(
        self,
        updater: TaskUpdater,
        pending: Sequence[PendingInterrupt],
        checkpoint: dict[str, Any] | None = None,
    ) -> None:
        """`interrupt()` becomes `input-required`, or `auth-required`.

        Both renderings ride the status message: the id-carrying `data` parts
        for a caller that can answer structurally, and the prompt as text for
        one that cannot. They are not artifacts — a pause is not a result, and a
        caller reading `artifacts` should not find a question there.
        """
        parts: list[Part] = [
            interrupts.interrupt_part(item, encoder=self.interrupt_encoder)
            for item in pending
        ]
        prompt = interrupts.prompt_text(pending)
        if prompt:
            parts.insert(0, text_part(prompt))
        message = updater.new_agent_message(parts)

        # The checkpoint travels on the task, so the answer resumes where the
        # pause happened rather than wherever the thread has since reached.
        metadata = {CHECKPOINT_METADATA_KEY: checkpoint} if checkpoint else None
        state = (
            TaskState.TASK_STATE_AUTH_REQUIRED
            if any(item.is_credential_request for item in pending)
            else TaskState.TASK_STATE_INPUT_REQUIRED
        )
        await updater.update_status(state, message=message, metadata=metadata)

    async def _emit_result(
        self,
        updater: TaskUpdater,
        state: Any,
        stream: _ArtifactStream,
        spoken: Sequence[tuple[Any, str]],
    ) -> None:
        """Close the response artifact, in whichever of its two shapes applies.

        **Transcript** — no output mapping. The artifact is what the model said,
        so a stream that opened is already the artifact and is closed where it
        stands: nothing is sent twice, and what a client reconstructed from the
        stream is what `GetTask` holds. Where nothing streamed, the same
        material is written at the end from what the run's nodes produced, one
        part per model turn, so the bytes do not depend on who was watching.

        **Mapping** — `output_text` or `output_parts`. Nothing streamed, so the
        artifact is written once, from the final state.

        `output_data` is appended in either shape, which in the streamed case
        means one more chunk on an artifact that is already open.
        """
        data = self.state.to_data(state)
        tail = [data_part(encode_payload(data))] if data is not None else []

        if self.state.maps_output:
            parts = [*self.state.to_parts(state), *tail]
        elif stream.opened:
            # Appending, not replacing: the transcript is what was streamed.
            await updater.add_artifact(
                tail,
                artifact_id=RESPONSE_ARTIFACT,
                name="response",
                append=True,
                last_chunk=True,
            )
            await updater.complete()
            return
        else:
            said = _transcript(stream.turns, spoken)
            parts = [*(text_part(text) for text in said), *tail]

        if not parts:
            # A completed task with no artifact reads as a protocol defect at
            # the far end. If the graph produced nothing usable, say so.
            parts = [text_part(EMPTY_RESULT_TEXT)]
        await updater.add_artifact(
            parts,
            artifact_id=RESPONSE_ARTIFACT,
            name="response",
            append=False,
            last_chunk=True,
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Report `CANCELED`.

        The run itself is already being cancelled by the SDK, which cancels the
        producer task before calling this. What matters here is that the
        terminal state is asserted rather than inferred: a cooperative
        LangGraph cancellation leaves the run `interrupted`, and a task that
        reports `input-required` forever is not a cancelled task.
        """
        updater = TaskUpdater(event_queue, context.task_id, self._context_id(context))
        await updater.cancel()


@asynccontextmanager
async def _deadline(seconds: float | None) -> AsyncIterator[None]:
    """Bound the body by `seconds`, without moving it to another task.

    `asyncio.timeout` does exactly this and arrived in 3.11; this package
    supports 3.10, so the older path is spelled out rather than reached for with
    `asyncio.wait_for`. That would be the obvious substitute and it is the wrong
    one: before 3.12 it runs the coroutine in a **new task**, and a graph run
    moved into a fresh task loses the context LangGraph resolves its config
    from — `interrupt()` then fails with "Called get_config outside of a
    runnable context", turning every pause into a failed task.

    So the fallback cancels the *current* task on expiry and converts that one
    cancellation into a timeout, leaving the run exactly where it was.
    """
    if seconds is None:
        yield
        return

    native = getattr(asyncio, "timeout", None)
    if native is not None:
        async with native(seconds):
            yield
        return

    task = asyncio.current_task()
    expired = False

    def fire() -> None:
        nonlocal expired
        expired = True
        if task is not None:
            task.cancel()

    handle = asyncio.get_running_loop().call_later(seconds, fire)
    try:
        yield
    except asyncio.CancelledError:
        if expired:
            raise TimeoutError from None
        raise
    finally:
        handle.cancel()


def _busy(blocking: Task) -> A2AError:
    """The refusal, shaped so a caller can act on it.

    Not `InvalidParamsError`: the caller's parameters were fine, the server was
    busy. A2A has no "busy" error and `UnsupportedOperationError` is the honest
    fit; the blocking task's id turns a dead end into a handshake, because the
    caller can fetch it, see `input-required`, and answer or cancel.
    """
    waiting = blocking.status.state in PAUSED_STATES
    what = "is waiting for an answer" if waiting else "is still running"
    return UnsupportedOperationError(
        message=(
            f"This conversation already has a task in progress: {blocking.id} {what}. "
            "Answer or cancel it and try again. Work that does not need this "
            "conversation's memory belongs in a conversation of its own."
        ),
        data={
            "blocking_task_id": blocking.id,
            "blocking_task_state": TaskState.Name(blocking.status.state),
        },
    )


def _nodes_of(snapshot: Any) -> dict[str, str]:
    """Which node raised each interrupt the stored state knows about."""
    return {
        item.id: task.name
        for task in getattr(snapshot, "tasks", ())
        for item in getattr(task, "interrupts", ())
    }


def _checkpoint_of(snapshot: Any) -> dict[str, Any]:
    """The checkpoint coordinates of a state snapshot.

    `checkpoint_ns` is carried deliberately: addressing a checkpoint without it
    raises inside the checkpointer rather than falling back to the default.
    """
    configurable = (getattr(snapshot, "config", None) or {}).get("configurable") or {}
    checkpoint_id = configurable.get("checkpoint_id")
    if not checkpoint_id:
        return {}
    return {
        "checkpoint_id": checkpoint_id,
        "checkpoint_ns": configurable.get("checkpoint_ns", ""),
    }


def _at_checkpoint(
    config: RunnableConfig, checkpoint: dict[str, Any]
) -> RunnableConfig:
    """`config` addressed at a specific checkpoint, when one is known."""
    if not checkpoint:
        return config
    pinned: RunnableConfig = dict(config)
    pinned["configurable"] = {**(config.get("configurable") or {}), **checkpoint}
    return pinned


def _paused_checkpoint(task: Task) -> dict[str, Any]:
    """The checkpoint a paused task recorded, if it recorded one."""
    if not task.HasField("metadata"):
        return {}
    metadata = json_format.MessageToDict(task.metadata)
    checkpoint = metadata.get(CHECKPOINT_METADATA_KEY)
    if isinstance(checkpoint, dict) and checkpoint.get("checkpoint_id"):
        return {
            "checkpoint_id": checkpoint["checkpoint_id"],
            "checkpoint_ns": checkpoint.get("checkpoint_ns", ""),
        }
    return {}


def _spoken_text(update: Any, message_key: str) -> list[tuple[Any, str]]:
    """The visible model turns one node update added, as `(id, text)`.

    The second half of the transcript's material. A node that calls a model
    streams tokens; a node that simply writes an assistant message does not,
    and both are things the agent said. Ids come from the same place in both
    cases, so a turn that streamed and was then written to state is one turn.
    """
    if not isinstance(update, dict):
        return []
    said = []
    for message in update.get(message_key) or ():
        if getattr(message, "type", None) != "ai":
            continue
        text = _content_text(getattr(message, "content", ""))
        if text:
            said.append((getattr(message, "id", None) or text, text))
    return said


def _transcript(
    streamed: dict[Any, str], spoken: Sequence[tuple[Any, str]]
) -> list[str]:
    """What the agent said this run, once each, in the order it said it."""
    turns = dict(streamed)
    for key, text in spoken:
        turns.setdefault(key, text)
    return [text for text in turns.values() if text]


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


class _ArtifactStream:
    """Buffers token deltas into artifact chunks, and remembers every turn.

    Streaming one part per token is quadratic against the task store: every
    delta rewrites the whole task, and a long answer arrives as thousands of
    single-token parts. Buffering by size and by age keeps the wire cheap
    without making a slow model look dead.

    Tokens are accumulated whether or not they are emitted, because the
    transcript has to be the same material either way — a run that streamed and
    the same run with streaming off must not produce different artifacts.
    """

    def __init__(self, updater: TaskUpdater, *, enabled: bool) -> None:
        self._updater = updater
        self._enabled = enabled
        self._buffer: list[str] = []
        self._size = 0
        self._deadline: float | None = None
        self._turn: Any = None
        self.opened = False
        self.turns: dict[Any, str] = {}
        """Text per model turn, in the order the turns began."""

    async def token(self, chunk: Any) -> None:
        text = _token_text(chunk)
        if not text:
            return
        key = _token_turn(chunk)
        self.turns[key] = self.turns.get(key, "") + text
        if not self._enabled:
            return
        # One part never spans two model turns. A graph that speaks twice would
        # otherwise run them together with no separator, and the transcript
        # written at the end for an unstreamed run could not match it.
        if self._buffer and key != self._turn:
            await self.flush()
        self._turn = key
        self._buffer.append(text)
        self._size += len(text)
        now = asyncio.get_running_loop().time()
        if self._deadline is None:
            self._deadline = now + STREAM_FLUSH_SECONDS
        if self._size >= STREAM_FLUSH_CHARS or now >= self._deadline:
            await self.flush()

    async def flush(self) -> None:
        if not self._buffer:
            return
        text = "".join(self._buffer)
        self._buffer.clear()
        self._size = 0
        self._deadline = None
        # The first chunk opens the artifact; the SDK rejects an append to an
        # artifact it has not seen created.
        await self._updater.add_artifact(
            [text_part(text)],
            artifact_id=RESPONSE_ARTIFACT,
            name="response",
            append=self.opened,
            last_chunk=False,
        )
        self.opened = True


def _token_turn(chunk: Any) -> Any:
    """Which model turn a token belongs to, as far as the stream can say."""
    token = chunk[0] if isinstance(chunk, tuple | list) and chunk else None
    return getattr(token, "id", None)


def _token_text(chunk: Any) -> str:
    if not isinstance(chunk, tuple | list) or not chunk:
        return ""
    token = chunk[0]
    if getattr(token, "type", None) not in (None, "ai", "AIMessageChunk", "assistant"):
        return ""
    content = getattr(token, "content", "") or ""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )
