"""Push notifications: where a callback may point, and what it carries.

A push config is a URL supplied by a caller and stored by the server, which then
makes requests to it. That is a server-side request forgery primitive with the
task's content as the payload unless something checks the destination, and
`a2a-sdk` 1.1.2's `BasePushNotificationSender` checks nothing: it POSTs wherever
it was told, following whatever redirects the client allows.

So enabling push here comes with a destination policy — https, no loopback, no
link-local, no private ranges unless the deployment opts in — applied when the
config is registered rather than when it fires, so a caller learns immediately.

The sender also carries the credentials the caller registered. The SDK's puts
the config token in `X-A2A-Notification-Token` and never reads the config's
`authentication` field at all, so a receiver checking `Authorization` drops every
delivery while the sender logs a success on any 2xx.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

import httpx
from google.protobuf.json_format import MessageToDict

from a2a.server.context import ServerCallContext
from a2a.server.tasks import (
    BasePushNotificationSender,
    PushNotificationConfigStore,
)
from a2a.types.a2a_pb2 import TaskPushNotificationConfig
from a2a.utils.errors import InvalidParamsError
from a2a.utils.proto_utils import to_stream_response

logger = logging.getLogger(__name__)

NOTIFICATION_TOKEN_HEADER = "X-A2A-Notification-Token"
"""What the SDK sends the config's `token` as. Kept, for callers that expect it."""


class WebhookRefused(InvalidParamsError):
    """The registered callback URL is not one this server will call."""


def check_webhook_url(url: str, *, allow_private: bool = False) -> None:
    """Refuse a callback this server should not be making requests to.

    Args:
        url: the destination a caller registered.
        allow_private: the deployment vouches for the network its receivers are
            on. Loopback and private addresses are then permitted, and so is
            plain http — a receiver inside your own network is exactly where TLS
            may not be, and the two questions have one answer. Never set this on
            a server reachable by callers it does not control.

    Raises:
        WebhookRefused: the URL is not a destination a caller should be able to
            point this server at.
    """
    parsed = urlparse(url)
    permitted = ("https", "http") if allow_private else ("https",)
    if parsed.scheme not in permitted:
        raise WebhookRefused(
            message=(
                "A push notification URL must be https, not "
                f"{parsed.scheme or 'relative'}."
            )
        )
    if not parsed.hostname:
        raise WebhookRefused(message="A push notification URL must name a host.")
    if allow_private:
        return

    for address in _resolve(parsed.hostname):
        if (
            address.is_loopback
            or address.is_private
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
        ):
            raise WebhookRefused(
                message=(
                    f"This agent will not send notifications to {parsed.hostname}: "
                    "loopback, link-local and private addresses are refused."
                )
            )


def _resolve(hostname: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Every address the hostname answers with, so none of them is a bypass.

    A name that resolves to a public address today and a private one on the next
    lookup is the classic way past a check like this. Resolving here narrows the
    window; it does not close it, and closing it needs the connection itself to
    be pinned — which is the deployment's HTTP client, not ours.
    """
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        raise WebhookRefused(
            message=f"This agent cannot resolve {hostname}, so it will not call it."
        ) from None
    return [ipaddress.ip_address(info[4][0]) for info in infos]


class GuardedPushNotificationConfigStore(PushNotificationConfigStore):
    """A config store that refuses destinations before it records them.

    Wraps whichever store the deployment supplies — the SDK's in-memory one, its
    database one — so the policy applies wherever the configs live.
    """

    def __init__(
        self, inner: PushNotificationConfigStore, *, allow_private: bool = False
    ) -> None:
        self._inner = inner
        self._allow_private = allow_private

    async def set_info(
        self,
        task_id: str,
        notification_config: TaskPushNotificationConfig,
        context: ServerCallContext,
    ) -> None:
        check_webhook_url(notification_config.url, allow_private=self._allow_private)
        await self._inner.set_info(task_id, notification_config, context)

    async def get_info(
        self, task_id: str, context: ServerCallContext
    ) -> list[TaskPushNotificationConfig]:
        return await self._inner.get_info(task_id, context)

    async def get_info_for_dispatch(
        self, task_id: str
    ) -> list[TaskPushNotificationConfig]:
        # Delegated rather than inherited: the base implementation falls back to
        # a context-less read, which resolves to the empty owner partition and
        # silently drops every notification in a multi-owner deployment.
        return await self._inner.get_info_for_dispatch(task_id)

    async def delete_info(
        self,
        task_id: str,
        context: ServerCallContext,
        config_id: str | None = None,
    ) -> None:
        await self._inner.delete_info(task_id, context, config_id)


class AuthenticatingPushSender(BasePushNotificationSender):
    """Sends the credentials the caller registered, in the header it named.

    `PushNotificationConfig.authentication` carries a scheme and credentials —
    "Bearer" and a token, say — and a receiver that registered them checks
    `Authorization`. The SDK's sender never reads that field, so those
    deliveries are dropped at the far end while it records a success.
    """

    def _headers(self, config: TaskPushNotificationConfig) -> dict[str, str]:
        headers: dict[str, str] = {}
        if config.token:
            headers[NOTIFICATION_TOKEN_HEADER] = config.token
        authentication = config.authentication
        if authentication.credentials:
            scheme = authentication.scheme or "Bearer"
            headers["Authorization"] = f"{scheme} {authentication.credentials}"
        return headers

    async def _dispatch_notification(  # type: ignore[override]
        self,
        event: object,
        push_info: TaskPushNotificationConfig,
        task_id: str,
    ) -> bool:
        try:
            response = await self._client.post(
                push_info.url,
                json=MessageToDict(to_stream_response(event)),
                headers=self._headers(push_info),
            )
            response.raise_for_status()
        except Exception:
            logger.exception(
                "push notification for task %s to %s failed", task_id, push_info.url
            )
            return False
        return True


def default_push_sender(
    config_store: PushNotificationConfigStore,
    *,
    timeout: float = 30.0,
) -> AuthenticatingPushSender:
    """A sender that carries credentials and does not follow redirects.

    Redirects are refused because a validated destination that answers with a
    301 elsewhere is the same forgery the destination check exists to stop.
    """
    return AuthenticatingPushSender(
        httpx.AsyncClient(timeout=timeout, follow_redirects=False), config_store
    )
