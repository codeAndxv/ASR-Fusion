
from fastapi import (
    APIRouter,
    WebSocket,
)
from openai import AsyncOpenAI

router = APIRouter(tags=["realtime"])
event_router = EventRouter()

@router.websocket("/v1/realtime")
async def realtime(
    ws: WebSocket,
    model: str,
    config: ConfigDependency,
    transcription_client: TranscriptionClientDependency,
) -> None:
    await ws.accept()
    logger.info("Accepted websocket connection")

    completion_client = AsyncOpenAI(
        base_url=f"http://{config.host}:{config.port}/v1",
        api_key=config.api_key.get_secret_value() if config.api_key else "cant-be-empty",
        max_retries=0,
    ).chat.completions
    ctx = SessionContext(
        transcription_client=transcription_client,
        completion_client=completion_client,
        session=create_session_object_configuration(model),
    )
    message_manager = WsServerMessageManager(ctx.pubsub)
    async with asyncio.TaskGroup() as tg:
        event_listener_task = tg.create_task(event_listener(ctx), name="event_listener")
        async with asyncio.timeout(OPENAI_REALTIME_SESSION_DURATION_SECONDS):
            mm_task = asyncio.create_task(message_manager.run(ws))
            # HACK: a tiny delay to ensure the message_manager.run() task is started. Otherwise, the `SessionCreatedEvent` will not be sent, as it's published before the `sender` task subscribes to the pubsub.
            await asyncio.sleep(0.001)
            ctx.pubsub.publish_nowait(SessionCreatedEvent(session=ctx.session))
            await mm_task
        event_listener_task.cancel()

    logger.info(f"Finished handling '{ctx.session.id}' session")
