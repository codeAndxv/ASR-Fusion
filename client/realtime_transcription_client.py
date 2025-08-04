import asyncio
import base64
import logging


from openai import AsyncOpenAI
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection
from openai.types.beta.realtime.session_update_event_param import (
    Session,
    SessionInputAudioTranscription,
    SessionTurnDetection,
)

import websockets

logger = logging.getLogger(__name__)

async def print_events(conn: AsyncRealtimeConnection, final_event: str | None = None) -> None:
    try:
        async for event in conn:
            if event.type == "response.audio.delta":
                size = len(base64.b64decode(event.delta))
                event.delta = f"base64 encoded audio of size {size} bytes"
            print(event.model_dump_json())
            if final_event is not None and event.type == final_event:
                break
    except websockets.exceptions.ConnectionClosedError:
        logger.info("Connection closed")

async def main() -> None:
    realtime_client = AsyncOpenAI(api_key="does-not-matter", websocket_base_url=WEBSOCKET_BASE_URL).beta.realtime
    async with asyncio.TaskGroup() as tg, realtime_client.connect(model=MODEL) as conn:
        tg.create_task(print_events(conn, final_event=None))
        await conn.session.update(
            session=Session(
                input_audio_transcription=SessionInputAudioTranscription(
                    model="Systran/faster-distil-whisper-small.en"  # controls the transcription model used
                ),
                turn_detection=SessionTurnDetection(
                    silence_duration_ms=1500,  # Shouldn't exceed 2500 due to how the current implementation works.
                    threshold=0.9,  # I've found this to be a good default value.
                    create_response=False,  # Ensures that the session is only used for audio transcription.
                ),
            )
        )
        await send_mic_audio(conn)


if __name__ == "__main__":
    asyncio.run(main())
