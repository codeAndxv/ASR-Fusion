import asyncio

from collections.abc import Generator, Iterable
from typing import Annotated, Literal
from fastapi import (
    APIRouter,
    Form,
    HTTPException,
    Request,
    Response,
)
from asr_fusion.dependencies import AudioFileDependency, ConfigDependency
router = APIRouter(tags=["automatic-speech-recognition"])

@router.post("/v1/audio/translations")
def translate_file(
    config: ConfigDependency,
    model_manager: WhisperModelManagerDependency,
    audio: AudioFileDependency,
    model: Annotated[ModelId, Form()],
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[ResponseFormat, Form()] = DEFAULT_RESPONSE_FORMAT,
    temperature: Annotated[float, Form()] = 0.0,
    stream: Annotated[bool, Form()] = False,
    vad_filter: Annotated[bool | None, Form()] = None,
) -> Response | StreamingResponse:
    # Use config default if vad_filter not explicitly provided
    effective_vad_filter = vad_filter if vad_filter is not None else config._unstable_vad_filter  # noqa: SLF001

    with model_manager.load_model(model) as whisper:
        whisper_model = BatchedInferencePipeline(model=whisper) if config.whisper.use_batched_mode else whisper
        segments, transcription_info = whisper_model.transcribe(
            audio,
            task="translate",
            initial_prompt=prompt,
            temperature=temperature,
            vad_filter=effective_vad_filter,
        )
        segments = TranscriptionSegment.from_faster_whisper_segments(segments)

        if stream:
            return segments_to_streaming_response(segments, transcription_info, response_format)
        else:
            return segments_to_response(segments, transcription_info, response_format)

# https://platform.openai.com/docs/api-reference/audio/createTranscription
# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L8915
@router.post(
    "/v1/audio/transcriptions",
    response_model=str | CreateTranscriptionResponseJson | CreateTranscriptionResponseVerboseJson,
)
def transcribe_file(
    config: ConfigDependency,
    model_manager: WhisperModelManagerDependency,
    request: Request,
    audio: AudioFileDependency,
    model: Annotated[ModelId, Form()],
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[ResponseFormat, Form()] = DEFAULT_RESPONSE_FORMAT,
    temperature: Annotated[float, Form()] = 0.0,
    timestamp_granularities: Annotated[
        TimestampGranularities,
        # WARN: `alias` doesn't actually work.
        Form(alias="timestamp_granularities[]"),
    ] = ["segment"],
    stream: Annotated[bool, Form()] = False,
    hotwords: Annotated[str | None, Form()] = None,
    vad_filter: Annotated[bool | None, Form()] = None,
) -> Response | StreamingResponse:
    # Use config default if vad_filter not explicitly provided
    effective_vad_filter = vad_filter if vad_filter is not None else config._unstable_vad_filter  # noqa: SLF001

    timestamp_granularities = asyncio.run(get_timestamp_granularities(request))
    if timestamp_granularities != DEFAULT_TIMESTAMP_GRANULARITIES and response_format != "verbose_json":
        logger.warning(
            "It only makes sense to provide `timestamp_granularities[]` when `response_format` is set to `verbose_json`. See https://platform.openai.com/docs/api-reference/audio/createTranscription#audio-createtranscription-timestamp_granularities."
        )

    model_repo_path = get_model_repo_path(model)
    if model_repo_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' is not installed locally. You can download the model using `POST /v1/models`",
        )
    cached_repo_info = _scan_cached_repo(model_repo_path)
    model_card_data = get_model_card_data_from_cached_repo_info(cached_repo_info)
    if model_card_data is None:
        raise HTTPException(
            status_code=500,
            detail=MODEL_CARD_DOESNT_EXISTS_ERROR_MESSAGE.format(model_id=model),
        )
    if whisper_utils.hf_model_filter.passes_filter(model_card_data):
        with model_manager.load_model(model) as whisper:
            whisper_model = BatchedInferencePipeline(model=whisper) if config.whisper.use_batched_mode else whisper
            segments, transcription_info = whisper_model.transcribe(
                audio,
                task="transcribe",
                language=language,
                initial_prompt=prompt,
                word_timestamps="word" in timestamp_granularities,
                temperature=temperature,
                vad_filter=effective_vad_filter,
                hotwords=hotwords,
            )
            segments = TranscriptionSegment.from_faster_whisper_segments(segments)

            if stream:
                return segments_to_streaming_response(segments, transcription_info, response_format)
            else:
                return segments_to_response(segments, transcription_info, response_format)
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' is not supported. If you think this is a mistake, please open an issue.",
        )

def segments_to_streaming_response(segments, transcription_info, response_format):
    def segment_responses() -> Generator[str, None, None]:
        for i, segment in enumerate(segments):
            if response_format == "text":
                data = segment.text
            elif response_format == "json":
                data = CreateTranscriptionResponseJson.from_segments([segment]).model_dump_json()
            elif response_format == "verbose_json":
                data = CreateTranscriptionResponseVerboseJson.from_segment(
                    segment, transcription_info
                ).model_dump_json()
            elif response_format == "vtt":
                data = segments_to_vtt(segment, i)
            elif response_format == "srt":
                data = segments_to_srt(segment, i)
            yield format_as_sse(data)

    return StreamingResponse(segment_responses(), media_type="text/event-stream")

