"""
Servidor de renderização de vídeo com FFmpeg
Roda na VPS com FastAPI + uvicorn
"""

import os
import uuid
import json
import asyncio
import subprocess
import tempfile
import traceback
import base64
import shutil
import sys
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Video Renderer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

RENDER_DIR = Path("/tmp/renders")
RENDER_DIR.mkdir(exist_ok=True)
FPS = 30

# Track jobs
jobs: dict[str, dict] = {}


@app.get("/")
async def health():
    return {"status": "ok", "engine": "ffmpeg", "tts": "edge-tts"}


@app.post("/tts")
async def tts(request: Request):
    body = await request.json()
    text = (body.get("text") or "").strip()
    voice = normalize_tts_voice(body.get("voice"))
    provider = (body.get("provider") or "auto").lower()

    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)
    if len(text) > 6000:
        return JSONResponse({"error": "text is too long for one TTS request"}, status_code=400)

    job_id = str(uuid.uuid4())
    work_dir = RENDER_DIR / f"tts_{job_id}"
    work_dir.mkdir(exist_ok=True)

    try:
        if provider in ("auto", "edge", "edge-tts", "local"):
            audio_path = work_dir / "speech.mp3"
            await synthesize_edge_tts(text, voice, audio_path)
            duration = await probe_duration(audio_path)
            return {
                "success": True,
                "providerUsed": "edge-tts",
                "voice": voice,
                "mimeType": "audio/mpeg",
                "audioBase64": base64.b64encode(audio_path.read_bytes()).decode("ascii"),
                "durationSeconds": duration,
            }

        return JSONResponse({"error": f"Unsupported TTS provider: {provider}"}, status_code=400)
    except Exception as e:
        print(f"TTS failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return job


@app.get("/render-status/{job_id}")
async def get_render_status(job_id: str):
    return await get_status(job_id)


@app.post("/render-video")
async def render_video(request: Request):
    body = await request.json()

    project_id = body.get("projectId")
    callback_url = body.get("callbackUrl")
    supabase_url = body.get("supabaseUrl")
    supabase_key = body.get("supabaseKey")
    scenes = body.get("scenes", [])
    settings = body.get("settings", {})

    if not project_id or not scenes:
        return JSONResponse({"error": "projectId and scenes are required"}, status_code=400)

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "jobId": job_id, "projectId": project_id}

    # Process in background
    asyncio.create_task(
        process_render(job_id, project_id, scenes, settings, callback_url, supabase_url, supabase_key)
    )

    return {"jobId": job_id, "status": "queued"}


async def download_file(url: str, dest: Path, attempts: int = 3) -> bool:
    """Download a file from URL to local path with short retries."""
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            if dest.exists():
                dest.unlink()
            async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                if not resp.content:
                    raise RuntimeError("empty response body")
                dest.write_bytes(resp.content)
                if dest.exists() and dest.stat().st_size > 0:
                    return True
                raise RuntimeError("downloaded file is empty")
        except Exception as e:
            last_error = e
            print(f"Failed to download {url} (attempt {attempt}/{attempts}): {e}")
            if dest.exists():
                try:
                    dest.unlink()
                except Exception:
                    pass
            if attempt < attempts:
                await asyncio.sleep(1.5 * attempt)
    print(f"Download permanently failed for {url}: {last_error}")
    return False


def require_download(path: Path | None, url: str | None, label: str, scene_order: int | str) -> Path:
    if path and path.exists() and path.stat().st_size > 0:
        return path
    raise RuntimeError(f"Scene {scene_order}: required {label} could not be downloaded from {url}")


def normalize_tts_voice(voice: str | None) -> str:
    voice = voice or "pt-BR-FranciscaNeural"
    aliases = {
        "default": "pt-BR-FranciscaNeural",
        "female": "pt-BR-FranciscaNeural",
        "feminina": "pt-BR-FranciscaNeural",
        "male": "pt-BR-AntonioNeural",
        "masculina": "pt-BR-AntonioNeural",
        "francisca": "pt-BR-FranciscaNeural",
        "antonio": "pt-BR-AntonioNeural",
    }
    return aliases.get(voice.lower(), voice)


async def synthesize_edge_tts(text: str, voice: str, output_path: Path):
    cmd = [
        sys.executable,
        "-m",
        "edge_tts",
        "--voice",
        voice,
        "--text",
        text,
        "--write-media",
        str(output_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode(errors="ignore")[-500:] or "edge-tts failed")
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("edge-tts did not create audio")


async def probe_duration(path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    proc = await asyncio.create_subprocess_exec(
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    if proc.returncode != 0:
        return None
    try:
        return round(float(stdout.decode().strip()), 2)
    except Exception:
        return None


async def process_render(
    job_id: str,
    project_id: str,
    scenes: list[dict],
    settings: dict,
    callback_url: str | None,
    supabase_url: str | None,
    supabase_key: str | None,
):
    """Main render pipeline: download assets, build video with FFmpeg, upload result."""
    work_dir = RENDER_DIR / job_id
    work_dir.mkdir(exist_ok=True)
    output_path = work_dir / "final.mp4"

    try:
        jobs[job_id]["status"] = "rendering"

        aspect = settings.get("aspectRatio", "16:9")
        caption_style = settings.get("captionStyle", "default")
        media_type = settings.get("mediaType", "ken_burns")
        bg_music_url = settings.get("bgMusicUrl")
        bg_music_volume = settings.get("bgMusicVolume", 0.15)
        narration_volume = settings.get("narrationVolume", 1.0)
        transition_style = settings.get("transitionStyle") or settings.get("transition") or "none"
        transition_duration = float(settings.get("transitionDuration") or 0)
        ken_burns_style = settings.get("kenBurnsStyle", "dynamic")
        color_grading = settings.get("colorGrading", "none")
        enable_vignette = bool(settings.get("enableVignette") or settings.get("vignette"))

        # Parse aspect ratio
        if aspect == "9:16":
            width, height = 1080, 1920
        elif aspect == "1:1":
            width, height = 1080, 1080
        else:
            width, height = 1920, 1080

        # Download all scene assets
        scene_clips = []
        scene_durations = []
        for i, scene in enumerate(sorted(scenes, key=lambda s: s.get("scene_order", 0))):
            scene_dir = work_dir / f"scene_{i}"
            scene_dir.mkdir(exist_ok=True)

            scene_order = scene.get("scene_order", i)
            duration = float(scene.get("scene_duration_seconds") or scene.get("estimated_duration") or 5)
            selected_media_url = scene.get("selected_media_url")
            selected_media_type = scene.get("selected_media_type")
            is_title_card = bool(scene.get("is_title_card"))
            image_url = selected_media_url if selected_media_type == "image" else scene.get("image_url")
            audio_url = scene.get("audio_url")
            video_url = selected_media_url if selected_media_type == "video" else scene.get("stock_video_url")
            text = scene.get("text_content", "")

            # Download image
            image_path = None
            if image_url:
                image_path = scene_dir / "image.jpg"
                await download_file(image_url, image_path)
                if not image_path.exists():
                    image_path = None
                if selected_media_type == "image":
                    image_path = require_download(image_path, image_url, "image", scene_order)

            # Download audio (narration)
            audio_path = None
            if audio_url:
                audio_path = scene_dir / "narration.mp3"
                await download_file(audio_url, audio_path)
                if not audio_path.exists():
                    audio_path = None
                audio_path = require_download(audio_path, audio_url, "audio", scene_order)

            # Download stock video
            stock_video_path = None
            if video_url:
                stock_video_path = scene_dir / "stock.mp4"
                await download_file(video_url, stock_video_path)
                if not stock_video_path.exists():
                    stock_video_path = None
                if selected_media_type == "video":
                    stock_video_path = require_download(stock_video_path, video_url, "stock video", scene_order)

            # Build scene clip with FFmpeg
            scene_output = scene_dir / "clip.mp4"

            if stock_video_path and stock_video_path.exists():
                # Use stock video as base
                cmd = build_video_scene_cmd(
                    stock_video_path, audio_path, scene_output,
                    width, height, duration, narration_volume,
                    text if caption_style != "none" else None, caption_style
                )
            elif image_path and image_path.exists():
                # Use image with ken burns / static
                cmd = build_image_scene_cmd(
                    image_path, audio_path, scene_output,
                    width, height, duration, narration_volume, media_type,
                    ken_burns_style, color_grading, enable_vignette, i,
                    text if caption_style != "none" else None, caption_style
                )
            else:
                if selected_media_type in ("image", "video") or not is_title_card:
                    raise RuntimeError(
                        f"Scene {scene_order}: no usable media after download; refusing to render black fallback"
                    )
                # Black screen with audio
                cmd = build_black_scene_cmd(
                    audio_path, scene_output,
                    width, height, duration, narration_volume,
                    text if caption_style != "none" else None, caption_style
                )

            print(f"Scene {i} command: {' '.join(cmd)}")
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                print(f"Scene {i} FFmpeg error: {stderr.decode()[-500:]}")
                raise RuntimeError(f"FFmpeg failed on scene {i}")

            scene_clips.append(scene_output)
            scene_durations.append(duration)

        concat_output = work_dir / "concat.mp4"
        concat_cmd = build_concat_cmd(
            scene_clips,
            scene_durations,
            concat_output,
            work_dir,
            transition_style,
            transition_duration,
        )
        proc = await asyncio.create_subprocess_exec(
            *concat_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Concat failed: {stderr.decode()[-500:]}")

        # Add background music if provided
        if bg_music_url and bg_music_url not in ("none", "", "undefined"):
            music_path = work_dir / "bgmusic.mp3"
            await download_file(bg_music_url, music_path)

            if music_path.exists():
                music_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(concat_output),
                    "-stream_loop", "-1", "-i", str(music_path),
                    "-filter_complex",
                    f"[1:a]volume={bg_music_volume}[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]",
                    "-map", "0:v", "-map", "[aout]",
                    "-c:v", "copy", "-c:a", "aac", "-shortest",
                    str(output_path)
                ]
                proc = await asyncio.create_subprocess_exec(
                    *music_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                _, stderr = await proc.communicate()
                if proc.returncode != 0:
                    print(f"Music mix failed, using without music: {stderr.decode()[-300:]}")
                    output_path = concat_output
            else:
                output_path = concat_output
        else:
            # No bg music — just rename
            os.rename(concat_output, output_path)

        if not output_path.exists():
            raise RuntimeError("Final output file not found")

        # Upload to Supabase Storage
        video_url = None
        if supabase_url and supabase_key:
            video_url = await upload_to_supabase(
                output_path, project_id, supabase_url, supabase_key
            )

        jobs[job_id] = {
            "status": "done",
            "jobId": job_id,
            "projectId": project_id,
            "videoUrl": video_url,
        }

        # Send callback
        if callback_url:
            await send_callback(callback_url, {
                "jobId": job_id,
                "projectId": project_id,
                "status": "done",
                "videoUrl": video_url,
            })

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"Render failed for {job_id}: {error_msg}")
        traceback.print_exc()

        jobs[job_id] = {
            "status": "failed",
            "jobId": job_id,
            "projectId": project_id,
            "error": error_msg,
        }

        if callback_url:
            await send_callback(callback_url, {
                "jobId": job_id,
                "projectId": project_id,
                "status": "failed",
                "error": error_msg,
            })

    finally:
        # Cleanup temp files (keep final output for a while)
        pass


def get_caption_filter(text: str, style: str, width: int, height: int) -> str:
    """Generate FFmpeg drawtext filter for captions."""
    # Escape special chars for FFmpeg
    escaped = text.replace("'", "'\\''").replace(":", "\\:").replace("%", "%%")
    
    font_size = int(height * 0.04)  # 4% of height
    margin = int(height * 0.08)

    if style == "bold":
        font_size = int(height * 0.05)
        return (
            f"drawtext=text='{escaped}':fontsize={font_size}:fontcolor=white:"
            f"borderw=3:bordercolor=black:x=(w-text_w)/2:y=h-{margin}:"
            f"font=Arial"
        )
    elif style == "minimal":
        return (
            f"drawtext=text='{escaped}':fontsize={font_size}:fontcolor=white@0.8:"
            f"x=(w-text_w)/2:y=h-{margin}:font=Arial"
        )
    else:  # default
        return (
            f"drawtext=text='{escaped}':fontsize={font_size}:fontcolor=white:"
            f"borderw=2:bordercolor=black@0.7:x=(w-text_w)/2:y=h-{margin}:"
            f"font=Arial:box=1:boxcolor=black@0.4:boxborderw=8"
        )


def color_grading_filter(grading: str) -> str:
    grading = (grading or "none").lower()
    if grading == "cinematic_warm":
        return "eq=contrast=1.08:saturation=1.10,colorbalance=rs=0.15:gs=0.05:bs=-0.10"
    if grading == "cinematic_cold":
        return "eq=contrast=1.05:saturation=0.95,colorbalance=rs=-0.10:gs=0.02:bs=0.18"
    if grading == "vintage":
        return "eq=contrast=0.92:saturation=0.75:gamma=1.05,colorbalance=rs=0.08:gs=0.04:bs=-0.15"
    if grading == "noir":
        return "hue=s=0,eq=contrast=1.35:brightness=-0.04"
    return ""


def vignette_filter(enabled: bool) -> str:
    return "vignette=PI/4" if enabled else ""


def ken_burns_filter(style: str, scene_index: int, duration: float, w: int, h: int) -> str:
    style = (style or "dynamic").lower()
    if style == "dynamic":
        style = "zoom_in" if scene_index % 2 == 0 else "zoom_out"

    frames = max(1, int(round(duration * FPS)))
    if style == "zoom_out":
        zoom_expr = "if(eq(on,0),1.25,max(zoom-0.0010,1.00))"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif style == "pan_horizontal":
        zoom_expr = "1.18"
        x_expr = "if(eq(on,0),0,min(x+1,iw-iw/zoom))"
        y_expr = "ih/2-(ih/zoom/2)"
    else:
        zoom_expr = "min(zoom+0.0010,1.25)"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"

    return f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}':d={frames}:s={w}x{h}:fps={FPS}"


def compose_video_filter(
    base_filter: str,
    text: str | None,
    caption_style: str,
    w: int,
    h: int,
    color_grading: str = "none",
    enable_vignette: bool = False,
) -> str:
    parts = [base_filter]
    grading = color_grading_filter(color_grading)
    if grading:
        parts.append(grading)
    vignette = vignette_filter(enable_vignette)
    if vignette:
        parts.append(vignette)
    if text:
        parts.append(get_caption_filter(text, caption_style, w, h))
    return ",".join(parts)


def build_image_scene_cmd(
    image: Path, audio: Path | None, output: Path,
    w: int, h: int, duration: float, narr_vol: float,
    media_type: str, ken_burns_style: str = "dynamic",
    color_grading: str = "none", enable_vignette: bool = False,
    scene_index: int = 0, text: str | None = None, caption_style: str = "default"
) -> list[str]:
    """Build FFmpeg command for an image-based scene."""
    # Ken burns: slow zoom effect
    if media_type == "ken_burns":
        base_vf = ken_burns_filter(ken_burns_style, scene_index, duration, w, h)
    else:
        base_vf = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={FPS}"

    vf = compose_video_filter(base_vf, text, caption_style, w, h, color_grading, enable_vignette)
    cmd = ["ffmpeg", "-y", "-loop", "1", "-t", str(duration), "-i", str(image)]

    if audio:
        cmd += ["-i", str(audio)]
        cmd += ["-filter_complex", f"[0:v]{vf}[v];[1:a]volume={narr_vol}[a]"]
        cmd += ["-map", "[v]", "-map", "[a]"]
    else:
        cmd += ["-f", "lavfi", "-t", str(duration), "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
        cmd += ["-filter_complex", f"[0:v]{vf}[v];[1:a]volume=0[a]"]
        cmd += ["-map", "[v]", "-map", "[a]"]

    cmd += [
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-pix_fmt", "yuv420p", "-r", str(FPS), "-shortest",
        str(output)
    ]
    return cmd


def build_video_scene_cmd(
    video: Path, audio: Path | None, output: Path,
    w: int, h: int, duration: float, narr_vol: float,
    text: str | None = None, caption_style: str = "default"
) -> list[str]:
    """Build FFmpeg command for a stock-video-based scene."""
    base_vf = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={FPS}"
    vf = compose_video_filter(base_vf, text, caption_style, w, h)

    # Stock clips are often shorter than the narration. Loop the video input so
    # the scene duration is controlled by the measured narration/scene length,
    # not by the source clip length.
    cmd = ["ffmpeg", "-y", "-stream_loop", "-1", "-i", str(video)]

    if audio:
        cmd += ["-i", str(audio)]
        cmd += ["-filter_complex", f"[0:v]{vf}[v];[1:a]volume={narr_vol}[a]"]
        cmd += ["-map", "[v]", "-map", "[a]", "-ignore_unknown"]
    else:
        cmd += ["-f", "lavfi", "-t", str(duration), "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
        cmd += ["-filter_complex", f"[0:v]{vf}[v];[1:a]volume=0[a]"]
        cmd += ["-map", "[v]", "-map", "[a]", "-ignore_unknown"]

    cmd += [
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-pix_fmt", "yuv420p", "-r", str(FPS), "-shortest",
        str(output)
    ]
    return cmd


def build_black_scene_cmd(
    audio: Path | None, output: Path,
    w: int, h: int, duration: float, narr_vol: float,
    text: str | None = None, caption_style: str = "default"
) -> list[str]:
    """Build FFmpeg command for a black screen scene (no image/video)."""
    vf = f"color=c=black:s={w}x{h}:r={FPS}:d={duration}"
    if text:
        vf_text = get_caption_filter(text, caption_style, w, h)
        vf = f"{vf},{vf_text}"

    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", vf]

    if audio:
        cmd += ["-i", str(audio)]
        cmd += ["-filter_complex", f"[1:a]volume={narr_vol}[a]"]
        cmd += ["-map", "0:v", "-map", "[a]"]
    else:
        cmd += ["-f", "lavfi", "-t", str(duration), "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
        cmd += ["-filter_complex", "[1:a]volume=0[a]"]
        cmd += ["-map", "0:v", "-map", "[a]"]

    cmd += [
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-pix_fmt", "yuv420p", "-r", str(FPS), "-shortest",
        str(output)
    ]
    return cmd


def build_concat_cmd(
    clips: list[Path],
    durations: list[float],
    output: Path,
    work_dir: Path,
    transition_style: str,
    transition_duration: float,
) -> list[str]:
    if len(clips) == 1:
        return [
            "ffmpeg", "-y", "-i", str(clips[0]),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-pix_fmt", "yuv420p", "-r", str(FPS),
            str(output),
        ]

    style = (transition_style or "none").lower()
    if style in ("none", "cut") or transition_duration <= 0:
        concat_list = work_dir / "concat.txt"
        with open(concat_list, "w") as f:
            for clip in clips:
                f.write(f"file '{clip}'\n")
        return [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-pix_fmt", "yuv420p", "-r", str(FPS),
            str(output),
        ]

    transition = normalize_xfade_transition(style)
    pair_limit = min([d / 3 for d in durations if d > 0] or [0])
    d = max(0.05, min(float(transition_duration), pair_limit))
    if d <= 0.05:
        return build_concat_cmd(clips, durations, output, work_dir, "none", 0)

    cmd = ["ffmpeg", "-y"]
    for clip in clips:
        cmd += ["-i", str(clip)]

    filters = []
    for i in range(len(clips)):
        filters.append(f"[{i}:v]settb=AVTB,setpts=PTS-STARTPTS[v{i}]")
        filters.append(f"[{i}:a]asetpts=PTS-STARTPTS[a{i}]")

    prev_v = "v0"
    prev_a = "a0"
    accumulated = durations[0]
    for i in range(1, len(clips)):
        offset = max(0, accumulated - d)
        out_v = f"xv{i}"
        out_a = f"xa{i}"
        filters.append(f"[{prev_v}][v{i}]xfade=transition={transition}:duration={d}:offset={offset}[{out_v}]")
        filters.append(f"[{prev_a}][a{i}]acrossfade=d={d}:c1=tri:c2=tri[{out_a}]")
        prev_v = out_v
        prev_a = out_a
        accumulated += durations[i] - d

    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", f"[{prev_v}]", "-map", f"[{prev_a}]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-pix_fmt", "yuv420p", "-r", str(FPS),
        str(output),
    ]
    return cmd


def normalize_xfade_transition(style: str) -> str:
    aliases = {
        "crossfade": "fade",
        "fade_black": "fadeblack",
        "fadeblack": "fadeblack",
        "fade_white": "fadewhite",
        "fadewhite": "fadewhite",
        "slide_left": "slideleft",
        "slide_right": "slideright",
        "slide_up": "slideup",
        "slide_down": "slidedown",
        "wipe_left": "wipeleft",
        "wipe_right": "wiperight",
        "wipe_up": "wipeup",
        "wipe_down": "wipedown",
    }
    normalized = aliases.get((style or "fade").lower(), (style or "fade").lower())
    supported = {
        "fade", "wipeleft", "wiperight", "wipeup", "wipedown",
        "slideleft", "slideright", "slideup", "slidedown",
        "circlecrop", "rectcrop", "distance", "fadeblack", "fadewhite",
        "radial", "smoothleft", "smoothright", "smoothup", "smoothdown",
        "circleopen", "circleclose", "vertopen", "vertclose",
        "horzopen", "horzclose", "dissolve", "pixelize",
        "diagtl", "diagtr", "diagbl", "diagbr",
        "hlslice", "hrslice", "vuslice", "vdslice",
    }
    if normalized not in supported:
        print(f"Unsupported xfade transition '{style}', falling back to fade")
        return "fade"
    return normalized


async def upload_to_supabase(
    file_path: Path, project_id: str,
    supabase_url: str, supabase_key: str
) -> str | None:
    """Upload the rendered video to Supabase Storage."""
    bucket = "videos"
    object_path = f"{project_id}/final.mp4"

    # Read file
    file_bytes = file_path.read_bytes()

    async with httpx.AsyncClient(timeout=300) as client:
        # Try to create bucket (ignore if exists)
        await client.post(
            f"{supabase_url}/storage/v1/bucket",
            headers={
                "Authorization": f"Bearer {supabase_key}",
                "apikey": supabase_key,
            },
            json={"id": bucket, "name": bucket, "public": True},
        )

        # Upload file
        resp = await client.put(
            f"{supabase_url}/storage/v1/object/{bucket}/{object_path}",
            headers={
                "Authorization": f"Bearer {supabase_key}",
                "apikey": supabase_key,
                "Content-Type": "video/mp4",
                "x-upsert": "true",
            },
            content=file_bytes,
        )

        if resp.status_code in (200, 201):
            public_url = f"{supabase_url}/storage/v1/object/public/{bucket}/{object_path}"
            print(f"Uploaded video: {public_url}")
            return public_url
        else:
            print(f"Upload failed ({resp.status_code}): {resp.text[:300]}")
            return None


async def send_callback(url: str, data: dict):
    """Send render result to the callback URL."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=data)
            print(f"Callback sent ({resp.status_code}): {data.get('status')}")
    except Exception as e:
        print(f"Callback failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
