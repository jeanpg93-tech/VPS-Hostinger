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
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Video Renderer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

RENDER_DIR = Path("/tmp/renders")
RENDER_DIR.mkdir(exist_ok=True)

# Track jobs
jobs: dict[str, dict] = {}


@app.get("/")
async def health():
    return {"status": "ok", "engine": "ffmpeg"}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return job


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


async def download_file(url: str, dest: Path) -> bool:
    """Download a file from URL to local path."""
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


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

        # Parse aspect ratio
        if aspect == "9:16":
            width, height = 1080, 1920
        elif aspect == "1:1":
            width, height = 1080, 1080
        else:
            width, height = 1920, 1080

        # Download all scene assets
        scene_clips = []
        for i, scene in enumerate(sorted(scenes, key=lambda s: s.get("scene_order", 0))):
            scene_dir = work_dir / f"scene_{i}"
            scene_dir.mkdir(exist_ok=True)

            duration = scene.get("estimated_duration", 5)
            image_url = scene.get("image_url")
            audio_url = scene.get("audio_url")
            video_url = scene.get("stock_video_url")
            text = scene.get("text_content", "")

            # Download image
            image_path = None
            if image_url:
                image_path = scene_dir / "image.jpg"
                await download_file(image_url, image_path)
                if not image_path.exists():
                    image_path = None

            # Download audio (narration)
            audio_path = None
            if audio_url:
                audio_path = scene_dir / "narration.mp3"
                await download_file(audio_url, audio_path)
                if not audio_path.exists():
                    audio_path = None

            # Download stock video
            stock_video_path = None
            if video_url:
                stock_video_path = scene_dir / "stock.mp4"
                await download_file(video_url, stock_video_path)
                if not stock_video_path.exists():
                    stock_video_path = None

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
                    text if caption_style != "none" else None, caption_style
                )
            else:
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

        # Concatenate all scene clips
        concat_list = work_dir / "concat.txt"
        with open(concat_list, "w") as f:
            for clip in scene_clips:
                f.write(f"file '{clip}'\n")

        concat_output = work_dir / "concat.mp4"
        concat_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy", str(concat_output)
        ]
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


def build_image_scene_cmd(
    image: Path, audio: Path | None, output: Path,
    w: int, h: int, duration: float, narr_vol: float,
    media_type: str, text: str | None = None, caption_style: str = "default"
) -> list[str]:
    """Build FFmpeg command for an image-based scene."""
    # Ken burns: slow zoom effect
    if media_type == "ken_burns":
        vf = f"zoompan=z='min(zoom+0.001,1.3)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={int(duration*25)}:s={w}x{h}:fps=25"
    else:
        vf = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"

    if text:
        vf += f",{get_caption_filter(text, caption_style, w, h)}"

    cmd = ["ffmpeg", "-y", "-loop", "1", "-i", str(image)]

    if audio:
        cmd += ["-i", str(audio)]
        cmd += ["-filter_complex", f"[0:v]{vf}[v];[1:a]volume={narr_vol}[a]"]
        cmd += ["-map", "[v]", "-map", "[a]"]
    else:
        cmd += ["-vf", vf]

    cmd += [
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-pix_fmt", "yuv420p",
        str(output)
    ]
    return cmd


def build_video_scene_cmd(
    video: Path, audio: Path | None, output: Path,
    w: int, h: int, duration: float, narr_vol: float,
    text: str | None = None, caption_style: str = "default"
) -> list[str]:
    """Build FFmpeg command for a stock-video-based scene."""
    vf = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"
    if text:
        vf += f",{get_caption_filter(text, caption_style, w, h)}"

    cmd = ["ffmpeg", "-y", "-i", str(video)]

    if audio:
        cmd += ["-i", str(audio)]
        cmd += ["-filter_complex", f"[0:v]{vf}[v];[1:a]volume={narr_vol}[a]"]
        cmd += ["-map", "[v]", "-map", "[a]", "-ignore_unknown"]
    else:
        cmd += ["-vf", vf]

    cmd += [
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-pix_fmt", "yuv420p",
        str(output)
    ]
    return cmd


def build_black_scene_cmd(
    audio: Path | None, output: Path,
    w: int, h: int, duration: float, narr_vol: float,
    text: str | None = None, caption_style: str = "default"
) -> list[str]:
    """Build FFmpeg command for a black screen scene (no image/video)."""
    vf = f"color=c=black:s={w}x{h}:d={duration}"
    if text:
        vf_text = get_caption_filter(text, caption_style, w, h)
        vf = f"{vf},{vf_text}"

    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", vf]

    if audio:
        cmd += ["-i", str(audio)]
        cmd += ["-filter_complex", f"[1:a]volume={narr_vol}[a]"]
        cmd += ["-map", "0:v", "-map", "[a]"]

    cmd += [
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-pix_fmt", "yuv420p",
        str(output)
    ]
    return cmd


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
