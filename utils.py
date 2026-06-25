import re
import socket

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
)


def extract_video_id(url):
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be\/([a-zA-Z0-9_-]{11})",
        r"shorts\/([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    if len(url) == 11:
        return url

    raise ValueError("Invalid YouTube URL")


def get_video_title(url):
    try:
        from pytube import YouTube

        yt = YouTube(url)
        return yt.title
    except Exception:
        return "YouTube Video"


def get_playlist_videos(url):
    if "list=" not in url:
        return [url]

    import yt_dlp

    socket.setdefaulttimeout(30)

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
        "socket_timeout": 30,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:
        raise Exception(
            "Could not load the playlist. "
            "This may be caused by YouTube rate limits, "
            "private playlists, or cloud IP restrictions."
        )

    if "entries" in info and info["entries"]:
        return [
            f"https://www.youtube.com/watch?v={entry['id']}"
            for entry in info["entries"]
            if entry.get("id")
        ]

    return [url]


def get_transcript(video_id):
    api = YouTubeTranscriptApi()
    languages = ["en", "en-US", "en-GB", "hi", "hi-IN", "ur"]

    try:
        return api.fetch(video_id, languages=languages)
    except NoTranscriptFound:
        try:
            transcript_list = api.list(video_id)
            available = list(transcript_list)
            if not available:
                raise Exception(
                    "This video does not have captions available "
                    "or YouTube temporarily blocked the request."
                )
            return available[0].fetch()
        except Exception:
            raise Exception(
                "This video does not have captions available "
                "or YouTube temporarily blocked the request."
            )
    except TranscriptsDisabled:
        raise Exception(
            "This video does not have captions available "
            "or YouTube temporarily blocked the request."
        )
    except Exception:
        raise Exception(
            "This video does not have captions available "
            "or YouTube temporarily blocked the request."
        )
