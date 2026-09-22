"""Transcription module."""

import json
import os
from faster_whisper import WhisperModel

def transcribe_episode(episode, model: WhisperModel):
    """Transcribes a single episode's audio using faster-whisper.

    Returns:
        dict: audio_path, language, duration, and segments with
            start/end/text and word-level timestamps.
    """
    segments, info = model.transcribe(episode["path"], word_timestamps=True)
    
    segment_list = []
    for segment in segments:
        segment_list.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "words": [{"word": w.word, "start": w.start, "end": w.end} for w in segment.words] if segment.words else []
        })
        
    print(f"Transcribed {episode['title']} - {len(segment_list)} segments, {info.duration:.1f}s")
    
    return {
        "audio_path": episode["path"],
        "language": info.language,
        "duration": info.duration,
        "segments": segment_list
    }
    
    
def transcribe_in_subprocess(episode):
    """Runs transcribe_episode in a subprocess. Whisper (CTranslate2) and
    torch/sentence-transformers deadlock if they share a process, this
    keeps Whisper isolated in its own subprocess.
    """
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return transcribe_episode(episode, whisper_model)
    
    
if __name__ == "__main__":
    model = WhisperModel("base", device="cpu", compute_type="int8")
    os.makedirs("transcripts", exist_ok=True)
    transcripts = os.listdir("transcripts")
    for filename in os.listdir("episodes"):
        if filename.endswith(".mp3"):
            guid = filename.replace(".mp3", "")
            if f"{guid}.json" in transcripts:
                continue
            episode = {"path": f"episodes/{filename}", "title": guid}
            result = transcribe_episode(episode, model)
            with open(f"transcripts/{guid}.json", "w") as f:
                json.dump(result, f, indent=2)