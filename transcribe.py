from faster_whisper import WhisperModel
import json
import os

def transcribe_episode(audio_path, model: WhisperModel):
    segments, info = model.transcribe(audio_path, word_timestamps=True)
    
    segment_list = []
    for segment in segments:
        segment_list.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "words": [{"word": w.word, "start": w.start, "end": w.end} for w in segment.words]
        })
    
    return {
        "audio_path": audio_path,
        "language": info.language,
        "duration": info.duration,
        "segments": segment_list
    }
    
if __name__ == "__main__":
    model = WhisperModel("base", device="cpu", compute_type="int8")

    os.makedirs("transcripts", exist_ok=True)
    transcripts = os.listdir("transcripts")
    for filename in os.listdir("episodes"):
        if filename.endswith(".mp3"):
            if filename.replace(".mp3", ".json") in transcripts:
                continue
            print(f"Transcribing {filename}...")
            result = transcribe_episode(f"episodes/{filename}", model)

            out_name = filename.replace(".mp3", ".json")
            with open(f"transcripts/{out_name}", "w") as f:
                json.dump(result, f, indent=2)

            print(f"Done - {len(result['segments'])} segments, {result['duration']:.1f}s")