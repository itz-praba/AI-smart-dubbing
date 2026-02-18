from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token="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"
)

pipeline.save_pretrained("D:/AI_Models/pyannote/speaker-diarization-3.1")
