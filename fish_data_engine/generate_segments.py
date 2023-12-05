import librosa
import torch
from pyannote.audio.pipelines import VoiceActivityDetection
from fish_data_engine.task import Task, IS_WORKER
from fish_data_engine.utils.file import list_files, AUDIO_EXTENSIONS
import click
from random import Random, choices
import soundfile as sf
from fish_data_engine.annotation.api import create_sample
from tempfile import NamedTemporaryFile


class GenerateSegments(Task):
    def __init__(self, input_dir) -> None:
        super().__init__()

        self.input_dir = input_dir

        if IS_WORKER:
            self.pipeline = VoiceActivityDetection(
                segmentation="pyannote/segmentation",
            )
            self.pipeline.to(torch.device("cuda"))

    def jobs(self) -> list:
        files = list_files(self.input_dir, AUDIO_EXTENSIONS, recursive=True, sort=True)
        Random(42).shuffle(files)
        return files

    def process(self, job):
        np_audio, sr = librosa.load(job, sr=16000)
        audio = torch.from_numpy(np_audio[None]).cuda()

        # apply pretrained pipeline
        diarization = self.pipeline(
            {
                "waveform": audio,
                "sample_rate": sr,
            }
        )

        # print the result
        segments = []
        for turn, _, label in diarization.itertracks(yield_label=True):
            # print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{label}")
            segments.append((turn.start, turn.end, label))

        if len(segments) < 15:
            return

        segments = segments[:5] + choices(segments[5:-5], k=5) + segments[-5:]
        for start, end, _ in segments:
            segment = np_audio[int(start * sr) : int(end * sr)]

            with NamedTemporaryFile(suffix=".mp3") as f:
                sf.write(f.name, segment, sr)
                create_sample("audio-quality", f.name)


@click.command()
@click.option("--input-dir", type=str, required=True)
def main(input_dir):
    task = GenerateSegments(input_dir)
    task.run()


if __name__ == "__main__":
    main()
