import torch
from fish_data_engine.tasks.task import Task, IS_WORKER
from fish_data_engine.utils.file import list_files, AUDIO_EXTENSIONS
import click
from random import Random
import soundfile as sf
import torchaudio
from loguru import logger
from modelscope.pipelines import pipeline as modelscope_pipeline
from modelscope.utils.constant import Tasks as ModelscopeTasks
import json
import sys


class ASR(Task):
    def __init__(self, input_dir, output_dir, language, punctuation):
        super().__init__()

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.demucs_model = None
        self.speaker_separation_pipeline = None

        if not IS_WORKER:
            return

        # ASR
        assert language in [
            "zh",
            "en",
        ], "Language must be either zh or en, whisper is not supported yet"

        MODELS = {
            "zh": "damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            "en": "damo/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020",
        }

        kwargs = {
            "model": MODELS[language],
        }

        if punctuation is False:
            kwargs["punc_model"] = ""

        logger.debug("Loading ASR model")
        self.asr_pipeline = modelscope_pipeline(
            task=ModelscopeTasks.auto_speech_recognition,
            **kwargs,
        )
        logger.debug("ASR model loaded")

    def jobs(self) -> list:
        files = list_files(self.input_dir, AUDIO_EXTENSIONS, recursive=True, sort=True)
        Random(42).shuffle(files)
        return files

    @torch.no_grad()
    def process(self, job):
        relative_path = job.relative_to(self.input_dir)
        output_path = self.output_dir / relative_path.parent / relative_path.stem
        output_path.mkdir(parents=True, exist_ok=True)

        if (output_path / "asr.json").exists():
            logger.debug(f"Chunk already done for {relative_path}")
            return

        # Always clear GPU cache before processing a new job
        torch.cuda.empty_cache()

        audio, sr = torchaudio.load(job)
        logger.debug(f"Audio loaded: {job}, duration={audio.shape[1]/sr:.1f}s")
        raw_audio, raw_sr = audio.clone(), sr
        audio = audio.cuda()

        # To mono
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)

        # Sample to 16kHz
        audio, sr = (
            torchaudio.functional.resample(audio, orig_freq=sr, new_freq=16000),
            16000,
        )

        # ASR
        logger.debug("Start ASR")
        sentences = self.asr_pipeline(audio_in=audio[0].cpu().numpy(), sample_rate=sr)[
            "sentences"
        ]
        logger.debug("ASR done")

        asr_result = [
            {
                "start": s["start"] / 1000,
                "end": s["end"] / 1000,
                "text": s["text"],
            }
            for s in sentences
        ]

        # Cut audio into segments
        for idx, s in enumerate(sentences):
            start = int(s["start"] / 1000 * raw_sr)
            end = int(s["end"] / 1000 * raw_sr)
            segment = raw_audio[:, start:end]
            sf.write(output_path / f"{idx:06d}.mp3", segment[0].cpu().numpy(), raw_sr)

            with open(output_path / f"{idx:06d}.txt", "w") as f:
                f.write(s["text"])

        with open(output_path / "asr.json", "w") as f:
            json.dump(asr_result, f, ensure_ascii=False)


@click.command()
@click.option("--input-dir", type=str, required=True)
@click.option("--output-dir", type=str, required=True)
@click.option("--language", type=str, default="zh")
@click.option("--punctuation/--no-punctuation", default=True)
@click.option("--debug/--no-debug", default=False)
def main(input_dir, output_dir, language, punctuation, debug):
    if debug:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    task = ASR(input_dir, output_dir, language, punctuation)
    task.run()


if __name__ == "__main__":
    main()
