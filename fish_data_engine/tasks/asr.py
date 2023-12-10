import torch
from fish_data_engine.tasks.task import Task, IS_WORKER
from fish_data_engine.utils.file import list_files, AUDIO_EXTENSIONS
import click
from random import Random
import soundfile as sf
from demucs.apply import apply_model as apply_demucs_model
from demucs.pretrained import get_model as get_demucs_model
import torchaudio
from loguru import logger
from pyannote.audio import Pipeline as PyannotePipeline
from modelscope.pipelines import pipeline as modelscope_pipeline
from modelscope.utils.constant import Tasks as ModelscopeTasks
import json
import sys


class ASR(Task):
    def __init__(
        self, input_dir, output_dir, demucs, speaker_separation, language, punctuation
    ):
        super().__init__()

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.demucs_model = None
        self.speaker_separation_pipeline = None

        if not IS_WORKER:
            return

        # Vocals extraction
        if demucs:
            logger.debug("Loading demucs model")
            self.demucs_model = get_demucs_model("htdemucs")
            self.demucs_model.eval()
            self.demucs_model.cuda()
            logger.debug("Demucs model loaded")

        # Voice activity detection
        if speaker_separation:
            logger.debug("Loading speaker separation model")
            self.speaker_separation_pipeline = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
            )
            self.speaker_separation_pipeline.to(torch.device("cuda"))
            logger.debug("Speaker separation model loaded")

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
        # Always clear GPU cache before processing a new job
        torch.cuda.empty_cache()

        audio, sr = torchaudio.load(job)
        logger.debug(f"Audio loaded, duration={audio.shape[1]/sr:.1f}s")
        audio = audio.cuda()

        relative_path = job.relative_to(self.input_dir)
        output_path = self.output_dir / relative_path.parent / relative_path.stem
        output_path.mkdir(parents=True, exist_ok=True)

        if (output_path / "asr.json").exists():
            logger.debug(f"ASR already done for {relative_path}")
            return

        # To mono
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)

        # Extract vocals
        if self.demucs_model:
            audio, sr = self.apply_demucs(audio, sr)
            torch.cuda.empty_cache()

        # Sample to 16kHz
        audio, sr = (
            torchaudio.functional.resample(audio, orig_freq=sr, new_freq=16000),
            16000,
        )

        # Speaker separation
        if not self.speaker_separation_pipeline:
            segments = [[(0, audio.shape[-1] / sr)]]
        else:
            segments = self.apply_speaker_separation(audio, sr)

            with open(output_path / "vad.json", "w") as f:
                json.dump(segments, f)

        if len(segments) == 0:
            logger.debug("No speaker found")
            return

        # Build speaker segments mask on CPU
        speaker_segments_masks = torch.zeros(
            (len(segments), audio.shape[-1]), dtype=torch.int
        )
        for i, turns in enumerate(segments):
            for start, end in turns:
                if start >= end:
                    continue

                speaker_segments_masks[i, int(start * sr) : int(end * sr)] = 1

        # Calculate overlapping and remove overlapping segments
        summed_speaker_segments_masks = speaker_segments_masks.sum(0)
        overlapping_segments_mask = summed_speaker_segments_masks > 1
        speaker_segments_masks[:, overlapping_segments_mask] = 0
        logger.debug(f"Removed {overlapping_segments_mask.sum()} overlapping segments")

        # ASR
        logger.debug("Start ASR")
        asr_results = []

        for spk_id, mask in enumerate(speaker_segments_masks):
            if mask.sum() == 0:
                continue

            coped_audio = audio.clone()
            coped_audio[:, mask == 0] = 0

            # Save the audio
            sf.write(
                output_path / f"speaker-{spk_id}.mp3", coped_audio[0].cpu().numpy(), sr
            )

            # ASR
            sentences = self.asr_pipeline(
                audio_in=coped_audio[0].cpu().numpy(), sample_rate=sr
            )["sentences"]
            asr_result = [
                {
                    "start": s["start"] / sr,
                    "end": s["end"] / sr,
                    "text": s["text"],
                }
                for s in sentences
            ]

            asr_results.append(asr_result)

        logger.debug("ASR done")
        with open(output_path / "asr.json", "w") as f:
            json.dump(asr_results, f, ensure_ascii=False)

    def apply_demucs(self, audio, sr):
        logger.debug("Apply demucs model")

        # Resample to 44.1kHz
        sampled_audio, sr = (
            torchaudio.functional.resample(
                audio, orig_freq=sr, new_freq=self.demucs_model.samplerate
            ),
            self.demucs_model.samplerate,
        )
        logger.debug(f"Audio resampled, shape={sampled_audio.shape}")

        # Make it 2 channels
        audio = torch.cat([sampled_audio, sampled_audio], dim=0)
        audio = (audio - sampled_audio.mean()) / audio.std()

        sources = apply_demucs_model(
            self.demucs_model,
            audio[None],
            device=audio.device,
            shifts=1,
            split=True,
            overlap=0.25,
            progress=False,
            num_workers=0,
        )[0]
        logger.debug(f"Demucs model applied, sources={sources.shape}")

        vocal = self.demucs_model.sources.index("vocals")
        sources = sources[vocal]
        audio = sources * sampled_audio.std() + sampled_audio.mean()
        logger.debug(f"Vocals extracted, duration={audio.shape[1]/sr:.1f}s")

        # Average the two channels
        audio = audio.mean(0, keepdim=True)

        return audio, sr

    def apply_speaker_separation(self, audio, sr):
        logger.debug("Apply speaker separation model")
        diarization = self.speaker_separation_pipeline(
            {"waveform": audio, "sample_rate": sr}
        )

        # Extract segments for each speaker
        speaker_to_turns = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # segments.append((turn.start, turn.end, speaker))
            start, end = turn.start, turn.end

            # Check if the speaker is already in the dictionary
            if speaker not in speaker_to_turns:
                speaker_to_turns[speaker] = [(start, end)]
            else:
                # Check if the current turn can be merged with the previous one
                last_start, last_end = speaker_to_turns[speaker][-1]
                if start <= last_end:
                    # Merge with the previous turn
                    speaker_to_turns[speaker][-1] = (last_start, max(end, last_end))
                else:
                    # Add a new turn for the speaker
                    speaker_to_turns[speaker].append((start, end))

        logger.debug(f"VAD model applied, found {len(speaker_to_turns)} speakers")

        # Calculate the total length of the audio in seconds
        audio.shape[-1] / sr

        # Initialize a new dictionary for filtered speaker turns
        filtered_speaker_to_turns = {}

        for speaker, turns in speaker_to_turns.items():
            total_speaking_time = sum(end - start for start, end in turns)

            # Check if total speaking time is more than 60s and more than 10% of the audio length
            if total_speaking_time > 60:
                filtered_speaker_to_turns[speaker] = turns

        # Output the filtered speaker turns
        logger.debug(
            f"After filtering, found {len(filtered_speaker_to_turns)} speakers"
        )

        return list(filtered_speaker_to_turns.values())


@click.command()
@click.option("--input-dir", type=str, required=True)
@click.option("--output-dir", type=str, required=True)
@click.option("--demucs/--no-demucs", default=True)
@click.option("--speaker-separation/--no-speaker-separation", default=True)
@click.option("--punctuation/--no-punctuation", default=True)
@click.option("--language", type=str, default="zh")
@click.option("--debug/--no-debug", default=False)
def main(
    input_dir, output_dir, demucs, speaker_separation, language, punctuation, debug
):
    if debug:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    task = ASR(input_dir, output_dir, demucs, speaker_separation, language, punctuation)
    task.run()


if __name__ == "__main__":
    main()
