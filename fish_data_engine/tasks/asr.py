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
    def __init__(self, input_dir, output_dir, demucs, vad, language, punctuation):
        super().__init__()

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.demucs_model = None
        self.vad_pipeline = None

        if not IS_WORKER:
            return

        # Vocals extraction
        if demucs:
            logger.debug("Loading demucs model")
            self.demucs_model = get_demucs_model("htdemucs")
            self.demucs_model.eval()
            self.demucs_model.to(torch.device("cuda"))
            logger.debug("Demucs model loaded")

        # Voice activity detection
        if vad:
            logger.debug("Loading VAD model")
            self.vad_pipeline = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
            )
            self.vad_pipeline.to(torch.device("cuda"))
            logger.debug("VAD model loaded")

        # ASR
        assert language in [
            "zh",
            "en",
        ], "Language must be either zh or en, whisper is not supported yet"

        MODELS = {
            "zh": "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            "en": "damo/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020",
        }

        kwargs = {
            "model": MODELS[language],
            "vad_model": "",  # Use our own VAD model
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
        audio, sr = torchaudio.load(job)
        logger.debug(f"Audio loaded, duration={audio.shape[1]/sr:.1f}s")
        audio = audio.cuda()

        relative_path = job.relative_to(self.input_dir)
        output_path = self.output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.with_suffix(".asr.json").exists():
            logger.debug(f"ASR already done for {relative_path}")
            return

        # To mono
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)

        # Extract vocals
        if self.demucs_model:
            # Apply demucs model
            audio, sr = self.apply_demucs(audio, sr)

            # Sample to 16kHz and save
            audio, sr = (
                torchaudio.functional.resample(audio, orig_freq=sr, new_freq=16000),
                16000,
            )
            sf.write(output_path.with_suffix(".vocals.mp3"), audio[0].cpu().numpy(), sr)

        # VAD
        if not self.vad_pipeline:
            segments = {
                "SPEAKER_0": [(0, audio.shape[-1] / sr)],
            }
        else:
            segments = self.apply_vad(audio, sr)

            with open(output_path.with_suffix(".vad.json"), "w") as f:
                json.dump(segments, f)

        if len(segments) == 0:
            logger.debug("No speaker found")
            return

        # ASR
        asr_results = {}
        logger.debug("Start ASR")

        for speaker, turns in segments.items():
            texts = []
            for start, end in turns:
                audio = audio[:, int(start * sr) : int(end * sr)]
                text = self.asr_pipeline(audio_in=audio, sample_rate=sr)["text"]
                texts.append({"start": start, "end": end, "text": text})

            asr_results[speaker] = texts

        logger.debug("ASR done")

        with open(output_path.with_suffix(".asr.json"), "w") as f:
            json.dump(asr_results, f)

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

    def apply_vad(self, audio, sr):
        logger.debug("Apply VAD model")
        diarization = self.vad_pipeline({"waveform": audio, "sample_rate": sr})

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
        total_audio_length = audio.shape[-1] / sr

        # Initialize a new dictionary for filtered speaker turns
        filtered_speaker_to_turns = {}

        for speaker, turns in speaker_to_turns.items():
            total_speaking_time = sum(end - start for start, end in turns)

            # Check if total speaking time is more than 60s and more than 10% of the audio length
            if (
                total_speaking_time > 60
                and total_speaking_time > 0.1 * total_audio_length
            ):
                filtered_speaker_to_turns[speaker] = turns

        # Output the filtered speaker turns
        logger.debug(
            f"After filtering, found {len(filtered_speaker_to_turns)} speakers"
        )

        # Initialize a new dictionary for processed speaker turns
        processed_speaker_to_turns = {}

        for speaker, turns in filtered_speaker_to_turns.items():
            if len(turns) == 1:
                processed_speaker_to_turns[speaker] = turns
                continue

            merged_turns = []
            current_start, current_end = turns[0]

            for start, end in turns[1:]:
                # Split the segment if its duration exceeds 30 seconds
                if current_end - current_start > 30:
                    merged_turns.append((current_start, current_end))
                    current_start, current_end = start, end
                    continue

                # Check if the gap between segments is 1 second or less
                if start - current_end <= 1:
                    # Extend the current segment
                    current_end = end
                else:
                    # Add the current segment and start a new one
                    merged_turns.append((current_start, current_end))
                    current_start, current_end = start, end

            # Add the last segment
            merged_turns.append((current_start, current_end))
            processed_speaker_to_turns[speaker] = merged_turns

        logger.debug(
            f"VAD done, found {sum(len(turns) for turns in processed_speaker_to_turns.values())} segments"
        )

        return processed_speaker_to_turns


@click.command()
@click.option("--input-dir", type=str, required=True)
@click.option("--output-dir", type=str, required=False)
@click.option("--demucs/--no-demucs", default=True)
@click.option("--vad/--no-vad", default=True)
@click.option("--punctuation/--no-punctuation", default=True)
@click.option("--language", type=str, default="zh")
@click.option("--debug/--no-debug", default=False)
def main(input_dir, output_dir, demucs, vad, language, punctuation, debug):
    if output_dir is None:
        output_dir = input_dir

    if debug:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    task = ASR(input_dir, output_dir, demucs, vad, language, punctuation)
    task.run()


if __name__ == "__main__":
    main()
