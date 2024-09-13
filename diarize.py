import argparse
import logging
import os
import re

import torch
import torchaudio
from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

from helpers import (
    cleanup,
    create_config,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    punct_model_langs,
    whisper_langs,
    write_srt,
    create_and_save_diarization_data,
    cut_audio_segments,
    expand_range
)
from transcription_helpers import transcribe_batched

from dotenv import load_dotenv

load_dotenv()

mtypes = {"cpu": "int8", "cuda": "float16"}

# Initialize parser
parser = argparse.ArgumentParser()
# parser.add_argument(
#     "-a", "--audio", help="name of the target audio file", required=True
# )
parser.add_argument(
    "--no-stem",
    action="store_false",
    dest="stemming",
    default=True,
    help="Disables source separation."
    "This helps with long files that don't contain a lot of music.",
)

parser.add_argument(
    "--suppress_numerals",
    action="store_true",
    dest="suppress_numerals",
    default=False,
    help="Suppresses Numerical Digits."
    "This helps the diarization accuracy but converts all digits into written text.",
)

parser.add_argument(
    "--whisper-model",
    dest="model_name",
    default="large",
    help="name of the Whisper model to use",
)

parser.add_argument(
    "--batch-size",
    type=int,
    dest="batch_size",
    default=8,
    help="Batch size for batched inference, reduce if you run out of memory, set to 0 for non-batched inference",
)

parser.add_argument(
    "--language",
    type=str,
    default=None,
    choices=whisper_langs,
    help="Language spoken in the audio, specify None to perform language detection",
)

parser.add_argument(
    "--device",
    dest="device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="if you have a GPU use 'cuda', otherwise 'cpu'",
)

args = parser.parse_args()

if args.stemming:
    # Isolate vocals from the rest of the audio

    return_code = os.system(
        f'python3 -m demucs.separate -n htdemucs --two-stems=vocals "{args.audio}" -o "temp_outputs"'
    )

    if return_code != 0:
        logging.warning(
            "Source splitting failed, using original audio file. Use --no-stem argument to disable it."
        )
        vocal_target = args.audio
    else:
        vocal_target = os.path.join(
            "temp_outputs",
            "htdemucs",
            os.path.splitext(os.path.basename(args.audio))[0],
            "vocals.wav",
        )
else:
    vocal_target = args.audio


# Transcribe the audio file

whisper_results, language, audio_waveform = transcribe_batched(
    vocal_target,
    args.language,
    args.batch_size,
    args.model_name,
    mtypes[args.device],
    args.suppress_numerals,
    args.device,
)

# Forced Alignment
alignment_model, alignment_tokenizer, alignment_dictionary = load_alignment_model(
    args.device,
    dtype=torch.float16 if args.device == "cuda" else torch.float32,
)

audio_waveform = (
    torch.from_numpy(audio_waveform)
    .to(alignment_model.dtype)
    .to(alignment_model.device)
)
emissions, stride = generate_emissions(
    alignment_model, audio_waveform, batch_size=args.batch_size
)

del alignment_model
torch.cuda.empty_cache()

full_transcript = "".join(segment["text"] for segment in whisper_results)

tokens_starred, text_starred = preprocess_text(
    full_transcript,
    romanize=True,
    language=langs_to_iso[language],
)

segments, scores, blank_id = get_alignments(
    emissions,
    tokens_starred,
    alignment_dictionary,
)

spans = get_spans(tokens_starred, segments, alignment_tokenizer.decode(blank_id))

word_timestamps = postprocess_results(text_starred, spans, stride, scores)


# convert audio to mono for NeMo combatibility
ROOT = os.getcwd()
temp_path = os.path.join(ROOT, "temp_outputs")
os.makedirs(temp_path, exist_ok=True)
torchaudio.save(
    os.path.join(temp_path, "mono_file.wav"),
    audio_waveform.cpu().unsqueeze(0).float(),
    16000,
    channels_first=True,
)


# Initialize NeMo MSDD diarization model
msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(args.device)
msdd_model.diarize()

del msdd_model
torch.cuda.empty_cache()

# Reading timestamps <> Speaker Labels mapping


speaker_ts = []
with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
    lines = f.readlines()
    for line in lines:
        line_list = line.split(" ")
        s = int(float(line_list[5]) * 1000)
        e = s + int(float(line_list[8]) * 1000)
        speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start", 100)

if language in punct_model_langs:
    # restoring punctuation in the transcript to help realign the sentences
    punct_model = PunctuationModel(model="kredor/punctuate-all")

    words_list = list(map(lambda x: x["word"], wsm))

    labled_words = punct_model.predict(words_list)

    ending_puncts = ".?!"
    model_puncts = ".,;:!?"

    # We don't want to punctuate U.S.A. with a period. Right?
    is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

    for word_dict, labeled_tuple in zip(wsm, labled_words):
        word = word_dict["word"]
        if (
            word
            and labeled_tuple[1] in ending_puncts
            and (word[-1] not in model_puncts or is_acronym(word))
        ):
            word += labeled_tuple[1]
            if word.endswith(".."):
                word = word.rstrip(".")
            word_dict["word"] = word

else:
    logging.warning(
        f"Punctuation restoration is not available for {language} language. Using the original punctuation."
    )

wsm = get_realigned_ws_mapping_with_punctuation(wsm)
ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

# Define the root output directory and the specific directory for the current audio file
root_output_dir = os.getenv("ROOT_OUTPUT_DIR")
audio_output_dir = os.getenv("AUDIO_OUTPUT_DIR")
audio_files = os.getenv('AUDIO_FILES')
input_audios = os.getenv('INPUT_AUDIOS')

extended_audio_files = expand_range(audio_files)

for audio_file in extended_audio_files:
    audio_path = os.path.join(input_audios, f'{audio_file}.wav')
    print(f"Diarizing audio file {audio_path}")
    audio_base_name = os.path.splitext(os.path.basename(args.audio))[0]
    specific_output_dir = os.path.join(root_output_dir, audio_base_name)
    output_audio = os.path.join(audio_output_dir, audio_base_name)

    # Ensure the specific output directory exists
    os.makedirs(specific_output_dir, exist_ok=True)

    # Define the paths for the output files
    output_text_file = os.path.join(specific_output_dir, f"{audio_base_name}.txt")
    output_srt_file = os.path.join(specific_output_dir, f"{audio_base_name}.srt")
    output_json_file = os.path.join(specific_output_dir, f"{audio_base_name}.json")
    output_diarization_file = output_json_file  # Reuse the path for diarization data

    # Write the transcript to a text file
    with open(output_text_file, "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)

    # Write the SRT file
    with open(output_srt_file, "w", encoding="utf-8-sig") as srt:
        write_srt(ssm, srt)

    # Create and save the diarization data to a JSON file
    diarization_data = create_and_save_diarization_data(ssm, output_diarization_file)

    # Cut the audio segments
    cut_audio_segments(vocal_target, diarization_data, output_audio, audio_base_name)

    # Clean up temporary files
    cleanup(temp_path)
