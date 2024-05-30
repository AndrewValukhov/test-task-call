import argparse
import os
import json
import librosa
import numpy as np
import pydub
from pydub import AudioSegment
from scipy.io.wavfile import write
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def pydub_to_np(audio: pydub.AudioSegment) -> (np.ndarray, int):

    return np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((-1, audio.channels)) / (
            1 << (8 * audio.sample_width - 1)), audio.frame_rate


def speed_and_volume(path, speedup=0, volume_increase_db=0):

    if speedup > 1000:
        speedup = 1000
    if speedup <= -100:
        speedup = -99

    if volume_increase_db > 30:
        volume_increase_db = 30
    if volume_increase_db < -30:
        volume_increase_db = -30

    volume_increase_db = int(volume_increase_db)

    try:
        aso = AudioSegment.from_wav(path)
    except FileNotFoundError:
        print('Убедитесь, что передаете wav-файл и что путь к wav-файлу корректен')
        return
    aso = aso + volume_increase_db
    arr, sr = pydub_to_np(aso)
    new_sr = int(sr * (1 + speedup / 100))
    
    file_name = f"{path.split('/')[-1]}_speedup_{new_sr}_amp_{volume_increase_db}.wav"
    dir_name = 'result_dir'
    os.makedirs(dir_name, exist_ok=True)
    write(dir_name + '/' + file_name, new_sr, arr)
    print(f"модифицированный файл сохранен в {dir_name}/{file_name}")


def asr(audio_path, model_path="whisper_tiny_downloaded"):

    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.config.forced_decoder_ids = None
    try:
        wav, sr = librosa.load(audio_path, mono=False)
    except FileNotFoundError:
        print('Убедитесь, что передаете wav-файл и что путь к wav-файлу корректен')
        return
    wav_res = librosa.resample(wav, orig_sr=sr, target_sr=16000)

    input_features = processor(wav_res, sampling_rate=16000, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    dir_name = 'transcriptions_dir'
    os.makedirs(dir_name, exist_ok=True)
    file_name = f"{dir_name}/transcription_{len(os.listdir(dir_name))}.json"
    
    with open(file_name, 'w') as f:
        json.dump(transcription, f, ensure_ascii=False, indent=4)

    print(f"транскрипция сохранена в {file_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", help="location_of_audio_to_be_processed", type=str)
    parser.add_argument('task_type', choices=['audio_modification', 'asr'])
    parser.add_argument('--speedup',
                        type=float, help='increase / decrease speed of audio on this percent',
                        const=1,
                        nargs='?',
                        default=0)
    parser.add_argument('--vol_inc_db',
                        type=int,
                        help='increase /deacrease volume of audio on certain amount of db',
                        const=1,
                        nargs='?',
                        default=0)
    args = parser.parse_args()
    
    if args.task_type == 'audio_modification':
        speed_and_volume(args.audio_path, speedup=args.speedup, volume_increase_db=args.vol_inc_db)
    elif args.task_type == 'asr':
        asr(args.audio_path)