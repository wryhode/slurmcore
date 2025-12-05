"""
 slurm.py
 wryhode / greenspace 2025
"""

import os
import time

from dataclasses import dataclass

import librosa
import numpy as np
import soundfile

from scipy import signal

@dataclass
class EchoSettings:
	mix: float = 0.5
	multiplier: float = 1
	slice_mix: float = 0.65
	internal_resample_multiplier: float = 1.1
	internal_resample_drywet: float = 0.85
	internal_flip: bool = True
	internal_flip_drywet: float = 0.4
	flipflop: bool = False

@dataclass
class SliceSettings:
	beat_offset: float = 0.0
	beat_size: float = 0.5
	mix: float = 0.15
	reverse: bool = False

def get_fractional_slice(in_slice: np.ndarray, start: float, length: float) -> np.ndarray:
	""" Returns an array with a cutout of the data in in_slice """
	slice_start = int(len(in_slice) * start)
	slice_end = slice_start + int(len(in_slice) * length)
	return in_slice[slice_start : slice_end]

def resample_multiplier(data: np.ndarray, sample_rate: float, multiplier: float) -> np.ndarray:
	""" Wrapper around librosa.resample. Resamples an array to the samplerate * divider """
	if multiplier == 1.:
		return data
	return librosa.resample(data, orig_sr = sample_rate, target_sr = sample_rate * multiplier)

def resample_divider(data: np.ndarray, sample_rate: float, divider: float) -> np.ndarray:
	""" Wrapper around librosa.resample. Resamples an array to the samplerate / divider """
	if divider == 1.:
		return data
	return librosa.resample(data, orig_sr = sample_rate, target_sr = sample_rate / divider)

def resample_n_samples(data: np.ndarray, target_size: int) -> tuple:
	""" Scales an array down to a target size """
	return signal.resample(data, target_size)

def path_extend_filename(path: str, extension: str = '-slurmed') -> str:
	""" Extends a filename by something """
	_filepath, _file = os.path.split(path)
	_filename, _fileext = os.path.splitext(_file)
	new_filename = _filename + extension
	return os.path.join(_filepath, new_filename + _fileext)

def load_audio_file(path: str) -> tuple[np.ndarray, float]:
	""" Returns a tuple containing the sound data (mono) and the samplerate """
	data, sample_rate = librosa.load(path, sr=None)
	return data, sample_rate

def slurm(data: np.ndarray, sample_rate: float, beats_per_minute: float, slice_settings: SliceSettings, echo_settings: EchoSettings, timing_function = lambda t: 1) -> np.ndarray:
	data_size = len(data)
	track_length = data_size / sample_rate
	seconds_per_beat = 60 / beats_per_minute
	beat_sample_length = np.floor(seconds_per_beat * sample_rate)
	beats = np.floor(data_size / beat_sample_length)

	slices = np.array_split(data, beats)
	slurmed_slice_size = get_fractional_slice(slices[0], slice_settings.beat_offset, slice_settings.beat_size).shape
	slices_cut = [np.zeros(slurmed_slice_size) for _ in range(len(slices))]
	echo_buf = np.zeros(slurmed_slice_size)

	for i, slice in enumerate(slices):
		t = i / len(slices)
		#v = 0.9 + np.sin((1 / 64) * i * np.pi) * 0.1
		#v = 0.9 + t * 0.2
		v = timing_function(t)

		# slurm and resample new slice
		slice_data = resample_divider(get_fractional_slice(slice, slice_settings.beat_offset, slice_settings.beat_size), sample_rate, v)
		slices_cut[i] = slice_data * slice_settings.mix

		if slice_settings.reverse:
			slices_cut[i] = np.flip(slices_cut[i])

		# slice length can vary so we need to change the length of the echo buffer as well
		echo_buf = resample_n_samples(echo_buf, slice_data.shape[0])
		slices_cut[i] += echo_buf * echo_settings.mix # mix in echo to output slice

		echo_buf *= echo_settings.multiplier # echo decay / feedback

		# resample echo, then stretch it back to its original size.
		# effectively changes playback speed of the previous echos
		if echo_settings.internal_resample_multiplier != 1:
			echo_resamp = np.resize(resample_multiplier(echo_buf, sample_rate, echo_settings.internal_resample_multiplier), slice_data.shape) # change echo sample rate keeping n samples constant, effectively changing its speed
			echo_buf = echo_buf * (1 - echo_settings.internal_resample_drywet) + echo_resamp * echo_settings.internal_resample_drywet # mix speed changed echo with normal echo

		if echo_settings.internal_flip:
			to_echo = (slice_data * (1 - echo_settings.internal_flip_drywet) + np.flip(slice_data) * echo_settings.internal_flip_drywet) # normal slice + reversed slice
		else:
			to_echo = slice_data # normal slice

		echo_buf += to_echo * echo_settings.slice_mix # mix current slice to echo

		if echo_settings.flipflop:
			echo_buf = np.flip(echo_buf)

	# combine slices
	inter_data = np.concatenate(slices_cut)
	inter_data_length = len(inter_data) / sample_rate
	len_ratio = track_length / inter_data_length
	return inter_data

def full_slurm(path: str, bpm: float, input_resample_multiplier: float = 1, output_resample_multiplier: float = 1, slice_settings = SliceSettings(), echo_settings = EchoSettings(), timing_function = lambda t: 1):
	start_time = time.time()
	output_path = path_extend_filename(path, '-slurmed')

	print(f"Loading {path}...", end='')
	data, sample_rate = load_audio_file(path)
	print("Done")

	# a resample to start with
	if input_resample_multiplier != 1:
		print(f"Resampling input by {input_resample_multiplier}...", end='')
		data = resample_multiplier(data, sample_rate, input_resample_multiplier)
		print("Done")

	# slurm it
	print(f"Slurming...", end='')
	data = slurm(data, sample_rate, bpm, slice_settings, echo_settings, timing_function)
	print("Done")

	# a final resampling
	if output_resample_multiplier != 1:
		print(f"Resampling output by {output_resample_multiplier}...", end='')
		data = resample_multiplier(data, sample_rate, output_resample_multiplier)
		print("Done")

	new_length = len(data) / sample_rate

	print(f"Saving to {output_path}...", end='')
	soundfile.write(output_path, data, sample_rate)
	print("Done")
	slurm_time = time.time() - start_time
	print(f"All done! Took {round(slurm_time, 4)} seconds")
	return output_path, data
