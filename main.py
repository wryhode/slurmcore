from slurm import *
import os  

def filepath_validator(prompt:str, invalid="Invalid input. Please try again.") -> bool:
	"""Ensures file at user's input path exists"""
	while True:
		path = input(prompt)
		if os.path.exists(path):
			return path
		print(invalid)

def get_float(prompt:str, minimum:float=-math.inf, maximum:float=math.inf, default=0.0) -> float:
	"""Gets floats from user"""
	running = True
	while running:
		try:
			response = input(f"{prompt} (default {default}): ")
			if response == "":
				return default
			else:
				response = float(response)
				running = not minimum <= response <= maximum
		except:
			print("Invalid input. Please try again.")
	return response

def get_bool(prompt:str, default="disabled") -> bool:
	"""Asks user for y/n, defaults to no if not y, yes, or yeah. Will default to `default` if user presses enter."""
	user_input = input(f"{prompt}? (default {default}) y/n: ").lower()
	if user_input == "":
		return default=="enabled"
	return user_input in ("y", "yes", "yeah")

if __name__ == "__main__":
	while True:
		input_slurmpath = filepath_validator("Please input file path: ", invalid="File not found. (please remove quotation marks if present)")
		input_bpm = get_float("What is the BPM?", default=120)
		input_resample_mult = get_float("Multiply song length by", default=1)
		if (get_bool("Advanced settings")):
			path, data = full_slurm(
			input_slurmpath,
			input_bpm,
			slice_settings= SliceSettings(
				beat_offset = get_float("Beat offset"),
				beat_size = get_float("Beat size",default=0.5),
				mix = get_float("Mix",default=1.0),
				reverse = get_bool("Reverse")
			),
			echo_settings= EchoSettings(
				mix=get_float("Echo mix"),
				multiplier = get_float("Echo multiplier", default=0.8),
				internal_flip = get_bool("Internal flip"),
				internal_resample_multiplier = get_float("Internal resample multiplier", default=1),
				flipflop = get_bool("Flipflop", default="enabled")
			),
			output_resample_multiplier=input_resample_mult
			)
		path, data = full_slurm(
			input_slurmpath,
			input_bpm,
			slice_settings= SliceSettings(
				beat_offset = 0.0,
				beat_size = 0.5,
				mix = 1.0,
				reverse = False
			),
			echo_settings= EchoSettings(
				mix=0.0,
				multiplier = 0.8,
				internal_flip = False,
				internal_resample_multiplier = 1,
				flipflop = True
			),
			output_resample_multiplier=input_resample_mult
		)