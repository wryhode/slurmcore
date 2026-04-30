from slurm import *

if __name__ == "__main__":
	path, data = full_slurm(
		"audio/scraper.wav",
		175,
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
		#output_resample_multiplier=1.25
	)
