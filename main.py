from slurm import *

if __name__ == "__main__":
	path, data = full_slurm(
		"audio/beep.mp3",
		161,
		slice_settings= SliceSettings(
			beat_size=1,
			mix=0.8
		),
		echo_settings= EchoSettings(
			mix=0.2,
			multiplier = 0.6,
			internal_flip = False,
			internal_resample_multiplier = 1
		),
		timing_function=lambda t: 1.5 - t * 0.5
	)