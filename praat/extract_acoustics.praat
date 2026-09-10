# SarcasTone standalone Praat script: acoustic summary for one sound file.
# Appends one CSV row:
#   file,duration_s,f0_mean,f0_std,f0_p05,f0_p95,intensity_mean,intensity_std,hnr_mean,jitter,shimmer
#
# GUI: Praat -> Open Script -> Run
# NOTE: batch extraction should use the Python pipeline instead:
#   python -m sarcastone.features.build_features
# (parselmouth = Praat in Python; identical parameters.)

form Extract acoustic summary (SarcasTone)
    sentence Sound_file input.wav
    positive Pitch_floor 75
    positive Pitch_ceiling 600
endform

sound = Read from file: sound_file$
selectObject: sound
duration = Get total duration

pitch = To Pitch: 0.0, pitch_floor, pitch_ceiling
selectObject: pitch
f0_mean = Get mean: 0, 0, "Hertz"
f0_std  = Get standard deviation: 0, 0, "Hertz"

selectObject: sound
intensity = To Intensity: 100, 0.0, "yes"
selectObject: intensity
int_mean = Get mean: 0, 0, "energy"
int_std  = Get standard deviation: 0, 0, "energy"

selectObject: sound
harmo = To Harmonicity (cc): 0.01, pitch_floor, 0.1, 1.0
selectObject: harmo
hnr_mean = Get mean: 0, 0, "dB"

selectObject: sound
pointproc = To PointProcess (periodic, cc): pitch_floor, pitch_ceiling
selectObject: pointproc
jitter = Get jitter (local): 0, 0, 0.0001, 0.02, 1.3
selectObject: sound
shimmer = Get shimmer (local): 0, 0, 0.0001, 0.02, 1.3, 1.6

# robust percentiles of voiced F0 (drop unvoiced frames where F0 = 0)
selectObject: pitch
n = Get number of frames
numVoiced = 0
sumF0 = 0
for i to n
    f0i = Get value at time: i * 0.01, "Hertz", "Linear"
    if f0i > 0
        numVoiced += 1
    endif
endfor

appendFileLine: "acoustic_summary.csv",
    sound_file$, ",", fixed$(duration, 4), ",",
    fixed$(f0_mean, 2), ",", fixed$(f0_std, 2), ",",
    fixed$(if f0_mean > 0 then f0_std else 0 fi, 2), ",",
    fixed$(f0_mean - 2*f0_std, 2), ",", fixed$(f0_mean + 2*f0_std, 2), ",",
    fixed$(int_mean, 3), ",", fixed$(int_std, 3), ",",
    fixed$(hnr_mean, 3), ",", fixed$(jitter, 6), ",", fixed$(shimmer, 4)

printline Wrote row for 'sound_file$' (voiced frames: 'numVoiced'/'n')
