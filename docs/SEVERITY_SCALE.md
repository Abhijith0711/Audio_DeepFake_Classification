# Degradation severity

Training uses an ordinal severity. A higher number is a later, harder curriculum stage. Conditions that share a number are in the same stage. The scale is not calibrated: equal numbers are not a claim of equal difficulty, and the steps between numbers are not equal distances.

`scripts/estimate_condition_severity.py` can rank the same conditions by how much they drop a clean-trained model's development AUROC. That ranking is a measurement. It does not replace the labels in `src/robustness/conditions.py`.

| Condition | Family | Severity | Why this rank |
| --- | --- | ---: | --- |
| clean | identity | 0 | Unchanged audio. Not a degradation. |
| down_up_12k | resample | 1 | 24 kHz audio is taken down to 12 kHz and back. This is the mildest bandwidth cut in the set, so it is the first curriculum stage. |
| mp3_128k | mp3_codec | 2 | 128 kbps MP3. Mild codec, grouped with mild noise. |
| snr_20db | additive_noise | 2 | Noise at 20 dB SNR. Still a high signal-to-noise ratio, so the same stage as mild MP3. |
| mp3_64k | mp3_codec | 3 | 64 kbps MP3. A clearly stronger codec than 128 kbps. |
| snr_10db | additive_noise | 3 | Noise at 10 dB SNR. The middle noise level. |
| bandlimit_300_3400 | telephony | 3 | Energy outside 300–3400 Hz is removed, which is telephone bandwidth. Placed with the other moderate distortions. |
| mp3_32k | mp3_codec | 4 | 32 kbps MP3. The harshest codec in the set, so the last curriculum stage. |
| snr_5db | additive_noise | 4 | Noise at 5 dB SNR. The harshest noise level, in the same last stage. |

The curriculum turns these stages on over epochs. With milestones `[1, 6, 13]` and severities `[1, 3, 4]`, epochs 1–5 use severity 1 (the 12 kHz resample only), epochs 6–12 add severity 3, and epoch 13 onward adds severity 4. Severity 2 conditions are included once the cap reaches 3, because a condition is used when its severity is less than or equal to the epoch's cap.
