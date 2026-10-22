# Literature Review (Week 1)

Before you cite any of these, check the details (year, pages, venue) against the original paper.

## Cognitive load
* **Sweller, J. (1988).** Cognitive load during problem solving: Effects on learning. *Cognitive Science, 12*(2), 257-285.
  This paper introduced cognitive load theory: working memory is limited, and task demands use it up.
  It is the theoretical basis for treating "load" as a quantity that changes over time.
* **Hart, S. G., & Staveland, L. E. (1988).** Development of NASA-TLX. In *Human Mental Workload*.
  NASA-TLX is the standard subjective workload questionnaire. It is the natural ground truth for a future validation study of this system.
* **Kahneman, D. (1973).** *Attention and Effort.* This book links mental effort to observable physiological changes, which supports the use of eye-based and behavioural proxies.

## Eye blinks and fatigue
* **Stern, J. A., Boyer, D., & Schroeder, D. (1994).** Blink rate: a possible measure of fatigue. *Human Factors, 36*(2), 285-297.
  The authors found that blink rate rises with time on task and fatigue. This supports our fatigue ratio.
* **Soukupova, T., & Cech, J. (2016).** Real-time eye blink detection using facial landmarks. *CVWW 2016.*
  This paper defines the Eye Aspect Ratio (EAR) used in `signals.eye_aspect_ratio`. EAR drops sharply when the eye closes, and thresholding it detects blinks without a dedicated eye model.

## Facial landmarks, gaze and head pose
* **Kartynnik, Y., Ablavatski, A., Grishchenko, I., & Grundmann, M. (2019).** Real-time facial surface geometry from monocular video on mobile GPUs. *arXiv:1907.06724.*
  This is the MediaPipe Face Mesh model (468 3D landmarks, real-time on CPU) used in `landmarks.py`.
* **Ablavatski, A., et al. (2020).** Real-time pupil tracking from monocular video for digital puppetry. *arXiv:2006.11341.*
  This paper adds the iris landmarks (468-477) that we use for gaze offset.
* **Murphy-Chutorian, E., & Trivedi, M. M. (2009).** Head pose estimation in computer vision: A survey. *IEEE TPAMI, 31*(4), 607-626.
  The survey covers model-based pose estimation. We use the Perspective-n-Point (PnP) approach with a generic 3D face model (`cv2.solvePnP`).

## Gap this project addresses
Most webcam attention systems use one cue, such as blinks or head pose, with fixed thresholds.
This project adds three things to that approach:
1. Each signal is normalised to a personal baseline.
2. The signals are fused with reliability-aware, data-calibrated weights (AWR-CLF).
3. The weights and events change with the monitoring context (Examination or Interview).
