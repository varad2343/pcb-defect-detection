# PCB defect detection: generalization fix report

## Root cause (confirmed from code + measurements)
1. CNN was a 6-class crop classifier with NO "not a defect" class, and app.py fed it every contour in the photo -> background, text, hands all got a defect label.
2. "100% validation" was leakage: random crop-level split over boards that share templates. On whole unseen board groups, accuracy is ~70-73%.
3. Contour candidates covered only 41% of true defects; trained only on grayscale DeepPCB.

## Changes (all in C:\pcbproj; originals untouched)
- pcb_common.py: board localization (hue-peak / Otsu), crop+rescale to training scale, sliding windows inside the board, per-crop contrast stretch (same prep in train and inference), NMS, background rejection.
- 6_train_robust.py: 7 classes (6 defects + background), board-GROUP split (group50600, group90100 held out), background from defect-free _temp images + far-from-defect windows + synthetic junk, augmentation (polarity, brightness/contrast, gamma, blur, noise, resolution loss, rotation/scale).
- 7_evaluate.py / sweep.py: before/after; app_v2.py: new Streamlit app.
- Model: trained on Colab T4 (73.4% best val acc on unseen boards) and locally on CPU (66.8%). Colab model adopted -> pcb_cnn_model_v2.keras (local one kept as pcb_cnn_model_v2_local.keras).

## Results, 15 held-out boards (recall / false alarms per board)
| set | old pipeline | local model @0.5 | COLAB model @0.5 |
|---|---|---|---|
| gray DeepPCB | 8.8% / 14.8 | 55.7% / 5.6 | 86.4% / 5.3 |
| synthetic colour photo (recoloured, cluttered table) | 6.6% / 2.7 | 34.1% / 9.1 | 56.8% / 8.9 |
False alarms outside the board: 0.00 for new model at conf>=0.7; 0.60/img at 0.5 on synthetic photos (old: 0.40 at 0.95).
Real downloaded photos (15, CC-licensed, unlabeled, in C:\pcb_real): old pipeline reported 185 detections on one board; new reports 0-16 per photo. No labels -> real-photo recall NOT measured.

## Limits
- Confidence threshold trades recall vs false alarms (0.4: 98%/9.6 FP; 0.6: 37%/2.8 FP on gray). Slider in app_v2.py.
- Board localizer failed on 2 of 15 photos (dark/low-saturation boards).
- Training data is grayscale DeepPCB only; to truly fix real colour photos, label ~100-200 real colour PCB photos and fine-tune.
- Synthetic-photo numbers are a proxy, not real-world proof.

## Run
cd C:\pcbproj ; streamlit run app_v2.py   (needs tensorflow, opencv, streamlit; TF env used here: C:\tfenv)
