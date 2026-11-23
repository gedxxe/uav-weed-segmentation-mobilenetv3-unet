# Retraining dari awal — Genze (protokol) + Zuo & Li (arsitektur)

Status: **cabang eksperimen**. Tidak mengubah `main`, checkpoint historis, atau original `train.py`/`retrain.py`.

## Batasan ilmiah

- Dataset: Genze et al. (2022), Mendeley Data v4, 3 kelas Background/Sorghum/Weed.
- Arsitektur: adaptasi MobileNetV3-Large + PPM + SE/H-Swish dari Zuo & Li (2024).
- Protokol: patch 256x256, augmentasi ala Genze, Adam dan Dice (eksperimen utama), grouped CV 4-fold, Optuna, final fit, test independen.
- **Perubahan disengaja dari Genze:** `GroupKFold` berdasarkan sumber citra agar patch satu capture tidak tersebar ke train/validation, hanya patch training yang difilter, jadwal `CosineAnnealingLR` (bukan `ReduceLROnPlateau`) yang sama untuk CV dan final training, dan anggaran awal 10 trial (bukan 50). Catat semua adaptasi dalam Bab III.
- Jika beberapa capture UAV saling tumpang tindih secara geografis, **group by source file saja belum cukup**. Tinjau koordinat/flight/field metadata dan gabungkan ID sumber yang merekam wilayah yang sama sebelum mengklaim bebas leakage.
- Test dan BBCH 15/19 tidak boleh dipakai untuk Optuna, early stopping, pemilihan model, atau menentukan epoch final.
- **Tidak ada metrik GPU yang dapat disimpulkan dari skrip atau unit test.** Semua angka penelitian harus berasal dari training dan evaluasi nyata.

## Langkah 0 — lingkungan Windows 11, RTX 5050

Buka PowerShell dari root repository. Jika baru clone gunakan:

```powershell
git clone https://github.com/gedxxe/uav-weed-segmentation-mobilenetv3-unet.git
cd uav-weed-segmentation-mobilenetv3-unet
git fetch origin
git switch research/grouped-retraining-2026
```

Verifikasi driver dan Python:

```powershell
nvidia-smi
py -3.13 --version
```

Instal setup yang sudah ada di repo:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows_cuda.ps1
.\.venv\Scripts\python.exe scripts\verify_cuda.py
```

Pemeriksaan dataset mungkin gagal pada setup bila file asli belum disalin. Letakkan raw files Genze sesuai `data/trainval/img/*.jpg`, `data/trainval/msk/*.png`, `data/test/...`, `data/test_different_bbch/...`. Jangan masukkan hasil atau checkpoint lama ke repo.

## Langkah 1 — patch dan manifest (hanya satu kali)

```powershell
.\.venv\Scripts\python.exe scripts\research_data.py --root .
.\.venv\Scripts\python.exe -m unittest tests.test_research_data tests.test_research_cv
```

File penting:

- `data/trainval/research_patches/manifest.csv` berisi `source_image`, nama patch, koordinat, dan indikator `train_eligible`.
- Tidak ada patch validation yang dibuang sebelum pemisahan source.
- Default filter training: setidaknya 1% piksel vegetasi **dihitung pada area valid non-padding**. Ini bukan formula `sum <=655` yang dipakai skrip legacy; perbedaan wajib dicatat.
- Jika folder riset sebelumnya sudah ada, `--overwrite` hanya menghapus dan menulis ulang `data/trainval/research_patches`. Gunakan hanya bila memang ingin mengulang patch preparation.

Siapkan patch untuk **test dan BBCH** menggunakan utility lama (utility ini juga akan menyimpan patch trainval legacy, tetapi tidak menyentuh `research_patches`):

```powershell
.\.venv\Scripts\python.exe save_patches.py --root_path .
.\.venv\Scripts\python.exe scripts\check_dataset.py --require-patches
```

## Langkah 2 — smoke test tanpa menghabiskan waktu GPU

```powershell
.\.venv\Scripts\python.exe scripts\research_train.py cv --root . --run-id smoke_r34 --architecture unet --encoder resnet34 --folds 2 --trials 1 --epochs 1 --patience 1 --batch-size 2 --workers 0 --loss dice --device cuda
```

Lulus bila `results/research_runs/smoke_r34/cv_summary.json`, `splits.csv`, dan checkpoint tiap fold terbentuk. **Jangan** gunakan angka smoke test di skripsi.

## Langkah 3 — eksperimen pertama: baseline Genze U-Net ResNet34

```powershell
.\.venv\Scripts\python.exe scripts\research_train.py cv --root . --run-id genze_r34_dice_s42 --architecture unet --encoder resnet34 --folds 4 --trials 10 --epochs 150 --patience 10 --batch-size 4 --workers 0 --loss dice --seed 42 --device cuda
.\.venv\Scripts\python.exe scripts\research_train.py final --root . --run-id genze_r34_dice_s42 --workers 0 --device cuda
```

`final` memakai median jumlah epoch terbaik yang ditemukan pada 4 fold. Model akhir dilatih dari bobot pretrained baru pada **seluruh patch trainval yang eligible**, bukan menyalin model salah satu fold. Tidak ada label test yang dibaca.

## Langkah 4 — proposed Zuo & Li tanpa auxiliary

```powershell
.\.venv\Scripts\python.exe scripts\research_train.py cv --root . --run-id zuoli_mnv3_dice_s42 --architecture unet_mobilenetv3 --encoder mobilenetv3_large --folds 4 --trials 10 --epochs 150 --patience 10 --batch-size 4 --workers 0 --loss dice --seed 42 --device cuda
.\.venv\Scripts\python.exe scripts\research_train.py final --root . --run-id zuoli_mnv3_dice_s42 --workers 0 --device cuda
```

Perhatikan bahwa implementasi MobileNetV3 saat ini adalah **adaptasi** dari diagram Zuo & Li, belum terbukti identik per layer/parameter/FLOPs dengan artikel aslinya.

## Langkah 5 — evaluasi held-out test (setelah model dibekukan)

Path checkpoint akhir tercatat di `results/research_runs/<run-id>/final_summary.json`.

Baseline:

```powershell
.\.venv\Scripts\python.exe predict_testset.py "models/research_runs/genze_r34_dice_s42/unet_resnet34_dil0_bilin1_pre1.pth.tar" test --root_path . --batch_size 4 --num_workers 0 --device cuda --model_name genze_r34_dice_s42 --output_dir results/research_runs/genze_r34_dice_s42/predictions/test --report_dir results/research_runs/genze_r34_dice_s42/reports/test
```

Proposed:

```powershell
.\.venv\Scripts\python.exe predict_testset.py "models/research_runs/zuoli_mnv3_dice_s42/unet_mobilenetv3_mobilenetv3_large_dil0_bilin1_pre1.pth.tar" test --root_path . --batch_size 4 --num_workers 0 --device cuda --model_name zuoli_mnv3_dice_s42 --output_dir results/research_runs/zuoli_mnv3_dice_s42/predictions/test --report_dir results/research_runs/zuoli_mnv3_dice_s42/reports/test
```

Ulangi dua perintah dengan mengganti `test` menjadi `test_different_bbch` serta nama folder `.../test` menjadi `.../test_different_bbch`. Jangan retrain pada BBCH.

`predict_testset.py` sekarang memprioritaskan `mean/std` dalam checkpoint riset, sehingga nilai statistik pada inference konsisten dengan final training. Untuk checkpoint lama tanpa metadata, fallback normalisasi lama dipertahankan.

## Langkah 6 — ablation setelah baseline dan proposed selesai

Gunakan run ID terpisah untuk:

1. `unet_mobilenetv3_base` (tanpa PPM/SE)
2. `unet_mobilenetv3_ppm` (PPM, tanpa decoder SE)
3. `unet_mobilenetv3` (PPM + SE; model utama)

Ketiganya menggunakan Dice, seed, grouped folds, trials, dan anggaran epoch yang sama. Perbandingan *weighted CE+Dice* / auxiliary dibuat sebagai eksperimen kedua, **jangan mencampurnya** dalam tabel ablation arsitektur utama.

Untuk menguji auxiliary, bandingkan `unet_mobilenetv3` dengan `ce_dice --class-weights auto` versus `unet_mobilenetv3_aux` dengan `ce_dice_aux_foreground --class-weights auto --aux-weight 0.3`, pada split dan anggaran identik. Jangan mengklaim auxiliary meningkatkan weed IoU kecuali hasil aktual membuktikannya.

## Langkah 7 — tabel laporan

- Uji utama: Pixel Accuracy, mIoU, Mean Dice, Weed IoU, Weed Precision, Weed Recall, Weed F1, confusion matrix, dan metrik per citra.
- Komputasi: parameter, GFLOPs, waktu inference batch=1, FPS, VRAM pada input dan GPU sama.
- Stabilitas: bila sempat, ulangi **training** minimal baseline dan proposed dengan seed 123 dan 2026 serta tampilkan rata-rata ± simpangan baku.
- Kesimpulan harus membedakan hasil CV untuk pemilihan model dari held-out test untuk pelaporan akhir.
- Jangan bandingkan langsung angka FLOPs atau mIoU dari dua artikel dengan definisi input/alat pengukuran yang berbeda.

### Catatan penggunaan

Untuk training yang terhenti gunakan perintah CV identik dengan tambahan `--resume`; `--trials` menyatakan **jumlah trial target total**. Jangan gunakan run ID lama untuk konfigurasi model lain. Script final hanya boleh dijalankan setelah `cv_summary.json` ada.

RTX 5050 8 GB: mulai batch size 4 atau 2 bila CUDA out of memory; bila batch berubah, ulangi eksperimen untuk model pembanding dengan kebijakan batch yang konsisten.

Dokumen ini bukan bukti bahwa training sudah berhasil. Tahap berikutnya adalah menjalankan langkah 0–2 pada PC GPU dan memeriksa `cv_summary.json` serta log CUDA yang dihasilkan.
