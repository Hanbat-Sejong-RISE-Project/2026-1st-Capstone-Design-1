# 0614 Range Mag/Phase Pipeline

이 디렉터리는 `cap_sensing_OFDM_sym64_bw40.mat` 송신 신호를 기준으로 `H=Y/X`를 만들고, people0 평균 background를 제거한 range-domain feature만 사용합니다.

## Pipeline

1. `/workspace/baseline/cap_sensing_OFDM_sym64_bw40.mat`에서 `tx_sym` 로드
2. `tx_sym`을 OFDM symbol 단위로 reshape
3. CP 제거 후 FFT로 송신 `X` 생성
4. people0 모든 raw sample에서 correlation으로 frame 시작점 탐색
5. CP 제거 후 FFT로 `Y0` 생성
6. `H0 = Y0 / X`
7. `H_bg = mean(H0)` 저장
8. people1/2/3에서 `H = Y / X`
9. `H_dyn = H - H_bg`
10. `h_range = IFFT(H_dyn, axis=subcarrier)`
11. `range_mag = abs(h_range)`, `range_phase = angle(h_range)` 저장
12. `[range_mag, range_phase]`를 멀티태스크/멀티헤드 모델 입력으로 사용

## Feature 생성

```bash
conda run -n khj python /workspace/khj/0614/run_make_range_mag_phase.py \
  --dataset_root /workspace/dataset/0529 \
  --tx_mat /workspace/baseline/cap_sensing_OFDM_sym64_bw40.mat \
  --out_mag_feature range_mag \
  --out_phase_feature range_phase \
  --h_bg_path /workspace/dataset/0529/H_bg_tx.npy \
  --people 1 2 3 \
  --rx_channel ch1 \
  --nfft_range 4096 \
  --normalize none \
  --overwrite
```

저장 위치:

```text
/workspace/dataset/0529/H_bg_tx.npy
/workspace/dataset/0529/people1_range_mag/{label}/*_range_mag.npy
/workspace/dataset/0529/people1_range_phase/{label}/*_range_phase.npy
/workspace/dataset/0529/people2_range_mag/{label}/*_range_mag.npy
/workspace/dataset/0529/people2_range_phase/{label}/*_range_phase.npy
/workspace/dataset/0529/people3_range_mag/{label}/*_range_mag.npy
/workspace/dataset/0529/people3_range_phase/{label}/*_range_phase.npy
```

## Dataset 확인

```bash
conda run -n khj python /workspace/khj/0614/test_dataset_rd.py \
  --dataset_root /workspace/dataset/0529 \
  --features range_mag range_phase \
  --people_values 1 2 3 \
  --batch_size 4 \
  --target_size 256 256
```

## 멀티태스크/멀티헤드 학습

```bash
conda run -n khj python /workspace/khj/0614/train_multitask_rd.py \
  --dataset_root /workspace/dataset/0529 \
  --features range_mag range_phase \
  --people_values 1 2 3 \
  --target_size 256 256 \
  --batch_size 16 \
  --epochs 100 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --num_workers 4 \
  --seed 42 \
  --device cuda \
  --run_name multitask_range_mag_phase
```

결과 저장 위치:

```text
/workspace/khj/0614/runs/{run_name}
```

## 테스트 입력/추론 결과 시각화

학습 후 test prediction CSV를 기준으로, 모델 입력에 사용된 `range_mag`/`range_phase`와 target/oracle/routed prediction을 한 이미지에 저장합니다.

```bash
conda run -n khj python /workspace/khj/0614/visualize_multitask_range_inference.py \
  --dataset_root /workspace/dataset/0529 \
  --run_dir /workspace/khj/0614/runs/multitask_range_mag_phase \
  --target_size 256 256 \
  --count 24 \
  --mode mixed
```

저장 위치:

```text
/workspace/khj/0614/runs/multitask_range_mag_phase/inference_visuals
```
