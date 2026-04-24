> **참고:** 이 한국어 버전은 초판 번역으로, 최신 기능(`elsa_strict_ref.py`, `elsa_ext_pack/`, `run_strict_coverage_matrix.py` 등)이 반영되지 않을 수 있습니다. 최신 내용은 [영어 README](../README.md)를 참조하세요.

# ELSA: 빠르고 메모리 효율적인 Vision Transformer를 위한 정확한 선형 스캔 어텐션

<p align="center">
  <a href="https://ming053l.github.io/ELSA_projectpage/">[프로젝트 페이지]</a> &nbsp;|&nbsp;
  <a href="#">[논문 (CVPR 2026)]</a> &nbsp;|&nbsp;
  <a href="#인용">[인용]</a>
</p>

<p align="center">
  <a href="../README.md">English</a> &nbsp;|&nbsp;
  <a href="README.zh-TW.md">繁體中文</a> &nbsp;|&nbsp;
  <a href="README.zh-CN.md">简体中文</a> &nbsp;|&nbsp;
  <a href="README.ja.md">日本語</a>
</p>

---

<div align="center">

**<a href="https://cchsu.info/wordpress/">Chih-Chung Hsu</a>, Xin-Di Ma, Wo-Ting Liao, <a href="https://ming053l.github.io/">Chia-Ming Lee</a>**

Advanced Computer Vision Laboratory, National Yang Ming Chiao Tung University

CVPR Findings 2026

*"Can't use FA on your device? Try ELSA on!"**

</div>

---

## 개요

ELSA는 softmax 어텐션을 상태 트리플 *(m, S, W)*의 결합 모노이드 위에서의 **병렬 프리픽스 스캔**으로 재구성하여 다음을 달성합니다:

- **정확한 softmax 의미론** — 증명 가능한 FP32 상대 오차 상한, 재학습 불필요
- **O(log n) 병렬 깊이** — 2단계 스캔 (블록 내 Hillis–Steele + 블록 간 Blelloch)
- **O(n) 추가 메모리** — O(n²) 스코어 행렬 불필요, 쿼리당 1회 I/O 패스
- **Tensor Core 독립적** — Triton 및 CUDA C++로 구현, A100·L4·Jetson TX2에서 동작

---

## 기능과 응용

### 이 저장소가 제공하는 것

| | |
|---|---|
| **Triton / CUDA 커널** | `ELSA_triton` (FP16/BF16), `ELSA_triton_fp32` (추론), `ELSA_triton_fp32_train` (학습+역전파) |
| **PyTorch 모듈** | `ElsaAttention` — 임의의 표준 어텐션 블록을 대체하는 `nn.Module` |
| **전체 모델 클래스** | `ElsaViT`, `ElsaSwinTransformerV2` — 바로 학습 가능한 아키텍처 |
| **패치 유틸리티** | `patch_vit_attention`, `patch_swin_attention` — 코드 변경 없이 timm 모델 가속화 |
| **벤치마크 하네스** | `fairbench_driver` / `fairbench_worker` — 재현 가능한 멀티 백엔드 속도·VRAM 비교 |

### 예상 응용 분야

- **고해상도 비전** — FP32 정밀도가 필요한 긴 시퀀스 ViT 추론 (의료 영상·위성 영상·하이퍼스펙트럴 분석)
- **메모리 제한 환경 배포** — 표준 SDPA로는 OOM이 발생하는 32K+ 토큰 LLM 추론을 일반 GPU에서 실행
- **임베디드 / 엣지 AI** — Tensor Core 독립적 설계로 Jetson TX2 등 디바이스에서도 동작
- **로보틱스 및 자율 주행** — 예산이 제한된 차량 탑재·로봇용 컴퓨팅 플랫폼(AGX Orin, Drive AGX)에서의 실시간 인식; O(n²) 어텐션은 레이턴시나 전력 소비의 병목이 되기 쉬우며, ELSA의 O(n) 메모리 풋프린트는 동일한 VRAM 용량 내에서 더 큰 컨텍스트 윈도우를 사용할 수 있게 해 하드웨어 업그레이드 없이도 풍부한 장면 표현을 실현
- **3D 장면 이해** — VGGT 등 멀티프레임 모델을 정밀도 손실 없이 가속화
- **임의의 커스텀 Transformer** — ELSA 커널은 원시 Q/K/V 텐서를 받아 어떤 아키텍처에도 통합 가능

### 제공된 사용 예시

| 사용 예시 | 위치 |
|---|---|
| Raw Q/K/V 커널 대체 | [사용 방법 — 레벨 1](#레벨-1--raw-커널커스텀-attention-클래스) |
| `ElsaAttention`을 사용한 커스텀 `TransformerBlock` | [사용 방법 — 레벨 2](#레벨-2--elsaattention-모듈단일-레이어-대체) |
| 사전 학습된 timm ViT / Swin 패치 | [사용 방법 — 레벨 3](#레벨-3--사전-학습된-timm-vit--swin-패치모델-코드-변경-없음) |
| `ElsaViT`를 처음부터 구축 | [사용 방법 — 레벨 4](#레벨-4--새로운-elsavit를-처음부터-구축) |
| HuggingFace LLaMA 패치 + 긴 컨텍스트 FP32 | [사용 방법 — 레벨 5](#레벨-5--llama--huggingface-언어-모델) |

---

## 성능 하이라이트

### FP32 추론 vs. SDPA-Math (CLIP ViT, A100)

| 모델 | 해상도 | 속도 향상 | 메모리 절감 |
|---|---|---|---|
| ViT-B/16 | 224→560 px | 최대 **1.98×** | 최대 **36.1%** |
| ViT-L/14 | 224→560 px | 최대 **2.15×** | 최대 **39.6%** |

### FP32 학습 (ViT, A100, batch=1/2, image=1024)

| 비교 대상 | 중앙값 속도 향상 | 중앙값 VRAM 비율 |
|---|---|---|
| ELSA vs SDPA-Math | **1.72×** | **0.23×** (−77%) |
| ELSA vs SDPA-Mem | **1.09×** | **1.05×** |

### 하이퍼스펙트럴 이미지 분류 (HSI-MAE, FP32)

| 모델 | 데이터셋 | ME-SDPA 대비 속도 |
|---|---|---|
| HSI-MAE-B | Pavia / Salinas / WHU | +37–40% |
| HSI-MAE-L | Pavia / Salinas / WHU | **+60–62%** |

### 임베디드 디바이스 (Jetson TX2, FP16)

전체 토큰 길이 (64–900 tokens)에 걸쳐 Math-SDPA 대비 안정적으로 **~37% 레이턴시 감소**.

### 3D 재구성 (VGGT, FP32 vs xFormers)

| 프레임 수 | 속도 향상 |
|---|---|
| 50 | **1.46×** |
| 100 | **2.09×** |
| 150 | **2.34×** |

---

## 방법론

ELSA는 온라인 softmax를 모노이드 `(m, S, W) ∈ ℝ × ℝ × ℝ^dv` 위의 프리픽스 스캔으로 변환합니다:

```
m  = 실행 중 최대 logit
S  = 정규화된 누적 exp 가중치 합
W  = exp 가중 값의 누산기
```

병합 연산자 ⊕는 3단계로 블록을 합성합니다: **역정규화 → 집계 → 재정규화**. 이를 통해:

| 특성 | 값 |
|---|---|
| 병렬 깊이 | O(log n) |
| 추가 메모리 | O(n) |
| 쿼리당 I/O | 1패스 (K, V를 각 1회 스트림) |
| FP32 오차 상한 | O(u · log n) |

---

## 지원 모델과 하드웨어

**비전:**
ViT (Tiny / Small / Medium / Base / Large), Swin Transformer, CLIP, SAM, VGGT, HSI-MAE

**언어:**
LLaMA (8B, 13B), BERT

**하드웨어:**
NVIDIA A100, L4, Jetson TX2 · 모든 CUDA 지원 GPU (Tensor Core 독립적)

---

## 다른 어텐션 커널과의 비교

| 방법 | 정확 | FP32 네이티브 | GPU 독립적 | 재학습 불필요 | 깊이 | 추가 메모리 |
|---|---|---|---|---|---|---|
| Standard SDPA | ✓ | ✗ | ✓ | ✓ | O(n) | O(n²) |
| FlashAttention-2/3 | ✓ | ✗ | ✗ | ✓ | O(n/Tk) | O(Tk·d) |
| Linear Attention | ✗ | ✓ | ✓ | ✗ | O(log n)† | O(n) |
| **ELSA (제안 방법)** | **✓** | **✓** | **✓** | **✓** | **O(log n)** | **O(n)** |

---

## 저장소 구조

```
code/
  stable/          # 프로덕션 준비 ELSA 커널 및 모델 통합
  future_exp/      # 실험적 경로 (최종 성능 주장에 사용되지 않음)
scripts/           # 재현 가능한 벤치마크·검증 스크립트
docs/              # 릴리스 노트, 상태 매트릭스, 재현성 가이드, 전체 보고서
results/           # 선별된 CSV 출력
validation/        # 빠른 검증 출력
manifests/         # 파일 매니페스트와 SHA256 체크섬
```

주요 소스 파일:

| 파일 | 용도 |
|---|---|
| `code/stable/elsa_triton.py` | Triton 어텐션 커널 |
| `code/stable/elsa.py` | `ElsaAttention` / `ElsaViT` 모델 클래스 |
| `code/stable/elsa_swin.py` | Swin Transformer 통합 |
| `code/stable/elsa_swin_fused.py` | 퓨전 Swin 커널 경로 |

---

## 설치

```bash
git clone https://github.com/your-org/elsa.git
cd elsa
pip install -e .           # elsa 패키지를 편집 가능 모드로 설치
```

> 의존성 (`torch`, `triton`, `timm`)은 자동으로 설치됩니다.
> 벤치마크 유틸리티가 필요한 경우 `[benchmark]` 옵션을 추가하세요:
> `pip install -e ".[benchmark]"`

---

## 사용 방법

### 레벨 1 — Raw 커널 (커스텀 attention 클래스)

Q, K, V 텐서가 있고 어텐션 계산만 대체하고 싶은 경우:

```python
import torch
from elsa import ELSA_triton, ELSA_triton_fp32, ELSA_pytorch

B, H, N, D = 2, 12, 1024, 64   # batch, heads, seq_len, head_dim
q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
scale = D ** -0.5

# FP16 / BF16 — 가장 빠른 경로
out = ELSA_triton.apply(q, k, v, scale)

# FP32 추론 — 메모리 효율적, 증명 가능한 정확도
out = ELSA_triton_fp32.apply(q.float(), k.float(), v.float(), scale)

# FP32 학습 (역전파 지원)
out = ELSA_triton_fp32_train.apply(q.float(), k.float(), v.float(), scale)

# 순수 PyTorch 폴백 — Triton 불필요, 완전한 autograd 지원
out = ELSA_pytorch(q, k, v, scale)
```

다음 표준 PyTorch 구현의 직접 대체품으로 사용할 수 있습니다:

```python
# 대체 전
out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)
```

---

### 레벨 2 — `ElsaAttention` 모듈 (단일 레이어 대체)

`ElsaAttention`은 완전한 `nn.Module` (QKV 프로젝션 → ELSA 커널 → 출력 프로젝션)로, `timm.models.vision_transformer.Attention`과 동일한 인터페이스를 가집니다:

```python
import torch
import torch.nn as nn
from elsa import ElsaAttention

class MyTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ElsaAttention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=0.0,
            proj_drop=0.0,
            backend="auto",
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x):            # x: (B, N, dim)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

model = MyTransformerBlock(dim=768, num_heads=12).cuda()
x = torch.randn(2, 196, 768, device="cuda")
out = model(x)    # (2, 196, 768)
```

사용 가능한 `backend` 옵션:

| 값 | 설명 |
|---|---|
| `"auto"` | **권장.** dtype과 하드웨어에 따라 최적 커널을 자동 선택 |
| `"triton"` | ELSA Triton 커널 — FP16 / BF16 |
| `"triton_fp32"` | ELSA Triton 커널 — FP32 추론 |
| `"triton_fp32_train"` | ELSA Triton 커널 — 역전파 포함 FP32 학습 |
| `"sdpa_math"` / `"sdpa_mem"` / `"sdpa_flash"` | PyTorch SDPA 백엔드 (기준선 비교용) |
| `"pytorch"` | 순수 PyTorch 폴백 |

런타임에 전역으로 백엔드를 전환할 수도 있습니다:

```python
from elsa import set_default_elsa_backend
set_default_elsa_backend("triton_fp32")
```

---

### 레벨 3 — 사전 학습된 timm ViT / Swin 패치 (모델 코드 변경 없음)

```python
import timm
from fairbench_worker import patch_vit_attention, patch_swin_attention

# ── ViT ──────────────────────────────────────────────────────────────────────
model = timm.create_model("vit_base_patch16_224", pretrained=True).cuda().eval()
patch_vit_attention(model, method="elsa", precision="fp32", full_model_mode=True)

with torch.no_grad():
    out = model(images)

# ── Swin Transformer ─────────────────────────────────────────────────────────
swin = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True).cuda().eval()
patch_swin_attention(swin, method="elsa", precision="fp32", full_model_mode=True)

with torch.no_grad():
    out = swin(images)
```

---

### 레벨 4 — 새로운 `ElsaViT`를 처음부터 구축

```python
import torch
from elsa import ElsaViT

model = ElsaViT(
    img_size=224, patch_size=16, embed_dim=768,
    depth=12, num_heads=12, num_classes=1000,
    elsa_backend="auto",
).cuda()

logits = model(torch.randn(4, 3, 224, 224, device="cuda"))   # (4, 1000)
```

---

### 레벨 5 — LLaMA / HuggingFace 언어 모델

```python
import types, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from elsa import ELSA_triton, ELSA_triton_fp32

def patch_llama_attention(model):
    def _make_forward(original_forward):
        def elsa_forward(self, hidden_states, attention_mask=None,
                         position_ids=None, past_key_value=None,
                         output_attentions=False, use_cache=False, **kwargs):
            _real_sdpa = torch.nn.functional.scaled_dot_product_attention
            def _elsa_sdpa(q, k, v, attn_mask=None, dropout_p=0.0,
                           is_causal=False, scale=None, **kw):
                s = scale if scale is not None else q.shape[-1] ** -0.5
                if q.dtype == torch.float32:
                    return ELSA_triton_fp32.apply(q, k, v, s)
                return ELSA_triton.apply(q, k, v, s)
            torch.nn.functional.scaled_dot_product_attention = _elsa_sdpa
            try:
                return original_forward(self, hidden_states,
                    attention_mask=attention_mask, position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache, **kwargs)
            finally:
                torch.nn.functional.scaled_dot_product_attention = _real_sdpa
        return elsa_forward
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.forward = types.MethodType(_make_forward(attn.__class__.forward), attn)
    return model

model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="cuda"
)
patch_llama_attention(model)

inputs = tokenizer("The quick brown fox", return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

**FP32 긴 시퀀스 추론 (메모리 제한 GPU)**

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float32, device_map="cuda"
)
patch_llama_attention(model)

long_input = tokenizer(
    very_long_document, return_tensors="pt", max_length=32768, truncation=True
).to("cuda")
with torch.no_grad():
    out = model(**long_input)
```

> **퍼플렉서티 동등성:** ELSA는 정확한 softmax 의미론을 유지합니다.
> WikiText-2에서 측정한 퍼플렉서티는 원본 모델과 부동소수점 정밀도로 일치합니다.

---

### 레벨 선택 가이드

| 상황 | 권장 |
|---|---|
| Q, K, V를 수동으로 계산하는 커스텀 `nn.Module` | **레벨 1** — `ELSA_triton` / `ELSA_triton_fp32` |
| 모델 내 단일 어텐션 블록 대체 | **레벨 2** — `ElsaAttention` |
| 사전 학습된 timm ViT / Swin, 모델 변경 없음 | **레벨 3** — `patch_vit_attention` |
| 새로운 ViT를 처음부터 학습 | **레벨 4** — `ElsaViT` |
| HuggingFace LLaMA / 디코더형 LLM | **레벨 5** — `F.scaled_dot_product_attention` 수동 대체 |

---

## 동작 요건

- Python ≥ 3.9
- PyTorch ≥ 2.1 (CUDA 빌드)
- Triton ≥ 2.2
- timm ≥ 0.9

```bash
pip install -r requirements.txt
```


---

## 환경 변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `ELSA_TRITON_FP32_TRAIN_BWD` | `auto` | FP32 학습 역전파 경로 (`triton`은 `ELSA_TRITON_ALLOW_UNSTABLE_PATHS=1` 필요) |
| `ELSA_TRITON_FP32_MEM_SAVE_OUT` | `1` | `1` = 속도 우선; `0` = VRAM 절감 |
| `ELSA_TRITON_ALLOW_UNSTABLE_PATHS` | `0` | `1`로 실험적 경로 활성화 |
| `ELSA_FORCE_ALLOW_TF32` | `0` | TF32 정책 오버라이드 |

---

## 문서

| 문서 | 설명 |
|---|---|
| [`docs/FULL_REPORT_20260301.md`](docs/FULL_REPORT_20260301.md) | 전체 설정의 완전한 벤치마크 결과 |
| [`docs/RELEASE_NOTES_20260301.md`](docs/RELEASE_NOTES_20260301.md) | 주요 업데이트 및 안정성 수정 |
| [`docs/STATUS_MATRIX_20260301.md`](docs/STATUS_MATRIX_20260301.md) | 설정별 지원 상태 |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | 공정성 제어 포함 재현 가이드 |

---

## 인용

ELSA를 연구에 사용하는 경우 다음을 인용해 주세요:

```bibtex
@inproceedings{hsu2026elsa,
  title={ELSA: Exact Linear-Scan Attention for Fast and Memory-Light Vision Transformers},
  author={Hsu, Chih-Chung and Ma, Xin-Di and Liao, Wo-Ting and Lee, Chia-Ming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

---

## 라이선스

본 프로젝트는 **학술 연구 및 비상업적 목적으로만** 공개되었습니다.
자세한 내용은 [LICENSE](LICENSE)를 참조하세요.
