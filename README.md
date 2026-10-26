# 모두의 LLM with 파이썬 — 실습 코드

한국어 이야기를 들려주는 작은 챗봇을 처음부터 만드는 실습 노트북과 패키지입니다. 모든 노트북은 Google Colab에서 실행합니다.

## 폴더 구성

| 폴더 | 내용 |
|---|---|
| `notebooks/` | 장별 실습 노트북 `ch01` ~ `ch12` (독자용) |
| `modu_llm/` | 각 장에서 만든 부품의 검증된 버전. 다음 장이 앞 장 코드를 다시 실행하지 않고 불러옵니다 |
| `extras/` | 11장 요청-이야기 데이터를 만든 노트북 `SFT_데이터_만들기.ipynb` (참고용, 실행하지 않아도 됩니다) |

## 장별 노트북

| 장 | 노트북 | GPU | 끝에 저장 |
|---|---|---|---|
| 1 | LLM, 다음 글자를 맞히는 기계 | 불필요 (있으면 2분 동안 더 많이 학습) | — |
| 2 | 데이터와 글자 사전 | 불필요 | `vocab.json` |
| 3 | 다음 글자 맞히기 문제와 임베딩 | 불필요 | — |
| 4 | 셀프 어텐션 | 불필요 | — |
| 5 | 멀티헤드 어텐션 | 불필요 | — |
| 6 | Transformer 블록 | 불필요 | — |
| 7 | 작은 GPT 조립하기 | 불필요 | — |
| 8 | 더 빠르게 (가속) | 속도 측정에 필요 | — |
| 9 | 사전학습 (약 30분) | **필요** | `pretrain.pt`, `ckpt_step*.pt` |
| 10 | 생성과 샘플링 | 불필요 | — |
| 11 | 파인튜닝 (약 10분) | **필요** | `sft.pt` |
| 12 | 이야기 챗봇 | 불필요 | — |

저장물은 구글 드라이브의 `MyDrive/modu_llm/` 폴더에 저장됩니다. 9·11장을 건너뛰면 다음 장에서 저자의 체크포인트를 자동으로 내려받습니다(`modu_llm.get_file`). 저자 체크포인트: [huggingface.co/JosephLee/modu-llm](https://huggingface.co/JosephLee/modu-llm) (CC BY 4.0)

## 모델 설정 (`GPTConfig` 기본값)

글자 사전 2,453자(한글 완성형 2,350자 + 영문·숫자·기호 + 끝 표시 `§`) · 문맥 384글자 · 벡터 길이 256 · 헤드 4개 · 블록 6층 · 약 600만 개 파라미터

## 데이터 출처

- [Tiny-Ko-Stories](https://huggingface.co/datasets/psymon/Tiny-Ko-Stories) (psymon, CC BY 4.0)
- [Tiny-Ko-Stories SFT](https://huggingface.co/datasets/JosephLee/tiny-ko-stories-sft): 11장 요청-이야기 데이터, Tiny-Ko-Stories를 가공 (CC BY 4.0)
- 모델 구조 참고: [nanoGPT](https://github.com/karpathy/nanoGPT) (Andrej Karpathy)

## 라이선스

코드는 MIT 라이선스(`LICENSE`)를 따릅니다. 저자 체크포인트와 데이터는 CC BY 4.0이며, 쓸 때 위 출처를 함께 밝혀 주세요.
