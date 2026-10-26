# 모두의 LLM with 파이썬 — 실습 코드

한국어 이야기를 들려주는 작은 챗봇을 처음부터 만드는 실습 노트북과 패키지입니다. 모든 노트북은 Google Colab에서 실행합니다.

## 폴더 구성

| 폴더 | 내용 |
|---|---|
| `notebooks/` | 장별 실습 노트북 `ch01` ~ `ch12` (독자용) |
| `modu_llm/` | 각 장에서 직접 만든 부품과 같은 코드(글자 사전, Embeddings, Head, MultiHead, Block, GPT)와 도우미 함수. 다음 장이 앞 장 코드를 다시 실행하지 않고 불러옵니다 |
| `extras/` | 11장 요청-이야기 데이터를 만든 노트북 `SFT_데이터_만들기.ipynb` (참고용, 실행하지 않아도 됩니다) |

## 장별 노트북

| 장 | 노트북 | 직접 만드는 것 | GPU | 끝에 저장 |
|---|---|---|---|---|
| 1 | LLM, 다음 글자를 맞히는 기계 | 맛보기(실행하고 관찰) | 불필요 (있으면 2분 동안 더 많이 학습) | — |
| 2 | 데이터와 글자 사전 | 글자 사전, encode, decode | 불필요 | — |
| 3 | 다음 글자 맞히기 문제와 임베딩 | 한 칸 밀기, Embeddings | 불필요 | — |
| 4 | 셀프 어텐션 | Q·K·V, 마스크, softmax, Head | 불필요 | — |
| 5 | 멀티헤드 어텐션 | MultiHead | 불필요 | — |
| 6 | Transformer 블록 | MLP, LayerNorm, 잔차, Block | 불필요 | — |
| 7 | 작은 GPT 조립하기 | 배치, GPT 조립, 손실 | 불필요 | — |
| 8 | 더 빠르게 (가속) | 헤드를 한 번에 계산하는 어텐션 | 속도 측정에 필요 | — |
| 9 | 사전학습 (약 30분) | 학습 한 단계, 이어 학습 | **필요** | `pretrain.pt`, `ckpt_step*.pt`, `tokens_200000.npy` |
| 10 | 생성과 샘플링 | 확률대로 뽑기, 온도 | 불필요 | — |
| 11 | 파인튜닝 (약 10분) | 요청 형식, 이어 학습 | **필요** | `sft.pt` |
| 12 | 이야기 챗봇 | chat 함수 | 불필요 | — |

저장물은 구글 드라이브의 `MyDrive/modu_llm/` 폴더에 저장됩니다. 9·11장을 건너뛰면 다음 장에서 저자의 체크포인트를 자동으로 내려받습니다(`modu_llm.get_file`). 저자 체크포인트: [huggingface.co/JosephLee/modu-llm](https://huggingface.co/JosephLee/modu-llm) (CC BY 4.0)

## 모델 설정 (`GPTConfig` 기본값)

글자 사전 2,453자(한글 완성형 2,350자 + 영문·숫자·기호 + 끝 표시 `§`) · 문맥 384글자 · 벡터 길이 256 · 헤드 4개 · 블록 6층 · 약 600만 개 파라미터

## 데이터 출처

- [Tiny-Ko-Stories](https://huggingface.co/datasets/psymon/Tiny-Ko-Stories) (psymon, CC BY 4.0)
- [Tiny-Ko-Stories SFT](https://huggingface.co/datasets/JosephLee/tiny-ko-stories-sft): 11장 요청-이야기 데이터, Tiny-Ko-Stories를 가공 (CC BY 4.0)
- 모델 구조 참고: [nanoGPT](https://github.com/karpathy/nanoGPT) (Andrej Karpathy)

## 라이선스

코드는 MIT 라이선스(`LICENSE`)를 따릅니다. 저자 체크포인트와 데이터는 CC BY 4.0이며, 쓸 때 위 출처를 함께 밝혀 주세요.
