"""어텐션 비율을 표로 보여 주는 도우미 (4~6장)."""


def show_weights(w, labels, digits=2):
    """어텐션 비율 표: 줄 = 지금 글자, 칸 = 참고하는 글자.
    w의 모양은 (배치, T, T)이고, 배치의 첫 번째만 보여 줍니다."""
    import pandas as pd
    table = w[0].detach().cpu().numpy().round(digits)
    return pd.DataFrame(table, index=labels, columns=labels)
