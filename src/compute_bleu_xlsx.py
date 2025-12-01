import pandas as pd
import sacrebleu

df = pd.read_excel("evaluation.xlsx")

df = df.fillna("")

bleu_scores = []

for index, row in df.iterrows():
    model_answer = row["Model_Answer"]
    real_answer = row["Real_Answer"]

    if not model_answer.strip() or not real_answer.strip():
        bleu_scores.append(0)
    else:
        score = sacrebleu.sentence_bleu(model_answer, [real_answer]).score
        bleu_scores.append(score)

df["BLEU_Score"] = bleu_scores


df.to_excel("evaluation_with_bleu.xlsx", index=False)

print("BLEU scores added! Check evaluation_with_bleu.xlsx")
