#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cgi
import cgitb
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 에러 출력 활성화
cgitb.enable()

# 모델 및 토크나이저 설정 (이미 학습된 모델 불러오기)
model_name = "klue/bert-base"  # 모델 이름 설정
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 웹 양식에서 데이터를 받기 위한 설정
form = cgi.FieldStorage()

# HTML 출력 (폼 및 결과 표시)
print("Content-Type: text/html; charset=cp949")  
print()  
print("""
<html><head>
<meta charset="cp949">
<title>악성 댓글 판별기</title>
</head><body align="center">
<h1>악성 댓글 판별기</h1>
""")

# 댓글 입력폼
print("""
<form method="post" action="/cgi-bin/toxic.py">
    판독해 보고 싶은 댓글을 입력하세요!  <br><br>
    <textarea name="sentence" rows="1" cols="50"></textarea><br><br>
    <input type="submit" value="분석하기">
</form>
""")

# 댓글 예측 결과
sentence = form.getvalue("sentence")
if sentence:
    # predict 함수 호출
    def predict(sentence):
        model.eval()
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).item()

        if pred == 0:
            return "정상 댓글입니다."
        else:
            return "악성 댓글이 탐지되었습니다."

    result = predict(sentence)
    print(f"<h2>결과: {result}</h2>")

print("</body></html>")


'''
python -m http.server --cgi 8060

http://localhost:8060/cgi-bin/toxic.py
'''