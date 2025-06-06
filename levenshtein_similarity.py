import Levenshtein

# 1. 학습 데이터: 질문과 답변 쌍
questions = [
    "안녕",
    "너 이름이 뭐야?",
    "오늘 날씨 어때?",
    "밥 먹었니?",
    "학교에 갔다 왔어"
]

answers = [
    "안녕하세요!",
    "저는 챗봇이에요.",
    "오늘 날씨는 맑고 따뜻해요.",
    "아직 안 먹었어요. 당신은요?",
    "학교는 재미있었나요?"
]

# 2. 사용자의 입력 문장 (챗)
chat = "너 학교에 다녀왔니?"

# 3. 입력 문장과 학습 질문들 간의 레벤슈타인 거리 계산
distances = []
for i, q in enumerate(questions):
    distance = Levenshtein.distance(chat, q)
    distances.append((i, distance))  # (인덱스, 거리)

# 4. 가장 유사한 질문(=가장 작은 거리)을 찾음
best_match_index, best_distance = min(distances, key=lambda x: x[1])

# 5. 결과 출력
print(f"[사용자 질문] {chat}")
print(f"[가장 유사한 학습 질문] {questions[best_match_index]}")
print(f"[답변] {answers[best_match_index]}")
print(f"[레벤슈타인 거리] {best_distance}")
