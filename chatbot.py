import pandas as pd
import Levenshtein  # 레벤슈타인 거리 계산을 위한 라이브러리

class SimpleChatBot:
    def __init__(self, filepath):
        # CSV 파일로부터 질문과 답변 데이터를 불러온다
        self.questions, self.answers = self.load_data(filepath)

    def load_data(self, filepath):
        # CSV 파일을 읽어 Pandas DataFrame으로 불러옴
        data = pd.read_csv(filepath)
        questions = data['Q'].tolist()  # 질문 리스트
        answers = data['A'].tolist()    # 답변 리스트
        return questions, answers

    def find_best_answer(self, input_sentence):
        # 모든 질문과의 레벤슈타인 거리 계산
        distances = []
        for i, question in enumerate(self.questions):
            distance = Levenshtein.distance(input_sentence, question)
            distances.append((i, distance))  # (인덱스, 거리)

        # 가장 유사한 질문(거리 가장 짧은 것)의 인덱스 선택
        best_match_index, best_distance = min(distances, key=lambda x: x[1])

        # 가장 유사한 질문에 대응하는 답변 반환
        return self.answers[best_match_index]

# CSV 파일 경로
filepath = 'ChatbotData.csv'

# 챗봇 인스턴스 생성
chatbot = SimpleChatBot(filepath)

# 사용자 입력 루프
while True:
    input_sentence = input('You: ')
    if input_sentence.lower() == '종료':
        break
    response = chatbot.find_best_answer(input_sentence)
    print('Chatbot:', response)
