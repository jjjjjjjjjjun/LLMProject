import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

# 모델 및 토크나이저 로드
model_path = "../LLMtuning/kogpt2_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer.pad_token = tokenizer.eos_token

# CSV 데이터 로드
csv_path = r"/maindata_crowling/all_regions_reviews/all_tourist_reviews.csv"
try:
    df = pd.read_csv(csv_path)
    print("CSV 로드 성공. 열 이름:", df.columns.tolist())
except FileNotFoundError:
    print("❌ CSV 파일 없음. 추천 기능은 모델만으로 동작.")
    df = None

# 추론 함수
def generate_response(user_input):
    # 입력 전처리
    user_input = user_input.strip()
    prompt = f"사용자: {user_input} AI:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 모델 생성
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ai_response = response.split("AI:")[-1].strip()

    # CSV 기반 추가 정보 (지역만 입력 시 추천, 관광지 포함 시 정보)
    if df is not None:
        # 지역만 입력된 경우 (예: "서울")
        if len(user_input.split()) == 1:
            region = user_input
            region_data = df[df['지역'].str.contains(region, case=False, na=False)]
            if not region_data.empty:
                # 평점 기준 top 3
                region_data['평점'] = pd.to_numeric(region_data['평점'], errors='coerce').fillna(0)
                top_attractions = region_data.sort_values('평점', ascending=False).head(3)
                rec_response = f"{region} 추천 관광지:\n"
                for _, row in top_attractions.iterrows():
                    review = str(row['리뷰'])[:100] + "..." if len(str(row['리뷰'])) > 100 else str(row['리뷰'])
                    rec_response += f"- {row['관광지']} (평점 {row['평점']}, 리뷰: {review})\n"
                return rec_response
        # 지역+관광지 입력된 경우 (예: "제주 한라산")
        else:
            region, attraction = user_input.split()[:2]
            attraction_data = df[(df['지역'].str.contains(region, case=False, na=False)) &
                               (df['관광지'].str.contains(attraction, case=False, na=False))]
            if not attraction_data.empty:
                row = attraction_data.iloc[0]  # 첫 번째 매칭
                review = str(row['리뷰'])[:200] + "..." if len(str(row['리뷰'])) > 200 else str(row['리뷰'])
                return f"{region}의 {row['관광지']}은 평점 {row['평점']}점입니다. 리뷰: {review}"

    # CSV 없거나 매칭 안 되면 모델 답변 그대로
    return ai_response

# 대화 루프
print("대화 시작! 'exit' 입력으로 종료.")
while True:
    user_input = input("사용자: ")
    if user_input.lower() == "exit":
        print("대화 종료!")
        break
    response = generate_response(user_input)
    print(f"AI: {response}")