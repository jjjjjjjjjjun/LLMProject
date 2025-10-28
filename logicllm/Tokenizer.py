import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# CUDA 설정 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")
if device.type == "cuda":
    print(f"CUDA 디바이스 이름: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ CUDA를 사용할 수 없습니다. CPU로 실행됩니다.")

# KoGPT 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
model = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2").to(device)
model.eval()

# 통합 CSV 파일 읽기
csv_path = r"/maindata_crowling/all_regions_reviews/all_tourist_reviews.csv"
if not os.path.exists(csv_path):
    print("❌ 통합 CSV 파일이 없습니다.")
    exit()

df = pd.read_csv(csv_path)
if df.empty:
    print("❌ CSV 파일에 데이터가 없습니다.")
    exit()

# 결과를 저장할 리스트
descriptions = []

# 각 지역 및 장소별로 처리
for region in tqdm(df["지역"].unique(), desc="지역 처리"):
    print(f"\n=== {region} ===")
    region_df = df[df["지역"] == region]

    for place in tqdm(region_df["관광지"].unique(), desc=f"{region} 장소 처리", leave=False):
        place_df = region_df[region_df["관광지"] == place]
        reviews = place_df["리뷰"].tolist()[:10]  # 최대 10개 리뷰
        ratings = place_df["평점"].tolist()[:10]

        # 프롬프트 생성 (f-string 대신 str.format 사용)
        prompt = "{}의 {}에 대한 리뷰: {} ### 이를 바탕으로 {}에 대한 간단한 소개글을 작성하세요.".format(
            region,
            place,
            ', '.join(['"{}" (평점: {})'.format(r, s) for r, s in zip(reviews, ratings)]),
            place
        )

        # 토크나이징 및 GPU로 데이터 이동
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # 텍스트 생성
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=600,
                    num_return_sequences=1,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id
                )
        except RuntimeError as e:
            print(f"❌ {place} 텍스트 생성 중 오류: {e}")
            continue

        # 디코딩
        description = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"\n=== {place} 소개 ===")
        print(description)

        # 결과 저장
        descriptions.append({
            "지역": region,
            "관광지": place,
            "소개글": description
        })

# 소개글 CSV 저장
if descriptions:
    save_dir = "../maindata_crowling/descriptions"
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "tourist_descriptions.csv")
    df_desc = pd.DataFrame(descriptions)
    df_desc.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ 모든 소개글 저장 완료: {csv_path}")
else:
    print("❌ 소개글 생성 실패, CSV 생성하지 않음")