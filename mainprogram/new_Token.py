import pandas as pd
import re
from tqdm import tqdm
import random

# CSV 로드
input_csv = r"C:\Users\user\PycharmProjects\LLMProject\mainfile\all_regions_reviews\all_tourist_reviews.csv"
df = pd.read_csv(input_csv)

# 데이터 점검
print("원본 데이터 크기:", df.shape)
print("열 이름:", df.columns.tolist())
print("샘플 데이터:\n", df.head(3))
print("\n평점 분포 (변환 전):\n", df['평점'].value_counts().sort_index())

# 평점 열을 숫자형으로 변환
df['평점'] = pd.to_numeric(df['평점'], errors='coerce')  # 문자열을 숫자로, 실패 시 NaN
print("\n평점 변환 후 데이터 타입:", df['평점'].dtype)
print("평점 분포 (변환 후):\n", df['평점'].value_counts().sort_index(dropna=False))

# 데이터 청소
df = df.dropna(subset=["리뷰", "지역", "관광지", "평점"])  # NaN 제거
df = df[df["리뷰"].str.strip() != ""]  # 빈 리뷰 제거
df = df[df['평점'] >= 4]  # 평점 4 이상만 사용
print(f"평점 필터링 후 데이터 크기: {len(df)} 행")

# 리뷰 텍스트 청소 및 변환
def clean_and_transform_review(review_text):
    if pd.isna(review_text):
        return ""
    # 줄바꿈을 마침표로 변환
    review_text = re.sub(r'\n+', '. ', str(review_text))
    # 반복 패턴 제거
    review_text = re.sub(r"리뷰|소개|에 대한|관광지입니다|방문했는데", "", review_text, flags=re.IGNORECASE)
    # 특수 문자 제거
    review_text = re.sub(r"[^\w\s.,!?가-힣ㄱ-ㅎㅏ-ㅣ]", "", review_text)
    # 다중 공백 제거
    review_text = re.sub(r"\s+", " ", review_text).strip()
    # 문장 단위로 분리 후 필터링
    sentences = [s.strip() for s in review_text.split('. ') if len(s.strip()) > 10]
    if not sentences:
        return ""
    # 긍정적인 문장 선택
    positive_sentences = [s for s in sentences if not any(word in s.lower() for word in ['아쉬웠', '불편', '문제', '어려웠'])]
    return ". ".join(positive_sentences[:3]) + "." if positive_sentences else ""

tqdm.pandas(desc="리뷰 청소 및 변환")
df['소개글'] = df['리뷰'].progress_apply(clean_and_transform_review)
df = df[df['소개글'] != ""]  # 빈 소개글 제거
print(f"청소 후 데이터 크기: {len(df)} 행")

# 데이터 증강
augmented_data = []
for _, row in df.iterrows():
    region, place, desc = row['지역'], row['관광지'], row['소개글']
    # 원본 추가
    augmented_data.append({"지역": region, "관광지": place, "소개글": desc})
    # 증강 1: 문장 순서 섞기
    sentences = desc.split(". ")
    if len(sentences) > 1:
        random.shuffle(sentences)
        augmented_desc = ". ".join(sentences)
        augmented_data.append({"지역": region, "관광지": place, "소개글": augmented_desc})
    # 증강 2: 간단히 재구성
    augmented_desc = f"{place}는 {region}의 대표적인 명소로, {desc[:50]}..."
    augmented_data.append({"지역": region, "관광지": place, "소개글": augmented_desc})

# 증강된 데이터프레임
augmented_df = pd.DataFrame(augmented_data)
print(f"증강 후 데이터 크기: {len(augmented_df)} 행")
print("증강 샘플:\n", augmented_df.head(5))

# 코퍼스 저장
corpus_csv = input_csv.replace("all_tourist_review.csv", "tourist_corpus.csv")
augmented_df.to_csv(corpus_csv, index=False)
print(f"코퍼스 저장: {corpus_csv}")