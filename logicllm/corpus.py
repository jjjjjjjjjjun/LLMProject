import pandas as pd

# CSV 파일 로드
csv_path = r"/mainprogram/descriptions/tourist_descriptions.csv"
df = pd.read_csv(csv_path)

# 데이터 점검
print("원본 데이터 크기:", len(df))
print("열 이름:", df.columns.tolist())
print("샘플 데이터:\n", df.head(3))

# 데이터 청소
df = df.dropna(subset=["소개글", "지역", "관광지"])  # NaN 제거
df = df[df["소개글"].str.strip() != ""]  # 빈 문자열 제거
df["소개글"] = df["소개글"].str.replace(r"리뷰|소개|에 대한", "", regex=True).str.strip()  # 반복 패턴 제거
df["소개글"] = df["소개글"].str.replace(r"[^\w\s.,!?]", "", regex=True)  # 특수 문자 제거
df["소개글"] = df["소개글"].str.replace(r"\s+", " ", regex=True).str.strip()  # 다중 공백 제거
df = df[df["소개글"].str.len() > 20]  # 너무 짧은 텍스트 제거
print(f"청소 후 데이터 크기: {len(df)} 행")

# 안동 하회마을 확인
hahoe = df[(df["지역"] == "안동시") & (df["관광지"] == "안동 하회마을")]
print("안동 하회마을 데이터:\n", hahoe)

# 정제된 CSV 저장
cleaned_csv_path = csv_path.replace(".csv", "_cleaned.csv")
df.to_csv(cleaned_csv_path, index=False)
print(f"정제된 CSV 저장: {cleaned_csv_path}")