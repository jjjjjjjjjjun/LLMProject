import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm

# CUDA 설정 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

# 파인튜닝된 모델 및 토크나이저 로드
model_path = "../LLMtuning/kogpt2_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    use_safetensors=True,
    dtype=torch.float32
).to(device)

# 패딩 토큰 설정
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = 1
model.config.pad_token_id = 1
print(
    f"Vocab size: {tokenizer.vocab_size}, EOS token ID: {tokenizer.eos_token_id}, Pad token ID: {tokenizer.pad_token_id}")


# 텍스트 생성 함수
def generate_description(region, place, max_new_tokens=200):
    prompt = f"{region}의 {place}에 대한 간단한 소개글을 자연스럽게 작성하세요:"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding=True
    ).to(device)

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.9,  # 더 집중된 샘플링
        temperature=0.6,  # 더 일관된 출력
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return generated_text.strip()


# 데이터셋에서 샘플 테스트
def test_with_dataset(csv_file, num_samples=5):
    df = pd.read_csv(csv_file)
    print(f"\n데이터셋에서 {num_samples}개 샘플로 테스트 시작...")
    for idx, row in enumerate(tqdm(df.sample(num_samples).itertuples(), total=num_samples, desc="테스트")):
        region = row.지역
        place = row.관광지
        original_desc = row.소개글
        generated_desc = generate_description(region, place, max_new_tokens=200)
        print(f"\n샘플 {idx + 1}:")
        print(f"입력: {region}의 {place}")
        print(f"원본 소개글: {original_desc[:100]}...")
        print(f"생성된 소개글: {generated_desc}")


# 사용자 입력으로 테스트
def interactive_test():
    print("\n인터랙티브 테스트 시작! (종료하려면 'exit' 입력)")
    while True:
        region = input("지역을 입력하세요 (예: 서울특별시): ")
        if region.lower() == 'exit':
            break
        place = input("관광지 이름을 입력하세요 (예: 경복궁): ")
        if place.lower() == 'exit':
            break
        generated_desc = generate_description(region, place, max_new_tokens=200)
        print(f"\n생성된 소개글: {generated_desc}\n")


# 테스트 실행
csv_path = r"/maindata_crowling/descriptions\tourist_descriptions.csv"
print("테스트 시작...")

# 데이터셋 테스트 (5개 샘플)
test_with_dataset(csv_path, num_samples=5)

# 인터랙티브 테스트
interactive_test()

# 메모리 정리
if device.type == "cuda":
    torch.cuda.empty_cache()