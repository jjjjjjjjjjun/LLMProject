import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

# 1️⃣ 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

model_path = r"/LLMtuning/kogpt2_finetuned"
#model_path = "skt/kogpt2-base-v2"  # 대체 모델로 테스트하려면 이 줄로 변경
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    use_safetensors=True,
    dtype=torch.float32,
    output_attentions=True  # 모델 로드 시 output_attentions 강제 설정 시도
).to(device)

# 패딩 토큰 설정
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = 1
model.config.pad_token_id = 1

# 어텐션 출력 강제 활성화
model.config.output_attentions = True

print("모델 로드 완료 ✅")
print("모델 설정:", model.config)

# 2️⃣ 토큰 단위로 생성 + 어텐션 추출
def generate_with_attention(prompt, max_new_tokens=50):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated_ids = input_ids.clone()
    print(f"\n[프롬프트]: {prompt}\n")
    all_attentions = []

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=generated_ids,
                attention_mask=torch.ones_like(generated_ids),
                output_attentions=True,
                use_cache=False
            )

        # 어텐션 출력 확인
        if outputs.attentions is None:
            print("경고: 이 모델은 어텐션 출력을 지원하지 않습니다. 어텐션 분석을 생략합니다.")
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            continue

        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        # 어텐션 저장 (마지막 레이어, 모든 헤드 평균)
        last_attn = outputs.attentions[-1].mean(dim=1)  # (batch, seq_len, seq_len)
        all_attentions.append(last_attn[:, -1, :].squeeze(0).cpu().numpy())

        if next_token_id.item() == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"📝 생성 문장: {generated_text}\n")

    if all_attentions:
        tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])
        importance = all_attentions[-1]
        top_idx = np.argsort(importance)[::-1][:5]
        print("🎯 마지막 토큰 기준 어텐션 상위 5개 토큰:")
        for i in top_idx:
            print(f"   {tokens[i]}: {importance[i]:.4f}")

# 3️⃣ 인터랙티브 테스트
def interactive_test():
    print("\n[지역 기반 문장 생성 + 어텐션 분석] 종료하려면 'exit' 입력")
    while True:
        region = input("\n지역 입력 (예: 서울, 부산, 제주도): ")
        if region.lower() == "exit":
            break
        prompt = f"{region}의 대표 관광지를 추천해줘"
        generate_with_attention(prompt, max_new_tokens=50)

interactive_test()