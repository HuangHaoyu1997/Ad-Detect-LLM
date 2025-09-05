import json
from transformers import AutoModelForCausalLM, AutoTokenizer



if __name__ == "__main__":
    # 读取./sft_dataset/test.json文件
    with open('./sft_dataset/test.json', 'r', encoding='utf-8') as f:
        data = json.load(f)


    model_name = "/mnt/e/LLM/Qwen3-0.6B"
    # model_name = "/mnt/e/LLM/Qwen3-4B-Instruct-2507"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    # 跑一遍测试集
    count = 0
    for i in range(len(data)):
        
        text = tokenizer.apply_chat_template(
            data[i]['messages'][:2],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768
            )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 


        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        print(i, content)
        # 如果判断正确
        if '是' in content and '是' in data[i]['messages'][2]['content']:
            count += 1
            print('yes')
        if '否' in content and '否' in data[i]['messages'][2]['content']:
            count += 1
            print('yes')
        if i % 100 == 0:
            print('accuracy:', count/len(data))
        # print("thinking content:", thinking_content)
        
    print("count:", count, 'accuracy:', count/len(data))
