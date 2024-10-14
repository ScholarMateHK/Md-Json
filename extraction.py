from md2json import MDToJSONConverter
import argparse
import glob
import os
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将Markdown文件转换为JSON格式")
    parser.add_argument("input_dir", help="输入Markdown文件所在的目录")
    parser.add_argument("--model", default="qwen2.5-72b-instruct", choices=["qwen2.5-72b-instruct", "gpt-4o"], help="选择使用的模型")
    args = parser.parse_args()

    llm_dict = {
        "qwen2.5-72b-instruct": {
            "key": "sk-a1ec916362f94e9daf9a0147ad376f54",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "input_price": 0.004,
            "output_price": 0.012
        },
        "gpt-4o": {
            "key": "fk216003-Hrxi38NKWLTTErTeh75OJiLIj95rmg69",
            "base_url": "https://openai.api2d.net",
            "input_price": 0.00875,
            "output_price": 0.035
        }
    }
    
    md_files = glob.glob(os.path.join(args.input_dir, '**', '*.md'), recursive=True)
    total_files = len(md_files)
    total_time = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for i, file_dir in enumerate(md_files, 1):
        converter = MDToJSONConverter(llm_dict[args.model]["key"], llm_dict[args.model]["base_url"], args.model)
        file_name = os.path.basename(file_dir)
        # 如果已经存在.json文件，则跳过
        if os.path.exists(file_dir.replace(".md", ".json")):
            print(f"文件 {file_name} 已处理过，跳过")
            continue
        output_file = file_dir.replace(".md", ".json")
        
        print(f"处理文件 {i}/{total_files}: {file_name}")
        start_time = time.time()
        converter.convert(file_dir, output_file)
        end_time = time.time()
        conversion_time = end_time - start_time
        total_time += conversion_time

        print(f"  转换耗时: {conversion_time:.2f} 秒")
        print(f"  输入tokens: {converter.prompt_tokens}, 输出tokens: {converter.completion_tokens}")
        print(f"  文件成本: ${converter.calculate_token_price(llm_dict[args.model]['input_price'], llm_dict[args.model]['output_price']):.4f}")
        
        total_prompt_tokens += converter.prompt_tokens
        total_completion_tokens += converter.completion_tokens
        
    print("\n总结:")
    print(f"总转换时间: {total_time:.2f} 秒")
    print(f"总输入tokens: {total_prompt_tokens}, 总输出tokens: {total_completion_tokens}")
    print(f"总成本: ${(total_prompt_tokens / 1000 * llm_dict[args.model]['input_price'] + total_completion_tokens / 1000 * llm_dict[args.model]['output_price']):.4f}")