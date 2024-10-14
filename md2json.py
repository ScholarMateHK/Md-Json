import time
import os
import re
import json
import json5
import copy
from openai import OpenAI

class MDToJSONConverter:
    def __init__(self, api_key, base_url, model):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=300
        )
        self.model = model
        self.completion_tokens = 0
        self.prompt_tokens = 0
        

    def read_md_file(self, file_path):
        """读取 Markdown 文件内容"""
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def read_json_file(self, file_path):
        """读取 JSON 文件内容"""
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def split_text_into_chunks_by_chapter(self, md_text):
        """按章节（基于 # 标题）拆分 Markdown 文件"""
        chunks = re.split(r"(?=^#{1,2} )", md_text, flags=re.MULTILINE)
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        return chunks

    def remove_special_characters(self, output):
        """移除特殊字符如 ```json 和 ```"""
        output = output.replace("```json", "").replace("```", "").strip()
        return output

    def count_token_len(self, string):
        # 首先判断是否包含中文
        if re.search(r"[\u4e00-\u9fff]", string):
            # 如果包含中文，直接返回字符串长度
            return len(string)
        else:
            # 如果不包含中文，通过空格分割字符串
            return len(string.split())
    
    def calculate_token_price(self, input_price, output_price):
        """计算 token 价格"""
        return (self.prompt_tokens / 1000) * input_price + (self.completion_tokens / 1000) * output_price

    def get_segmented_chunks(self, chunks):
        """根据章节拆分后的chunks，进一步拆分出first, then, final的chunks"""
        first_chunk = ""
        then_chunks = []
        final_chunk = ""

        # 处理first_chunk
        while chunks and self.count_token_len(first_chunk + chunks[0]) <= 3000:
            first_chunk += chunks.pop(0)

        # 处理final_chunk
        if len(chunks) >= 2:
            final_chunk = chunks.pop() + chunks.pop()
        elif len(chunks) == 1:
            final_chunk = chunks.pop()
        # final_chunk 可能很长，所以需要每3000字符分段处理
        final_chunks = [final_chunk[i:i+3000] for i in range(0, self.count_token_len(final_chunk), 3000)]

        # 处理then_chunks
        current_chunk = ""
        for chunk in chunks:
            if self.count_token_len(current_chunk + chunk) <= 3000:
                current_chunk += chunk
            else:
                if current_chunk:
                    then_chunks.append(current_chunk)
                current_chunk = chunk
        if current_chunk:
            then_chunks.append(current_chunk)

        return first_chunk, then_chunks, final_chunks

    def process_md_chunks(self, md_text, max_token=3000):
        chunks = self.split_text_into_chunks_by_chapter(md_text)
        first_chunk, then_chunks, final_chunks = self.get_segmented_chunks(chunks)
        paper_metadata = {}
        sections_metadata = []
        previous_hierarchy = {'sections': []}

        # 处理 first_chunk
        if first_chunk:
            chunk_result = self.process_single_chunk_first(first_chunk)
            paper_metadata = {key: value for key, value in chunk_result.items() if key != 'sections'}
            if 'sections' in chunk_result:
                sections_metadata.extend(chunk_result['sections'])
            previous_hierarchy = self.update_paper_structure(chunk_result, previous_hierarchy)
            print("First chunk processed")
            print(f"Updated paper_metadata: {paper_metadata}")

        # 处理 then_chunks
        for chunk in then_chunks:
            chunk_result = self.process_single_chunk_then(chunk, previous_hierarchy)
            if 'sections' in chunk_result:
                sections_metadata.extend(chunk_result['sections'])
            previous_hierarchy = self.update_paper_structure(chunk_result, previous_hierarchy)
            print("Then chunk processed")
            print(f"Updated paper_metadata: {paper_metadata}")

        # 处理 final_chunks
        if final_chunks:
            for chunk in final_chunks:
                chunk_result = self.process_single_chunk_final(chunk, previous_hierarchy)
                if 'sections' in chunk_result:
                    sections_metadata.extend(chunk_result['sections'])
                previous_hierarchy = self.update_paper_structure(chunk_result, previous_hierarchy)
                paper_metadata['references'] = chunk_result['references']
                print("Final chunk processed")
                print(f"Final paper_metadata: {paper_metadata}")
        
        # 将sections_metadata添加到paper_metadata中
        paper_metadata_items = list(paper_metadata.items())
        paper_metadata_items.insert(-1, ('sections', sections_metadata))
        paper_metadata = dict(paper_metadata_items)

        return paper_metadata, previous_hierarchy

    def call_openai_api(self, chunk, system_prompt):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk},
            ],
        )

        completion_dict = completion.to_dict()
        output = completion_dict["choices"][0]["message"]["content"]
        self.prompt_tokens += completion_dict["usage"]["prompt_tokens"]
        self.completion_tokens += completion_dict["usage"]["completion_tokens"]
        return output


    def process_single_chunk_first(self, chunk):
        """处理单个 chunk 并调用 OpenAI 生成 JSON 数据"""
        system_prompt = f"""【Task Description】: Transform the OCR-scanned text of an academic paper into a structured JSON file. The text may include errors, formatting issues, and random encodings from images. Remove corrupted or unreadable text (i.e., non-ASCII characters or sequences that do not resemble meaningful words or sentences, including all mathematical formulas and tables).
        【Objective】: Create a JSON document that preserves all original text from the OCR-scanned paper, excluding any images, formulas, tables, or corrupted characters, while retaining paragraph segmentation and the hierarchical structure of sections and subsections. 
        For **each individual element** (e.g., heading, subheading, paragraph, or list item), classify it as one of the following categories: 'title', 'authors', 'abstract', 'keywords', or 'sections'. for sections, specify its content as 'heading', 'content', and 'subsections'. subsections also have the same format.
        【Steps to Follow】:
        1. **Text Cleaning**: Remove corrupted characters, non-ASCII symbols, artifacts from images, formulas, and tables, without altering valid textual content.
        2. **Classification**: For **each element**, classify it into one of the following categories: 'title', 'authors', 'abstract', 'keywords', 'sections'. **For sections, specify if it’s a 'heading' or 'content' or 'subsections'**. Ensure the text from each element is preserved in the final JSON.
        3. **Segmentation**: Detect and preserve paragraphs, sections, and subsections based on structural and formatting cues (e.g., blank lines, indentation).
        4. **Hierarchy Identification**: Ensure the sections hierarchical structure is consistent with the paper's hierarchy.
        5. **JSON Conversion**: Convert the cleaned and classified text into JSON format, set key followed by these categories: 'title', 'authors', 'abstract', 'keywords', 'sections', for authors and keywords, the value is a list. for content, the value is a string. for sections, the value is a list of dictionaries with the keys: 'heading', 'content', 'subsections'. subsections also have the same format.
        **For sections and subsections**:
        1. Clearly indicate whether it is a **heading** or **content** or **subsections**.
        2. **Group all paragraphs** that belong to the same heading or subheading under that heading.
        【Final Notes】: Do not remove any valid text during processing. Ensure the final JSON file accurately reflects the organization and content of the original document, focusing solely on the textual data.
        【Important】:
        - Only return valid JSON data for **each individual element**. Do not include any additional explanations or comments in the response.
        - Ensure the JSON structure is valid and properly formatted. Ensure the new JSON structure follows the same logical order as the original text.
        - Avoid any omissions in the content generated, ensuring the text is complete.
        【Raw OCR-scanned text】:"""

        output = self.call_openai_api(chunk, system_prompt)
        output_cleaned = self.remove_special_characters(output)
        print(f"Chunk 返回的内容: {output_cleaned}\n")
        chunk_json = json5.loads(output_cleaned)

        return chunk_json


    def process_single_chunk_then(self, chunk, previous_hierarchy):
        system_prompt = f"""【Task Description】: Transform the OCR-scanned text of an academic paper into a structured JSON file. The text may include errors, formatting issues, and random encodings from images. Remove corrupted or unreadable text (i.e., non-ASCII characters or sequences that do not resemble meaningful words or sentences, including all mathematical formulas and tables).
        【Objective】: Create a JSON document that preserves all original text from the OCR-scanned paper, excluding any images, formulas, tables, or corrupted characters, while retaining paragraph segmentation and the hierarchical structure of sections and subsections. 
        For **each section or subsection**:
        1. **Text Cleaning**: Remove corrupted characters, non-ASCII symbols, artifacts from images, formulas, and tables, without altering valid textual content.
        2. **Classification**: For **each section or subsection**, classify it into one of the following keys: 'heading', 'content', 'subsections'. Ensure the text from each element is preserved in the final JSON.
        3. **Segmentation**: Detect and preserve paragraphs, sections, and subsections based on structural and formatting cues (e.g., blank lines, indentation).
        4. **Hierarchy Identification**: Ensure the new section hierarchical structure is consistent with the previously generated hierarchy {previous_hierarchy}. 
        5. **JSON Conversion**: Convert the cleaned and classified text into JSON format, set the key followed by these categories: 'heading', 'content', 'subsections'. For headings and contents, the value is a string. for subsections, the value is a list of dictionaries with the keys: 'heading', 'content', 'subsections'. subsections also have the same format.
        **For sections**:
        1. Clearly indicate whether it is a **heading** or **content** or **subsections**.
        2. **Group all paragraphs** that belong to the same heading or subheading under that heading.
        3. **Only considering text marked with # or ## as headings and subheadings when classifying elements in sections.**
        【Final Notes】: Do not remove any valid text during processing. Ensure the final JSON file accurately reflects the organization and content of the original document, focusing solely on the textual data.
        【Important】:
        - Only return valid JSON data for **each individual element**. Do not include any additional explanations or comments in the response.
        - Ensure the JSON structure is valid and properly formatted. Ensure the new JSON structure follows the same logical order as the original text.
        - Avoid any omissions in the content generated, ensuring the text is complete.
        - **Do not generate or classify any text as 'title'. Only classify as 'heading', 'content', or 'subsections'.**
        【Raw OCR-scanned text】: """

        output = self.call_openai_api(chunk, system_prompt)
        output_cleaned = self.remove_special_characters(output)
        print(f"Chunk 返回的内容: {output_cleaned}\n")
        chunk_json = json5.loads(output_cleaned)

        return chunk_json


    # todo
    def process_single_chunk_final(self, chunk, previous_hierarchy):
        system_prompt = f"""【Task Description】: Convert OCR-scanned text from an academic paper's sections into a structured JSON file. The text may contain errors, formatting issues, and random encodings. Your task is to **remove any corrupted or unreadable text** (i.e., non-ASCII characters or sequences that do not resemble meaningful words or sentences, including mathematical formulas and tables).
        【Objective】: Create a JSON document that preserves all original text from the OCR-scanned paper, excluding any images, formulas, tables, or corrupted characters, while retaining paragraph segmentation and the hierarchical structure of sections and subsections. 
        For **each section or subsection**:
        1. **Text Cleaning**: Remove corrupted characters, non-ASCII symbols, artifacts from images, formulas, and tables, without altering valid textual content.
        2. **Classification**: For **each section or subsection**, classify it into one of the following keys: 'heading', 'content', 'subsections'. Ensure the text from each element is preserved in the final JSON.
        3. **Segmentation**: Detect and preserve paragraphs, sections, and subsections based on structural and formatting cues (e.g., blank lines, indentation).
        4. **Hierarchy Identification**: Ensure the new section hierarchical structure is consistent with the previously generated hierarchy {previous_hierarchy}. 
        5. **JSON Conversion**: Convert the cleaned and classified text into JSON format, set the key followed by these categories: 'heading', 'content', 'subsections'. For headings and contents, the value is a string. for subsections, the value is a list of dictionaries with the keys: 'heading', 'content', 'subsections'. subsections also have the same format.
        **For sections**:
        1. Clearly indicate whether it is a **heading** or **content** or **subsections**.
        2. **Group all paragraphs** that belong to the same heading or subheading under that heading.
        3. **Only considering text marked with # or ## as headings and subheadings when classifying elements in sections.**
        **For each reference**:
        1. **Keep Right Hierarchy**: The reference should be placed under the `'references'` field, even if the chunk does not contain any sections.
        2. **Paper Name Extraction**: Extract the **title** of the referenced paper and classify it under the `'paper_name'` field.
        3. **Content Preservation**: Place the remaining content of the reference (authors, publication year, journal name, etc.) under the `'content'` field.
        4. **JSON Conversion**: Convert the cleaned reference text into a JSON object with the keys:
            - `'paper_name'`: The paper title.
            - `'content'`: The full reference text.
        【Final Notes】:
        - Return valid JSON data for each reference entry.
        - Ensure that the JSON structure is properly formatted.
        【Important】:
        - Only return valid JSON data for **each individual element**. Do not include any additional explanations or comments in the response.
        - Ensure the JSON structure is valid and properly formatted. Ensure the new JSON structure follows the same logical order as the original text.
        - Avoid any omissions in the content generated, ensuring the text is complete.
        **Raw OCR-scanned text**: """

        output = self.call_openai_api(chunk, system_prompt)
        output_cleaned = self.remove_special_characters(output)
        print(f"Chunk 返回的内容: {output_cleaned}\n")
        chunk_json = json5.loads(output_cleaned)

        return chunk_json

    def update_paper_structure(self, original_chunk_result, paper_structure):
        # hard copy chunk_result
        chunk_result = copy.deepcopy(original_chunk_result)
        """更新生成的 paper 结构，只保留章节和子章节框架"""
        if not chunk_result:
            return paper_structure

        # 遍历 chunk_result 中的每个元素，只要key为sections，就更新paper_structure
        for key, value in chunk_result.items():
            if key == "sections":
                paper_structure['sections'].extend(value)
        
        # 删除paper_structure中的content, 只保留heading和subsections
        def remove_content(section):
            if "content" in section:
                del section["content"]
            if "subsections" in section:
                for subsection in section["subsections"]:
                    remove_content(subsection)

        for section in paper_structure['sections']:
            remove_content(section)
        
        return paper_structure

    def save_json_to_file(self, paper_metadata, output_file):
        """将 JSON 数据保存到文件"""
        with open(output_file, "w", encoding="utf-8") as out_file:
            json.dump(paper_metadata, out_file, ensure_ascii=False, indent=4)

    def convert(self, input_file, output_file):
        md_text = self.read_md_file(input_file)
        paper_metadata, _ = self.process_md_chunks(md_text, max_token=3000)
        self.save_json_to_file(paper_metadata, output_file)
        print(f"JSON 文件已保存至 {output_file}")

if __name__ == "__main__":
    # (0.00875 + 0.035) / (0.004 + 0.012) = 2.73
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
    selected_llm = "qwen2.5-72b-instruct"
    converter = MDToJSONConverter(llm_dict[selected_llm]["key"], llm_dict[selected_llm]["base_url"], selected_llm)
    
    file_name = "1-s2.0-S0040162523008284-main.md"
    start_time = time.time()
    converter.convert(file_name, file_name.replace(".md", ".json"))
    end_time = time.time()
    conversion_time = end_time - start_time
    
    print(f"转换耗时: {conversion_time:.2f} 秒")
    print(f"输入tokens: {converter.prompt_tokens} , 输出tokens: {converter.completion_tokens}")
    print(f"总成本: ${converter.calculate_token_price(llm_dict[selected_llm]['input_price'], llm_dict[selected_llm]['output_price']):.4f}")