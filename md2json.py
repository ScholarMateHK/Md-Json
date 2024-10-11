import os
import re
import json
from openai import OpenAI


def read_md_file(file_path):
    """读取 Markdown 文件内容"""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def read_json_file(file_path):
    """读取 JSON 文件内容"""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def split_text_into_chunks_by_chapter(md_text):
    """按章节（基于 # 标题）拆分 Markdown 文件"""
    chunks = re.split(r"(?=^#{1,2} )", md_text, flags=re.MULTILINE)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    return chunks


def remove_special_characters(output):
    """移除特殊字符如 ```json 和 ```"""
    output = output.replace("```json", "").replace("```", "").strip()
    return output


def count_token_len(string):
    # 首先判断是否包含中文
    if re.search(r"[\u4e00-\u9fff]", string):
        # 如果包含中文，直接返回字符串长度
        # 因为中文每个字符通常被视为一个 token
        return len(string)
    else:
        # 如果不包含中文，通过空格分割字符串
        # 并返回分割后的单词数量作为 token 数
        return len(string.split())


def get_segmented_chunks(chunks):
    """根据章节拆分后的chunks，进一步拆分出first, then, final的chunks"""
    first_chunk = ""
    then_chunks = []
    final_chunk = ""

    # 先处理first_chunk
    while chunks and count_token_len(first_chunk + chunks[0]) <= 3000:
        first_chunk += chunks.pop(0)

    # 再处理final_chunk
    if len(chunks) >= 2:
        final_chunk = chunks.pop() + chunks.pop()
    elif len(chunks) == 1:
        final_chunk = chunks.pop()

    # 最后处理then_chunks
    current_chunk = ""
    for chunk in chunks:
        if count_token_len(current_chunk + chunk) <= 3000:
            current_chunk += chunk
        else:
            if current_chunk:
                then_chunks.append(current_chunk)
            current_chunk = chunk
    if current_chunk:
        then_chunks.append(current_chunk)

    return first_chunk, then_chunks, final_chunk


def process_md_chunks(md_text, client, max_token=3000):
    chunks = split_text_into_chunks_by_chapter(md_text)
    first_chunk, then_chunks, final_chunk = get_segmented_chunks(chunks)
    paper_metadata = {}
    sections_metadata = []
    previous_hierarchy = []

    """处理拆分后的 chunk，并确保每个 chunk 的字符数不超过 max_token，保持层次一致性"""
    # 处理 first_chunk
    if first_chunk:
        # chunk_result = process_single_chunk_first(first_chunk, client)
        import pickle
        chunk_result = pickle.load(open('chunk_result1.pkl','rb'))
        previous_hierarchy = update_paper_structure(chunk_result, previous_hierarchy)
        # chunk_result中的所有元素, 除了sections, 更新到paper_metadata中
        paper_metadata = {key: value for key, value in chunk_result.items() if key != 'sections'}
        # 将chunk_result中的sections更新到sections_metadata中
        sections_metadata.extend(chunk_result['sections'])
        print("First chunk processed")
        print(f"Updated paper_metadata: {paper_metadata}")

    # 处理 then_chunks
    for chunk in then_chunks:
        # chunk_result = process_single_chunk_then(chunk, client, previous_hierarchy)
        chunk_result = pickle.load(open('chunk_result2.pkl','rb'))
        previous_hierarchy = update_paper_structure(chunk_result, previous_hierarchy)
        sections_metadata.extend(chunk_result['sections'])
        print("Then chunk processed")
        print(f"Updated paper_metadata: {paper_metadata}")

    # 处理 final_chunk
    if final_chunk:
        chunk_result = process_single_chunk_final(
            final_chunk, client, previous_hierarchy
        )
        previous_hierarchy = update_paper_structure(chunk_result, previous_hierarchy)
        # 如果chunk_result中有sections，则更新sections_metadata
        if chunk_result['sections']:
            sections_metadata.extend(chunk_result['sections'])
        # 更新references到paper_metadata
        paper_metadata['references'] = chunk_result['references']
        print("Final chunk processed")
        print(f"Final paper_metadata: {paper_metadata}")
    
    # 最后,添加sections_metadata到paper_metadata中
    paper_metadata['sections'] = sections_metadata

    return paper_metadata, previous_hierarchy


def process_single_chunk_first(chunk, client):
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
    【Raw OCR-scanned text】:"""

    completion = client.chat.completions.create(
        model="qwen2.5-72b-instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk},
        ],
    )

    completion_dict = completion.to_dict()
    output = completion_dict["choices"][0]["message"]["content"]

    output_cleaned = remove_special_characters(output)
    print(f"Chunk 返回的内容: {output_cleaned}\n")
    chunk_json = json.loads(output_cleaned)

    return chunk_json


def process_single_chunk_then(chunk, client, previous_hierarchy):
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
    - **Do not generate or classify any text as 'title'. Only classify as 'heading', 'content', or 'subsections'.**
    【Raw OCR-scanned text】: """

    completion = client.chat.completions.create(
        model="qwen2.5-72b-instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk},
        ],
    )
    ##这里补充了client，不然process_single_chunk_then函数不会调取API
    completion_dict = completion.to_dict()
    output = completion_dict["choices"][0]["message"]["content"]

    output_cleaned = remove_special_characters(output)
    print(f"Chunk 返回的内容: {output_cleaned}\n")
    chunk_json = json.loads(output_cleaned)

    return chunk_json


# todo
def process_single_chunk_final(chunk, client, previous_hierarchy):
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
    1. **Text Cleaning**: Remove corrupted characters, non-ASCII symbols, image artifacts, and any irrelevant content (formulas, tables, etc.) while preserving the complete reference text.
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
**Raw OCR-scanned text**: """

    completion = client.chat.completions.create(
        model="qwen2.5-72b-instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk},
        ],
    )
    ##这里补充了client，不然process_single_chunk_then函数不会调取API
    completion_dict = completion.to_dict()
    output = completion_dict["choices"][0]["message"]["content"]

    output_cleaned = remove_special_characters(output)
    print(f"Chunk 返回的内容: {output_cleaned}\n")
    chunk_json = json.loads(output_cleaned)

    return chunk_json


def update_paper_structure(
    chunk_result, paper_structure
):  ##这个按照下面metadata的报错来看，他也是错的，但是这个函数本身不影响模型的输出和最终结果
    """更新生成的 paper 结构，只保留章节和子章节框架"""
    if not chunk_result:
        return paper_structure  # 如果没有chunk_result，返回当前的paper_structure

    # 遍历 chunk_result 中的每个元素，只要key为sections，就更新paper_structure
    for key, value in chunk_result.items():
        if key == "sections":
            paper_structure.extend(value)
    
    # 删除paper_structure中的content, 只保留heading和subsections
    def remove_content(section):
        if "content" in section:
            del section["content"]
        if "subsections" in section:
            for subsection in section["subsections"]:
                remove_content(subsection)

    for section in paper_structure:
        remove_content(section)
    
    return paper_structure

def save_json_to_file(paper_metadata, output_file):
    """将 JSON 数据保存到文件"""
    with open(output_file, "w", encoding="utf-8") as out_file:
        json.dump(paper_metadata, out_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    file_path = "227364298.md"
    output_file = "227364298.json"
    md_text = read_md_file(file_path)

    client = OpenAI(
        api_key="sk-a1ec916362f94e9daf9a0147ad376f54",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 处理 MD 文件并生成 JSON 数据，确保每个 chunk 不超过 3000 字符
    paper_metadata, _ = process_md_chunks(md_text, client, max_token=3000)

    # 保存生成的 JSON 数据
    save_json_to_file(paper_metadata, output_file)

    print(f"JSON 文件已保存至 {output_file}")
