import os
import re
import json
from openai import OpenAI


def read_md_file(file_path):
    """读取 Markdown 文件内容"""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def split_text_into_chunks_by_chapter(md_text):
    """按章节（基于 # 标题）拆分 Markdown 文件"""
    chunks = re.split(r"(?=^#{1,2} )", md_text, flags=re.MULTILINE)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    return chunks


def remove_special_characters(output):
    """移除特殊字符如 ```json 和 ```"""
    output = output.replace("```json", "").replace("```", "").strip()
    return output


def process_md_chunks(md_text, client, previous_hierarchy, max_chars=3000):
    """处理拆分后的 chunk，并确保每个 chunk 的字符数不超过 max_chars，保持层次一致性"""
    chunks = split_text_into_chunks_by_chapter(md_text)
    paper_structure = {
        "paper": {
            "title": "",  # 初始化标题
            "authors": [],  # 初始化作者
            "abstract": "",  # 初始化摘要
            "index_terms": [],  # 初始化关键词
            "sections": [],  # 初始化章节
            "references": [],  # 初始化参考文献
        }
    }

    chunk_all = ""

    for i, chunk in enumerate(chunks):
        if len(chunk_all) + len(chunk) > max_chars:
            chunk_result = process_single_chunk(chunk_all, client, previous_hierarchy)
            update_paper_structure(chunk_result, paper_structure)
            chunk_all = chunk  # 重置 chunk_all
        else:
            chunk_all += chunk

    if chunk_all:
        chunk_result = process_single_chunk(chunk_all, client, previous_hierarchy)
        update_paper_structure(chunk_result, paper_structure)

    return paper_structure


def process_single_chunk(chunk, client, previous_hierarchy):
    """处理单个 chunk 并调用 OpenAI 生成 JSON 数据"""
    text = f"""【Task Description】: Transform the OCR-scanned text of an academic paper into a structured JSON file. The text may include errors, formatting issues, and random encodings from images. Remove corrupted or unreadable text (i.e., non-ASCII characters or sequences that do not resemble meaningful words or sentences, including all mathematical formulas and tables).
    【Objective】: Create a JSON document that preserves all original text from the OCR-scanned paper, excluding any images, formulas, tables, or corrupted characters, while retaining paragraph segmentation and the hierarchical structure of sections and subsections. 
    For **each individual element** (e.g., heading, paragraph, or list item), classify it as one of the following categories: 'title', 'authors', 'abstract', 'keywords', 'sections', or 'references'. **For sections, indicate whether it is a heading, subheading, or paragraph**.

    【Steps to Follow】:
    1. **Text Cleaning**: Remove corrupted characters, non-ASCII symbols, artifacts from images, formulas, and tables, without altering valid textual content.
    2. **Classification**: For **each element**, classify it into one of the following categories: 'title', 'authors', 'abstract', 'keywords', 'sections', or 'references'. For sections, specify if it’s a 'heading', 'subheading', or 'paragraph'. Ensure the text from each element is preserved in the final JSON.
    3. **Segmentation**: Detect and preserve paragraphs, sections, and subsections based on structural and formatting cues (e.g., blank lines, indentation).
    4. **Hierarchy Identification**: Ensure the new hierarchical structure is consistent with the previously generated hierarchy.
    5. **JSON Conversion**: Convert the cleaned and classified text into JSON format, using 'type' to define the category (e.g., 'title', 'authors', 'abstract', 'keywords', 'sections', 'references'), and 'content' for the actual text content.

    【Final Notes】: Do not remove any valid text during processing. Ensure the final JSON file accurately reflects the organization and content of the original document, focusing solely on the textual data.

    【Important】:
    - Only return valid JSON data for **each individual element**. Do not include any additional explanations or comments in the response.
    - Ensure the JSON structure is valid and properly formatted. Ensure the new JSON structure follows the same logical order as the original text.

    **Classify each element in the following text into the correct category and return each element's classification in JSON format**:
    【Raw OCR-scanned text】:{chunk}"""

    completion = client.chat.completions.create(
        model="qwen2.5-72b-instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
        ],
    )

    completion_dict = completion.to_dict()
    output = completion_dict["choices"][0]["message"]["content"]

    output_cleaned = remove_special_characters(output)
    print(f"Chunk 返回的内容: {output_cleaned}\n")
    chunk_json = json.loads(output_cleaned)

    return chunk_json


def update_paper_structure(chunk_result, paper_structure):
    """更新生成的 paper 结构"""
    if not chunk_result:
        return

    for element in chunk_result:
        if element["type"] == "title":
            paper_structure["paper"]["title"] = element["content"]
        elif element["type"] == "authors":
            paper_structure["paper"]["authors"].extend(element["content"])
        elif element["type"] == "abstract":
            paper_structure["paper"]["abstract"] = element["content"]
        elif element["type"] == "keywords":
            paper_structure["paper"]["index_terms"].extend(element["content"])
        elif element["type"] == "sections":
            paper_structure["paper"]["sections"].append(
                {
                    "type": element.get("section_type", "paragraph"),
                    "content": element["content"],
                }
            )
        elif element["type"] == "references":
            paper_structure["paper"]["references"].append(
                {"reference": element["content"]}
            )


def save_json_to_file(json_data, output_file):
    """将 JSON 数据保存到文件"""
    with open(output_file, "w", encoding="utf-8") as out_file:
        json.dump(json_data, out_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    file_path = "227364298/227364298.md"
    output_file = "227364298.json"
    md_text = read_md_file(file_path)

    client = OpenAI(
        api_key="sk-a1ec916362f94e9daf9a0147ad376f54",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 传递之前生成的层次结构作为参数
    previous_hierarchy = []

    # 处理 MD 文件并生成 JSON 数据，确保每个 chunk 不超过 3000 字符
    paper_structure = process_md_chunks(md_text, client, previous_hierarchy)

    # 保存生成的 JSON 数据
    save_json_to_file(paper_structure, output_file)

    print(f"JSON 文件已保存至 {output_file}")
