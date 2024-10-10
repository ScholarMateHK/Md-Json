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


def process_md_chunks(md_text, client, max_chars=3000):
    chunks = split_text_into_chunks_by_chapter(md_text)
    chunk_all = ""
    paper_metadata = {"sections": []}
    previous_hierarchy = {}

    first_chunk_processed = False

    for chunk in chunks:
        if len(chunk_all) + len(chunk) > max_chars:
            if not first_chunk_processed:
                chunk_result = process_single_chunk_first(
                    chunk_all, client, previous_hierarchy
                )
                print("First chunk processed")
                first_chunk_processed = True
            else:
                print("Then chunk processed")
                chunk_result = process_single_chunk_then(
                    chunk_all, client, previous_hierarchy
                )
            previous_hierarchy = update_paper_structure(
                chunk_result, previous_hierarchy
            )
            paper_metadata = update_paper_metadata(chunk_result, paper_metadata)
            print(
                f"Paper_metadata after processing this chunk: {json.dumps(paper_metadata, indent=2, ensure_ascii=False)}"
            )
            chunk_all = ""
        chunk_all += "\n" + chunk

    if chunk_all.strip():
        if not first_chunk_processed:
            print("First chunk processed (final)")
            chunk_result = process_single_chunk_first(
                chunk_all, client, previous_hierarchy
            )
        else:
            print("Then chunk processed (final)")
            chunk_result = process_single_chunk_then(
                chunk_all, client, previous_hierarchy
            )
            print(
                f"Paper_metadata final processing this chunk: {json.dumps(paper_metadata, indent=2, ensure_ascii=False)}"
            )

        previous_hierarchy = update_paper_structure(chunk_result, previous_hierarchy)
        paper_metadata = update_paper_metadata(chunk_result, paper_metadata)

    return paper_metadata, previous_hierarchy


def process_single_chunk_then(chunk, client, previous_hierarchy):
    text = f"""【Task Description】: Transform the OCR-scanned text of an academic paper into a structured JSON file. The text may include errors, formatting issues, and random encodings from images. Remove corrupted or unreadable text (i.e., non-ASCII characters or sequences that do not resemble meaningful words or sentences, including all mathematical formulas and tables).
    【Objective】: Create a JSON document that preserves all original text from the OCR-scanned paper, excluding any images, formulas, tables, or corrupted characters, while retaining paragraph segmentation and the hierarchical structure of sections and subsections. 
    For **each individual element** (e.g., heading, subheading, paragraph, or list item), classify it as one of the following categories: 'sections' or 'references'.
    【Steps to Follow】:
    1. **Text Cleaning**: Remove corrupted characters, non-ASCII symbols, artifacts from images, formulas, and tables, without altering valid textual content.
    2. **Classification**: For **each element**, classify it into one of the following categories: 'heading', 'subheading', 'paragraph', or 'references'. Ensure the text from each element is preserved in the final JSON.
    3. **Segmentation**: Detect and preserve paragraphs, sections, and subsections based on structural and formatting cues (e.g., blank lines, indentation).
    4. **Hierarchy Identification**: Ensure the new hierarchical structure is consistent with the previously generated hierarchy {previous_hierarchy}. 
    5. **JSON Conversion**: Convert the cleaned and classified text into JSON format, using 'type' to define the category (e.g., 'heading', 'subheading', 'paragraph', or 'references'), and 'content' for the actual text content.
    **For sections**:
    1. Clearly indicate whether it is a **heading** or **subheading** or **paragraph.
    2. **Group all paragraphs** that belong to the same heading or subheading under that heading.
    3. **Only considering text marked with # or ## as headings and subheadings when classifying elements in sections.**
    【Final Notes】: Do not remove any valid text during processing. Ensure the final JSON file accurately reflects the organization and content of the original document, focusing solely on the textual data.
    【Important】:
    - Only return valid JSON data for **each individual element**. Do not include any additional explanations or comments in the response.
    - Ensure the JSON structure is valid and properly formatted. Ensure the new JSON structure follows the same logical order as the original text.
    - **Do not generate or classify any text as 'title'. Only classify as 'heading', 'subheading', 'paragraph', or 'references'.**
    【Raw OCR-scanned text】: {chunk}"""

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


def process_single_chunk_first(chunk, client, previous_hierarchy):
    """处理单个 chunk 并调用 OpenAI 生成 JSON 数据"""
    text = f"""【Task Description】: Transform the OCR-scanned text of an academic paper into a structured JSON file. The text may include errors, formatting issues, and random encodings from images. Remove corrupted or unreadable text (i.e., non-ASCII characters or sequences that do not resemble meaningful words or sentences, including all mathematical formulas and tables).
    【Objective】: Create a JSON document that preserves all original text from the OCR-scanned paper, excluding any images, formulas, tables, or corrupted characters, while retaining paragraph segmentation and the hierarchical structure of sections and subsections. 
    For **each individual element** (e.g., heading, subheading, paragraph, or list item), classify it as one of the following categories: 'title', 'authors', 'abstract', 'keywords', 'sections', or 'references'. 
    【Steps to Follow】:
    1. **Text Cleaning**: Remove corrupted characters, non-ASCII symbols, artifacts from images, formulas, and tables, without altering valid textual content.
    2. **Classification**: For **each element**, classify it into one of the following categories: 'title', 'authors', 'abstract', 'keywords', 'sections', or 'references'. **For sections, specify if it’s a 'heading' or 'subheading' or 'paragraph'**. Ensure the text from each element is preserved in the final JSON.
    3. **Segmentation**: Detect and preserve paragraphs, sections, and subsections based on structural and formatting cues (e.g., blank lines, indentation).
    4. **Hierarchy Identification**: Ensure the new hierarchical structure is consistent with the previously generated hierarchy {previous_hierarchy}. 
    5. **JSON Conversion**: Convert the cleaned and classified text into JSON format, using 'type' to define the category (e.g., 'title', 'authors', 'abstract', 'keywords', 'sections', 'references'), and 'content' for the actual text content.
     **For sections**:
    1. Clearly indicate whether it is a **heading** or **subheading** or **paragraph.
    2. **Group all paragraphs** that belong to the same heading or subheading under that heading.
    【Final Notes】: Do not remove any valid text during processing. Ensure the final JSON file accurately reflects the organization and content of the original document, focusing solely on the textual data.
    【Important】:
    - Only return valid JSON data for **each individual element**. Do not include any additional explanations or comments in the response.
    - Ensure the JSON structure is valid and properly formatted. Ensure the new JSON structure follows the same logical order as the original text.
    【Raw OCR-scanned text】: {chunk}"""

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


def ensure_chunk_result_is_list(chunk_result):
    """
    确保 chunk_result 是一个包含字典的列表。如果 chunk_result 是字典，则将其转化为列表。
    """
    if isinstance(chunk_result, dict):
        return [chunk_result]
    elif isinstance(chunk_result, list):
        return chunk_result
    else:
        print(
            f"Unexpected chunk_result type: {type(chunk_result)}. Returning an empty list."
        )
        return []


def update_paper_structure(chunk_result, paper_structure):
    """更新生成的 paper 结构，只保留章节和子章节框架"""
    if not isinstance(chunk_result, list):
        chunk_result = [chunk_result]

    for element in chunk_result:
        if "sections" in element:
            section_content = element["sections"]
            for sub_element in section_content:
                if sub_element["type"] == "heading":
                    paper_structure.setdefault("sections", []).append(
                        {
                            "type": "heading",
                            "content": sub_element["content"],
                            "subsections": [],
                        }
                    )

                elif sub_element["type"] == "subheading":
                    if len(paper_structure["sections"]) > 0:
                        paper_structure["sections"][-1].setdefault(
                            "subsections", []
                        ).append(
                            {"type": "subheading", "content": sub_element["content"]}
                        )
    return paper_structure


def update_paper_metadata(chunk_result, paper_metadata):
    """更新生成的 paper metadata，将所有信息保留到 JSON"""
    if not isinstance(chunk_result, list):
        chunk_result = [chunk_result]
        print(f"Chunk result: {chunk_result}")  # 打印 chunk_result，查看数据格式

    for element in chunk_result:
        # 处理 sections 键
        if "sections" in element:
            section_content = element["sections"]
            for sub_element in section_content:
                if sub_element["type"] == "heading":
                    # 如果没有同样的 heading，追加新的 heading
                    paper_metadata.setdefault("sections", []).append(
                        {
                            "type": "heading",
                            "content": sub_element["content"],
                            "subsections": [],
                            "paragraphs": [],
                        }
                    )

                elif sub_element["type"] == "paragraph":
                    # 段落应当追加到最后一个 heading
                    if len(paper_metadata["sections"]) > 0:
                        paper_metadata["sections"][-1]["paragraphs"].append(
                            {"type": "paragraph", "content": sub_element["content"]}
                        )

                elif sub_element["type"] == "subheading":
                    # 子标题应当追加到最后一个 heading
                    if len(paper_metadata["sections"]) > 0:
                        paper_metadata["sections"][-1]["subsections"].append(
                            {
                                "type": "subheading",
                                "content": sub_element["content"],
                                "paragraphs": [],
                            }
                        )

        # 处理 title
        elif "title" in element:
            if "title" not in paper_metadata:
                paper_metadata["title"] = element["content"]  # 避免重复追加

        # 处理 authors
        elif "authors" in element:
            if "authors" not in paper_metadata:
                paper_metadata["authors"] = element["content"]  # 避免重复追加

        # 处理 references
        elif "references" in element:
            paper_metadata.setdefault("references", []).append(
                {"type": "paragraph", "content": element["content"]}
            )

    return paper_metadata


def save_json_to_file(paper_metadata, output_file):
    """将 JSON 数据保存到文件"""
    with open(output_file, "w", encoding="utf-8") as out_file:
        json.dump(paper_metadata, out_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    file_path = "227364298/227364298.md"
    output_file = "227364298.json"
    md_text = read_md_file(file_path)

    client = OpenAI(
        api_key="sk-a1ec916362f94e9daf9a0147ad376f54",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    paper_metadata, _ = process_md_chunks(md_text, client, max_chars=3000)

    save_json_to_file(paper_metadata, output_file)

    print(f"JSON 文件已保存至 {output_file}")
