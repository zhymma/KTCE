import json
import os
from utils.utils import *
from utils.parser import *
from utils.grader import *
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import xml.etree.ElementTree as ET
import re

# 定义项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 定义数据目录
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"
MATH_DIR = DATASET_DIR / "math"

# 确保目录存在
for dir_path in [DATA_DIR, DATASET_DIR, MATH_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 数学领域定义
math_domains = {
    "Calculus": ["Functions and their Properties", "Optimization", "Limits"],
    "Algebra": [
        "Solving Equations",
        "Polynomials",
        "Inequalities",
        "Functions",
        "Simplifying Expressions",
        "Linear Equations",
        "Quadratic Equations",
        "Square Roots",
        "Radicals",
        "Sequences and Series",
        "Linear Functions",
        "Complex Numbers",
        "Function Operations",
        "Exponents",
        "Rational Functions",
        "Function Transformations",
        "Proportions",
        "Proportional Relationships",
        "Logarithms",
        "Substitution",
        "Exponential Growth",
        "Summation",
        "Absolute Value",
        "Variables and Expressions",
        "Ratios and Proportions",
        "Geometric Series",
        "Interval Notation",
        "Polynomial Expansion",
        "Real Numbers",
    ],
    "Others": ["Problem Context", "Graph Interpretation", "Problem Solving"],
    "Arithmetic": [
        "Order of Operations",
        "Time Calculations",
        "Division",
        "Basic Operations",
        "Fractions",
        "Multiplication",
        "Percentages",
        "Addition",
        "Averages",
        "Rate Problems",
        "Unit Conversion",
        "Rounding Numbers",
    ],
    "Number Theory": [
        "Fractions and Decimals",
        "Integer Properties",
        "Powers and Roots",
        "Floor Function",
        "Floor and Ceiling Functions",
        "Perfect Squares",
        "Divisibility",
        "Factors and Multiples",
        "Prime Numbers",
        "Multiples",
        "Odd and Even Numbers",
        "Digit Sums",
        "Modulo Arithmetic",
        "Properties of Integers",
        "Units Digit",
        "Greatest Common Divisor (GCD)",
        "Perfect Squares and Cubes",
        "Counting Digits",
        "Modular Arithmetic",
        "Division and Remainders",
        "Powers and Exponents",
    ],
    "Geometry": [
        "Circles",
        "Coordinate Geometry",
        "Distance Formula",
        "Polygons",
        "Midpoint Formula",
        "Reflections",
        "Area Calculation",
        "Lines and Angles",
        "Perimeter",
        "Parabolas",
        "Area of a Circle",
        "Rectangles",
        "Triangles",
        "Transformations",
        "Squares",
        "3D Shapes",
        "Angles",
        "Volume of Solids",
        "Pyramids",
        "Similar Triangles",
        "Cones",
        "Parallelograms",
        "Conic Sections",
        "Ellipse",
        "Coordinate Systems",
        "Planes in Three Dimensions",
    ],
    "Financial Mathematics": ["Compound Interest"],
    "Sequences and Series": ["Infinite Series"],
    "Complex Numbers": ["Absolute Value"],
    "Combinatorics": [
        "Counting Problems",
        "Factorials",
        "Binomial Coefficients",
        "Pascal's Triangle",
    ],
    "Measurement": ["Unit Conversion"],
    "Statistics": ["Mean", "Mean and Median"],
    "Probability": ["Basic Concepts", "Expected Value", "Geometric Probability"],
    "Data Interpretation": ["Bar Graphs"],
    "Trigonometry": [
        "Tangent Function",
        "Sine and Cosine Functions",
        "Polar Coordinates",
    ],
    "Set Theory": ["Overlapping Sets"],
    "Number Systems": ["Base Conversion", "Binary Numbers"],
    "Linear Algebra": [
        "Matrices",
        "Vectors",
        "Determinants",
        "Vectors and Parametric Equations",
    ],
    "Vector Algebra": ["Dot Product"],
}


def process_none_field(topic):
    """处理没有明确字段的主题"""
    if topic in list(math_domains.keys()):
        field = topic
        subfield = math_domains[topic][0]
        return field, subfield
    else:
        for field in math_domains:
            if topic in math_domains[field]:
                return field, topic
    return "", topic


def escape_xml_text_for_latex(text):
    """转义文本中的特定字符为 LaTeX 符号，但仅当它们周围有空格时"""
    text = re.sub(r"\\", r"\\\\", text)
    text = re.sub(r" < ", r" \\lt ", text)
    text = re.sub(r" > ", r" \\gt ", text)
    return text


def parse_xml(id, xml_data, dataset):
    """解析XML数据并提取知识信息"""
    # 预处理XML数据
    xml_data = xml_data.replace("<1>", "<l>").replace("</1>", "</l>")
    xml_data = escape_xml_text_for_latex(xml_data)

    # 包装XML数据以确保正确解析
    try:
        root = ET.fromstring(f"<root>{xml_data}</root>")
    except ET.ParseError as e:
        print(f"XML Parse Error for ID {id}: {e}")
        print(xml_data)
        return []

    results = []
    for an in root.findall("AN"):
        elements = list(an)
        l_index = 0
        while l_index < len(elements):
            elem = elements[l_index]
            if elem.tag == "l":
                topic = elem.text.strip() if elem.text else ""
                if "-" in topic:
                    field, subfield = topic.split("-", 1)
                elif "–" in topic:
                    field, subfield = topic.split("–", 1)
                else:
                    field, subfield = process_none_field(topic)
                    if field == "":
                        field = dataset[int(id)]["type"]
                        print(id, field, subfield)

                keypoints = []
                l_index += 1
                while l_index < len(elements) and elements[l_index].tag == "k":
                    if elements[l_index].text:
                        keypoints.append(elements[l_index].text.strip())
                    l_index += 1

                result_dict = {
                    "id": id,
                    "field": field.strip(),
                    "subfield": subfield.strip(),
                    "keypoints": keypoints,
                }
                results.append(result_dict)
            else:
                l_index += 1

    return results


def process_knowledge_file(knowledge_file, dataset_file, output_file):
    """处理知识文件并保存解析结果"""
    print(f"处理文件: {knowledge_file}")
    knowledge = read_json(knowledge_file)
    dataset = load_jsonl(dataset_file)
    print(f"知识条目数: {len(knowledge)}")

    parsed_knowledge = []
    for id in tqdm(knowledge, desc="解析知识"):
        parsed_data = parse_xml(id, knowledge[id], dataset)
        parsed_knowledge.extend(parsed_data)

    print(f"解析后的知识条目数: {len(parsed_knowledge)}")
    save_jsonl(parsed_knowledge, output_file)
    print(f"已保存到: {output_file}")


def main():
    """主函数"""
    # 处理训练集知识
    train_knowledge_file = MATH_DIR / "train_knowledge.json"
    train_dataset_file = MATH_DIR / "train.jsonl"
    train_output_file = MATH_DIR / "train_knowledge_parsed.jsonl"
    process_knowledge_file(train_knowledge_file, train_dataset_file, train_output_file)

    # 处理测试集知识
    test_knowledge_file = MATH_DIR / "test_knowledge.json"
    test_dataset_file = MATH_DIR / "test.jsonl"
    test_output_file = MATH_DIR / "test_knowledge_parsed.jsonl"
    process_knowledge_file(test_knowledge_file, test_dataset_file, test_output_file)


if __name__ == "__main__":
    main()
