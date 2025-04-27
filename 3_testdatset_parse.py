import json
from concurrent.futures import ProcessPoolExecutor
from utils.parser import *
from utils.utils import *
from utils.api import *
from utils.grader import *
import xml.etree.ElementTree as ET
import re
from collections import defaultdict
from FlagEmbedding import BGEM3FlagModel
import torch
import textwrap
from pathlib import Path
import logging
from typing import Dict, List, Any, Union
from datetime import datetime

# 定义项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 定义数据目录
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"
KNOWLEDGE_DIR = DATA_DIR / "dataset_knowledge_exact"

# 确保目录存在
for dir_path in [DATA_DIR, DATASET_DIR, KNOWLEDGE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


# 输入信息
test_dataset = load_jsonl("dataset/math/test.jsonl")
all_type = [
    "Algebra",
    "Intermediate Algebra",
    "Prealgebra",
    "Geometry",
    "Counting & Probability",
    "Precalculus",
    "Number Theory",
]

test_knowledge_parsed = load_jsonl(
    "data/dataset_knowledge_exact/test_knowledge_parsed.jsonl"
)
math_domains = read_json("data/dataset_knowledge_exact/math_domains.json")


def escape_xml_text_for_latex(text: str) -> str:
    """
    转义文本中的特定字符为 LaTeX 符号，但仅当它们周围有空格时

    Args:
        text: 输入文本

    Returns:
        str: 转义后的文本
    """
    text = re.sub(r"\\", r"\\\\", text)
    text = re.sub(r" < ", r" \\lt ", text)
    text = re.sub(r" > ", r" \\gt ", text)
    return text


def parse_xml(id: str, xml_data: str) -> Union[List[Dict], str]:
    """
    解析XML数据

    Args:
        id: 数据ID
        xml_data: XML格式的数据

    Returns:
        Union[List[Dict], str]: 解析结果或错误标识
    """
    # 预处理XML数据
    xml_data = xml_data.replace("<1>", "<l>").replace("</1>", "</l>")
    xml_data = escape_xml_text_for_latex(xml_data)

    try:
        # 包装XML数据以确保正确解析
        root = ET.fromstring(f"<root>{xml_data}</root>")
        results = []

        # 遍历每个<AN>元素
        for an in root.findall("AN"):
            elements = list(an)
            l_index = 0
            while l_index < len(elements):
                elem = elements[l_index]
                if elem.tag == "l":
                    # 处理主题文本
                    topic = elem.text.strip() if elem.text else ""
                    if "-" in topic:
                        field, subfield = topic.split("-", 1)
                    elif "–" in topic:
                        field, subfield = topic.split("–", 1)
                    else:
                        logging.warning(f"ID {id}: 无法解析主题 {topic}")
                        l_index += 1
                        continue

                    # 收集关键点
                    keypoints = []
                    l_index += 1
                    while l_index < len(elements) and elements[l_index].tag == "k":
                        if elements[l_index].text:
                            keypoints.append(elements[l_index].text.strip())
                        l_index += 1

                    # 创建结果字典
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
    except Exception as e:
        logging.error(f"XML解析错误 ID {id}: {e}")
        return "wrong"


prompt_template = """As a task summary agent, we now have a task of  Math Word Problems, which requires mathematical reasoning on  textual data.  To extract and summarize the high-level paradigms and general approaches for solving such problems, please analyze the general task in solving this question.

We describe a task through field-subfield and key points. The field and subfield represent a class of tasks and subtasks in Math Word Problems, and key points describe the knowledge or general methods needed for this type of task. Be sure to avoid repetition of key points for clarity and conciseness. Make sure the response is general, objective, and independent of the context of the given question, such as specific numerical values or names. Specific requirements are as follows: 

1. Identify and categorize the most important Mathematical field-subfield (one or two) involved in solving the problem.

2. For each field-subfield, enumerate the essential Key Points relevant to the problem like math rules, formulas, theorems, common sense, pandas, etc.

3. **Output Format**: For each task entry, please ouput a xml <AN></AN> with a field-subfield in <l></l> and some key points in <k></k>. If there are multiple tasks, it is expressed as <AN>...</AN>\n<AN>...</AN>\n<AN>...</AN>.

4. Your field-subfields are list below. Your generated field-subfields must in it .
```json
{
    "Calculus": [
        "Limits",
        "Function Operations",
        "Optimization",
        "Infinite Series"
    ],
    "Algebra": [
        "Quadratic Equations",
        "Linear Equations",
        "Functions",
        "Systems of Equations",
        "Inequalities",
        "Polynomials",
        "Solving Equations",
        "Simplifying Expressions",
        "Sequences and Series",
        "Radicals and Root Operations",
        "Function Operations",
        "Ratios and Proportions",
        "Polynomial Expansions",
        "Absolute Value",
        "Rational Functions",
        "Polynomial Factoring",
        "Exponents and Logarithms",
        "Function Transformations",
        "Completing the Square",
        "Complex Numbers",
        "Polynomial Equations",
        "Summation",
        "Substitution",
        "Fractions",
        "Variables and Expressions",
        "Exponential Growth",
        "Multiplication",
        "Percentages",
        "Divisibility"
    ],
    "Arithmetic": [
        "Basic Operations",
        "Rounding Numbers",
        "Order of Operations",
        "Unit Conversion",
        "Rate Problems",
        "Averages",
        "Fractions",
        "Addition",
        "Ratios and Proportions",
        "Division",
        "Percentages",
        "Multiplication",
        "Summation",
        "Time Calculations"
    ],
    "Number Theory": [
        "Properties of Integers",
        "Floor and Ceiling Functions",
        "Integer Properties",
        "Powers and Roots",
        "Fractions and Decimals",
        "Floor Function",
        "Prime Numbers",
        "Powers and Exponents",
        "Perfect Squares",
        "Factors and Multiples",
        "Integers Properties",
        "Perfect Squares and Cubes",
        "Divisibility",
        "Multiples",
        "Modular Arithmetic",
        "Factorials",
        "Counting Digits",
        "Division and Remainders",
        "Greatest Common Divisor (GCD)",
        "Factorials and Multiples",
        "Digit Sums",
        "Modulo Arithmetic",
        "Base Conversion",
        "Least Common Multiple",
        "Integers",
        "Units Digit",
        "Prime Factorization",
        "Least Common Multiple (LCM)"
    ],
    "Geometry": [
        "Circles",
        "Coordinate Geometry",
        "Rectangles",
        "Midpoint Formula",
        "Distance Formula",
        "Area Calculation",
        "Triangles",
        "Parabolas",
        "Polygons",
        "Right Triangles",
        "Lines and Angles",
        "Volume of Solids",
        "Perimeter",
        "3D Shapes",
        "Transformations",
        "Squares",
        "Planes in Three Dimensions",
        "Similar Triangles",
        "Pythagorean Theorem",
        "Reflections",
        "Trapezoids",
        "Angles",
        "Similarity",
        "Parallelograms",
        "Coordinate Systems",
        "Conic Sections",
        "Quadrilaterals",
        "Ellipse"
    ],
    "Financial Mathematics": [
        "Compound Interest"
    ],
    "Sequences and Series": [
        "Infinite Series",
        "Geometric Sequences",
        "Arithmetic Sequences",
        "Sequences and Series"
    ],
    "Complex Numbers": [
        "Absolute Value",
        "Function Operations"
    ],
    "Combinatorics": [
        "Factorials",
        "Counting Problems",
        "Binomial Coefficients",
        "Pascal's Triangle"
    ],
    "Measurement": [
        "Unit Conversion"
    ],
    "Statistics": [
        "Mean and Median",
        "Mean",
        "Median"
    ],
    "Probability": [
        "Basic Concepts",
        "Expected Value"
    ],
    "Data Interpretation": [
        "Bar Graphs"
    ],
    "Trigonometry": [
        "Sine and Cosine Functions",
        "Sine Function",
        "Tangent Function",
        "Cosine Function",
        "Trigonometric Functions",
        "Polar Coordinates"
    ],
    "Set Theory": [
        "Overlapping Sets"
    ],
    "Number Systems": [
        "Base Conversion"
    ],
    "Linear Algebra": [
        "Matrices",
        "Vectors",
        "Determinants",
        "Vectors and Parametric Equations"
    ],
    "Vector Algebra": [
        "Dot Product",
        "Cross Product"
    ]
}
```
Here are some output format you can refer to:

### Analysis:
<AN>
  <l> Trigonometry-Cosine Function </l>
  <k> Understanding co-terminal angles in trigonometry </k>
  <k> Trigonometric identities, specifically the cosine of an angle related to a reference angle </k>
  <k> Knowledge of exact values of cosine for common angles (30°, 45°, 60°, etc.) </k>
  <k> Subtraction of angles and use of angle identities </k>
</AN>


<AN>\n  <l>Geometry-Distance Formula</l>\n  <k>Using the distance formula for points in 3-dimensional space</k>\n  <k>Equating distances to form an equation</k>\n  <k>Manipulating and solving the expanded equation derived from the distance formula</k>\n</AN>\n<AN>\n  <l>Algebra-Solving Equations</l>\n  <k>Setting up equations based on geometric contexts</k>\n  <k>Solving simultaneous linear equations</k>\n  <k>Isolating variables and substitution</k>\n</AN>

<AN>
  <l>Algebra-Quadratic Equations</l>
  <k>Formulation of a quadratic equation from given data points</k>
  <k>Solving for coefficients a, b, and c using initial conditions from the sequence</k>
</AN>
<AN>
  <l>Algebra-Sequences and Series</l>
  <k>Calculation of the sum of the series using the derived quadratic equation</k>
  <k>Use of series sum formula to determine total points over multiple games</k>
</AN>

---

we have a wrong format here, may be wrong in xml format ,plsease correct and output the right format.

### Analysis (wrong format)
===wrong===

### Analysis:

"""

context_template = """"""


def func_0():
    datas = load_jsonl(
        "data/dataset_knowledge_exact/math_tabmwp_kp_extract_test_queries_all.json"
    )
    test_dataset = load_jsonl("dataset/math/test.jsonl")
    import os
    from datetime import datetime

    paese_knowledge = []
    error_num = 0
    for line in datas:
        if "math" not in line["id"]:
            continue
        id = line["id"].split("@")[-1]
        if "answer_mode4" not in line:
            continue
        knowledge = line["answer_mode4"]
        parsed_data = parse_xml(id, knowledge)

        if parsed_data == "wrong":
            query = prompt_template.replace("===wrong===", knowledge)
            response = chat_api("gpt-3.5-turbo-0125", query, 0, "", temperature=0.0)
            parsed_data = parse_xml(id, response)
            if parsed_data != "wrong":
                print("self-refine!")
                line["answer_mode4"] = response
                paese_knowledge.extend(parsed_data)
            else:
                error_num += 1
        else:
            paese_knowledge.extend(parsed_data)
    print(len(paese_knowledge))
    print(error_num)
    save_jsonl(
        paese_knowledge,
        "data/dataset_knowledge_exact/test_knowledge_parsed.jsonl",
    )
    save_jsonl(
        datas,
        "data/dataset_knowledge_exact/math_tabmwp_kp_extract_test_queries_all.json",
    )


def get_kps_by_id(id):
    kps = []
    for line in test_knowledge_parsed:
        if line["id"] == id:
            field = line["field"]
            subfield = line["subfield"]
            if field in math_domains:
                if subfield in math_domains[field]:
                    kps.append(
                        {
                            "filed": field,
                            "subfield": subfield,
                            "knowledge_point": line["keypoints"],
                        }
                    )
    return kps


if __name__ == "__main__":
    # func_0()
    # exit(0)
    test_ids_with_kps = {}
    for test_type in all_type:
        type_ids = []
        type_without_kps_ids = []
        for line in test_dataset:
            if line["type"] == test_type:
                id = str(line["idx"])
                kps = get_kps_by_id(id)
                if len(kps) == 0:
                    type_without_kps_ids.append(id)
                else:
                    type_ids.append(id)
                    called_tools = []
        print(
            "type",
            test_type,
            "with kps",
            len(type_ids),
            "without kps",
            len(type_without_kps_ids),
        )
        test_ids_with_kps[test_type] = type_ids
        dump_json(
            test_ids_with_kps, "data/dataset_knowledge_exact/test_ids_with_kps.json"
        )
