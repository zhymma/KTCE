import json
from concurrent.futures import ProcessPoolExecutor
from utils.parser import *
from utils.utils import *
from utils.api import *
from utils.grader import *
from pathlib import Path
import os

# 定义项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 定义数据目录
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"
KNOWLEDGE_DIR = DATA_DIR / "dataset_knowledge_exact"

# 确保目录存在
for dir_path in [DATA_DIR, DATASET_DIR, KNOWLEDGE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

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
 Here are some examples you can refer to:


### Question: Compute cos 330°.
### Answer: We know that 330° = 360° - 30°. Since cos(360° - θ) = cos θ for all angles θ, we have cos 330° = cos 30°. Since cos 30° = √3/2, we can conclude that cos 330° = √3/2.
### Analysis:
<AN>
  <l> Trigonometry-Cosine Function </l>
  <k> Understanding co-terminal angles in trigonometry </k>
  <k> Trigonometric identities, specifically the cosine of an angle related to a reference angle </k>
  <k> Knowledge of exact values of cosine for common angles (30°, 45°, 60°, etc.) </k>
  <k> Subtraction of angles and use of angle identities </k>
</AN>

---

### Question: The set of points $(x,y,z)$ that are equidistant to $(1,2,-5)$ and point $P$ satisfy an equation of the form\n\\[10x - 4y + 24z = 55.\\]Find the point $P.$

### Answer: Let $P = (a,b,c).$  If the point $(x,y,z)$ is equidistant to $(1,2,-5)$ and $(a,b,c),$ then\n\\[(x - 1)^2 + (y - 2)^2 + (z + 5)^2 = (x - a)^2 + (y - b)^2 + (z - c)^2.\\]Expanding, we get\n\\[x^2 - 2x + 1 + y^2 - 4y + 4 + z^2 + 10z + 25 = x^2 - 2ax + a^2 + y^2 - 2by + b^2 + z^2 - 2cz + c^2,\\]which simplifies to\n\\[(2a - 2) x + (2b - 4) y + (2c + 10) z = a^2 + b^2 + c^2 - 30.\\]We want this to coincide with the equation\n\\[10x - 4y + 24z = 55.\\]If we set $2a - 2 = 10,$ $2b - 4 = -4,$ and $2c + 10 = 24,$ then $a = 6,$ $b = 0,$ and $c = 7.$  Note that $a^2 + b^2 + c^2 - 30 = 55,$ so these values work.  Thus, $(a,b,c) = \\boxed{(6,0,7)}.$

### Analysis:
<AN>\n  <l>Geometry-Distance Formula</l>\n  <k>Using the distance formula for points in 3-dimensional space</k>\n  <k>Equating distances to form an equation</k>\n  <k>Manipulating and solving the expanded equation derived from the distance formula</k>\n</AN>\n<AN>\n  <l>Algebra-Solving Equations</l>\n  <k>Setting up equations based on geometric contexts</k>\n  <k>Solving simultaneous linear equations</k>\n  <k>Isolating variables and substitution</k>\n</AN>

---

### Question: A basketball player scores points in each game according to a quadratic pattern. In the first game, he scores 6 points, and in the second game, he scores 11 points. If the points scored per game continue to follow this quadratic pattern, how many total points will he have scored by the 10th game?

### Answer: To calculate the total points a basketball player scores by the 10th game following a quadratic pattern, we derive the quadratic equation \( p(n) = 2n^2 - n + 5 \) based on initial scores of 6 and 11 points in the first two games and an extrapolated score of 20 points in the third game. We then sum the points for the first 10 games using the formula for the sum of squares and the sum of the first ten natural numbers, resulting in a total score of 765 points. This method effectively combines quadratic equation derivation and series summation techniques to determine cumulative outcomes.

### Analysis:
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


"""
context_template = """### Question: {question}\n### Answer: {answer}\n### Analysis:\n"""


def process_dataset():
    """处理数据集并生成知识提取查询"""
    all_querys = []
    dataset = load_jsonl(DATASET_DIR / "math/math_train.jsonl")

    for idx, line in enumerate(dataset):
        id = "math@" + str(idx)
        question = line["problem"]
        answer = line["solution"]
        query = (
            "[Temperature = 0.0]\n"
            + prompt_template
            + context_template.format(
                question=question,
                answer=answer,
            )
        )
        all_querys.append({"id": id, "query": query})

    # 保存查询结果
    output_file = KNOWLEDGE_DIR / "math_kp_extract_queries_all.jsonl"
    save_jsonl(all_querys, output_file)
    print(f"已保存 {len(all_querys)} 条查询到 {output_file}")


if __name__ == "__main__":
    process_dataset()
