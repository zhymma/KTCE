import re
from typing import Any, Dict
import openai
import subprocess
import requests


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string


def strip_string(string):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    string = string.replace("\\!", "")
    string = string.replace("\\ ", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    string = string.replace("\\text", "")
    string = string.replace("x\\in", "")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    string = string.replace("\\cdot", "")

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace('"', "")

    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def extract_answer(pred_str):
    try:
        if "boxed" in pred_str:
            ans = pred_str.split("boxed")[-1]
            if len(ans) == 0:
                return ""
            elif ans[0] == "{":
                stack = 1
                a = ""
                for c in ans[1:]:
                    if c == "{":
                        stack += 1
                        a += c
                    elif c == "}":
                        stack -= 1
                        if stack == 0:
                            break
                        a += c
                    else:
                        a += c
            else:
                a = ans.split("$")[0].strip()
            pred = a
        elif "he answer is" in pred_str:
            pred = pred_str.split("he answer is")[-1].strip()
        elif extract_program_output(pred_str) != "":
            # fall back to program
            pred = extract_program_output(pred_str)
        else:  # use the last number
            pattern = "-?\d*\.?\d+"
            pred = re.findall(pattern, pred_str.replace(",", ""))
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ""

        # multiple line
        pred = pred.split("\n")[0]
        if pred != "" and pred[0] == ":":
            pred = pred[1:]
        if pred != "" and pred[-1] == ".":
            pred = pred[:-1]
        if pred != "" and pred[-1] == "/":
            pred = pred[:-1]
        pred = strip_string(pred)
    except:
        pred = ""
    return pred


def extract_program(result: str, lang="python", last_only=False):
    """
    extract the program after "```python", and before "```"
    """
    target = "```" + lang
    if target not in result:
        return ""
    program = ""
    start = False
    for line in result.split("\n"):
        if line.startswith(target):
            if last_only:
                program = ""  # only extract the last program
            else:
                if len(program) > 0:
                    program += "\n\n"
            start = True
        elif line.startswith("```"):
            start = False
        elif start:
            program += line + "\n"
    return program


def extract_program_multiple(result: str, lang="python"):
    """
    extract the program after "```python", and before "```"
    """
    target = "```" + lang
    if target not in result:
        return result
    programs = []
    program = ""
    start = False
    for line in result.split("\n"):
        if line.startswith(target):
            if len(program) > 0:
                programs.append(program.strip())
                program = ""
            start = True
        elif line.startswith("```"):
            start = False
        elif start:
            program += line + "\n"
    return programs


import re
import json


def extract_json(result: str):
    """
    Extract the json strings found between "```python" and "```".

    Parameters:
    - result: A string that may contain one or more JSON strings wrapped between "```python" and "```".

    Returns:
    - jsons: A list of parsed JSON objects.
    """
    jsons = []
    # Find all occurrences of strings between the specified markers
    pattern = re.compile(r"```json(.*?)```", re.DOTALL)
    matches = pattern.findall(result)
    temp_json_strings = "\n".join(matches)
    # Attempt to parse each found string as JSON
    for match in matches:
        try:
            json_str = match.strip()  # Remove leading/trailing whitespace
            json_regex = r"{[^{}]*}"
            json_strings = re.findall(json_regex, json_str)
            for json_str in json_strings:
                temp = json.loads(json_str)
                if len(temp) > 0:
                    jsons.append(temp)

        except json.JSONDecodeError:
            # raise ValueError(f"A string could not be parsed as JSON: {json_str}")
            # print("A string could not be parsed as JSON:", json_str)
            return temp_json_strings

    return jsons


from collections import defaultdict


def merge_imports(import_statements):
    import_dict = defaultdict(set)
    other_imports = []

    for statement in import_statements:
        if statement.startswith("import "):
            other_imports.append(statement)
        elif statement.startswith("from "):
            parts = statement.split(" import ")
            module = parts[0][5:]
            items = parts[1].split(", ")
            for item in items:
                import_dict[module].add(item)

    merged_imports = []

    for module, items in import_dict.items():
        merged_imports.append(f'from {module} import {", ".join(sorted(items))}')

    merged_imports.extend(other_imports)

    return merged_imports


def execute_code(code, api_call=None, code_file="code_exec/tmp0", add_imports=True):

    f = open(f"{code_file}.py", "w", encoding="utf-8")
    if api_call is not None:
        code = code + "\n\n" + api_call
    code = code.split("\n")
    if add_imports:
        import_statements = """
from sympy import (
    Abs,
    Add,
    Eq,
    Function,
    I,
    Integral,
    Interval,
    Matrix,
    Matrix,
    Mul,
    N,
    Piecewise,
    Poly,
    Polygon,
    Pow,
    Rational,
    S,
    Sum,
    Symbol,
    Union,
    acos,
    apart,
    asin,
    atan,
    binomial,
    cancel,
    ceiling,
    cos,
    deg,
    diff,
    divisors,
    exp,
    expand,
    factor,
    factorial,
    factorial as sp_factorial,
    factorial as sympy_factorial,
    factorint,
    floor,
    floor as sympy_floor,
    gcd,
    im,
    integrate,
    isprime,
    lambdify,
    lcm,
    limit,
    linsolve,
    log,
    mod_inverse,
    nextprime,
    nsimplify,
    oo,
    pi,
    poly,
    primefactors,
    primerange,
    rad,
    re,
    simplify,
    sin,
    solve,
    solve_poly_inequality,
    solveset,
    sqrt,
    summation,
    symbols,
    sympify,
    tan,
)
from sympy.parsing.sympy_parser import parse_expr
from decimal import (
    Decimal,
    ROUND_DOWN,
    ROUND_HALF_DOWN,
    ROUND_HALF_UP,
    ROUND_UP,
    getcontext,
)
from typing import Callable, Dict, List, Optional, Tuple, Union, Union, List
from math import (
    acos,
    asin,
    atan,
    atan2,
    ceil,
    comb,
    comb as math_comb,
    cos,
    cosh,
    degrees,
    erf,
    erfc,
    exp,
    fabs,
    factorial,
    factorial as compute_factorial,
    factorial as math_factorial,
    floor,
    fmod,
    gamma,
    gcd,
    gcd,
)  # Importing gcd from math module, gcd as compute_gcd, gcd as math_gcd, hypot, isclose, lcm, lgamma, log, log10, log2, modf, perm as math_perm, pi, pow, radians, sin, sinh, sqrt, tan, tanh, trunc
from scipy.linalg import eig, inv
from scipy.optimize import fsolve, minimize, minimize_scalar
from sympy.core.numbers import Integer
from sympy.abc import a, b, c, d, n, x, y
from scipy.stats import gmean, hmean
from operator import add, mul, sub, truediv as div
from sympy.ntheory import factorint
from sympy.geometry import Line, Plane, Point, Point3D
from scipy.special import comb
from sympy.calculus.util import continuous_domain
from pint import UnitRegistry
from shapely.geometry import Polygon
from functools import lru_cache, reduce
from fractions import Fraction
from scipy import stats, stats as lib_stats
from cmath import exp, phase, pi, polar
from matplotlib import pyplot
from datetime import datetime, timedelta
from sympy.physics import units
from scipy.integrate import quad
from numpy import poly1d
from sympy.solvers import *
from scipy.spatial import ConvexHull
from sympy.core.sympify import SympifyError
from collections import Counter, OrderedDict, defaultdict, deque, namedtuple
from itertools import permutations, product
from sympy.core.relational import Relational
from sympy.sets import Interval
from sympy.sets.sets import Union
from sympy.solvers.inequalities import *
from numpy.linalg import LinAlgError, solve
import scipy.integrate as integrate
import re
import pint
import numpy as np  # Importing numpy for power and complex number support
import sympy as sp
import scipy.linalg as la
import fractions
import functools
import cmath
import dateutil.parser
import numpy as here
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.integrate as spint
import scipy.optimize as optimize
import datetime
import matplotlib.pyplot as plt
import decimal
import sympy
import numpy as np
import math
import numpy as lib_np
import math
import sympy
import random
import numpy as np
import pandas as pd
import cmath
import spicy
import sympy as sp
import sympy
import itertools
import statistics
import cmath
import math
import fractions
"""
        f.write(import_statements + "\n")

    f.write("\n".join(code))

    f.close()
    i = 0
    while i < 3:
        try:
            result = subprocess.run(
                ["python", f"{code_file}.py"],
                capture_output=True,
                check=False,
                text=True,
                timeout=90,
            )
        except Exception as e:
            if code_file in str(e):
                i += 1
                continue
            else:
                return False, e
        else:
            if result.returncode != 0:
                error_msg = result.stderr.strip()
                msgs = error_msg.split("\n")
                new_msgs = []
                want_next = False
                for m in msgs:
                    if "Traceback" in m:
                        new_msgs.append(m)
                    elif m == msgs[-1]:
                        new_msgs.append(m)
                    elif f"/{code_file}.py" in m:
                        st = m.index("/") + 1
                        ed = m.index(f"/{code_file}.py") + 1
                        clr = m[st:ed]
                        m = m.replace(clr, "")
                        new_msgs.append(m)
                        want_next = True
                    elif want_next:
                        new_msgs.append(m)
                        want_next = False
                error_msg = "\n".join(new_msgs)
                if error_msg == "":
                    error_msg = "Code execution failed!"
                return False, error_msg.strip()
            else:
                output = result.stdout
                if output == "":
                    return (
                        False,
                        "0",
                    )  
                return True, output.strip()
            break

    return False, "Code run time out!"


def extract_program_output(pred_str):
    """
    extract output between the last ```output\n...\n```
    """
    if "```output" not in pred_str:
        return ""
    if "```output" in pred_str:
        pred_str = pred_str.split("```output")[-1]
    if "```" in pred_str:
        pred_str = pred_str.split("```")[0]
    output = pred_str.strip()
    return output


def parse_ground_truth(example: Dict[str, Any], data_name):
    if "gt_cot" in example:
        return example["gt_cot"], strip_string(example["gt"])

    # parse ground truth
    if data_name in ["math", "ocw"]:
        gt_cot = example["solution"]
        gt_ans = extract_answer(gt_cot)
    elif data_name == "gsm8k":
        gt_cot, gt_ans = example["answer"].split("####")
    elif data_name == "gsm-hard":
        gt_cot, gt_ans = example["code"], example["target"]
    elif data_name == "svamp":
        gt_cot, gt_ans = example["Equation"], example["Answer"]
    elif data_name == "asdiv":
        gt_cot = example["formula"]
        gt_ans = re.sub(r"\(.*?\)", "", example["answer"])
    elif data_name == "mawps":
        gt_cot, gt_ans = None, example["target"]
    elif data_name == "tabmwp":
        gt_cot = example["solution"]
        gt_ans = example["answer"]
        if example["ans_type"] in ["integer_number", "decimal_number"]:
            if "/" in gt_ans:
                gt_ans = int(gt_ans.split("/")[0]) / int(gt_ans.split("/")[1])
            elif "," in gt_ans:
                gt_ans = float(gt_ans.replace(",", ""))
            elif "%" in gt_ans:
                gt_ans = float(gt_ans.split("%")[0]) / 100
            else:
                gt_ans = float(gt_ans)
    elif data_name == "bbh":
        gt_cot, gt_ans = None, example["target"]
    else:
        raise NotImplementedError(data_name)
    # post process
    gt_cot = str(gt_cot).strip()
    gt_ans = strip_string(gt_ans)
    return gt_cot, gt_ans


def parse_question(example, data_name):
    question = ""
    if data_name == "asdiv":
        question = f"{example['body'].strip()} {example['question'].strip()}"
    elif data_name == "svamp":
        body = example["Body"].strip()
        if not body.endswith("."):
            body = body + "."
        question = f'{body} {example["Question"].strip()}'
    elif data_name == "tabmwp":
        title_str = (
            f'regarding "{example["table_title"]}" ' if example["table_title"] else ""
        )
        question = f"Read the following table {title_str}and answer a question:\n"
        question += f'{example["table"]}\n{example["question"]}'
        if example["choices"]:
            question += (
                f' Please select from the following options: {example["choices"]}'
            )
    else:
        for key in ["question", "problem", "Question", "input"]:
            if key in example:
                question = example[key]
                break
    assert question != ""
    return question.strip()


def run_execute(executor, result, prompt_type, execute=False):
    if not result or result == "error":
        return None, None
    report = None

    if "program_only" in prompt_type:
        prediction = extract_program_output(result)
    elif prompt_type in ["pot", "pal"] and execute:
        code = extract_program(result)
        prediction, report = executor.apply(code)
    else:
        prediction = extract_answer(result)

    prediction = strip_string(prediction)
    return prediction, report


# def extract_math_tools(result: str):
#     """
#     extract the functions from code string"
#     """
#     functions = []

#     function = ""
#     for line in result.split("\n"):
#         if line.startswith("def"):
#             if function:
#                 functions.append(function)
#             function = line + "\n"
#         elif function and line.startswith(" "):
#             function += line + "\n"
#     if function:
#         functions.append(function)
#     return functions


def extract_api_calls(code: str):
    """
    Extract non-indented code lines that appear after all function definitions in the provided code string.
    """
    lines = code.split("\n")  # Split the code into lines
    api_calls = []
    in_function = False  # To check if the current line is inside a function
    past_functions = False  # To mark when all functions definitions have been parsed

   
    last_def = 0
    for i, line in enumerate(lines):
        if line.startswith("def "):
            last_def = i

    for i, line in enumerate(lines):
        if i < last_def:
            continue
        if line.startswith("def "):
            in_function = True  # Start of a new function
        elif line.strip() == "" or (line.startswith(" ") and in_function):
            continue  # Skip empty lines and lines inside functions
        else:
            in_function = False  # Any non-indented line after 'def' ends the function

        if not in_function and not line.startswith("def "):
            past_functions = True  # If we encounter a non-function, non-indented line, all functions are assumed past

        if past_functions and not line.startswith(" ") and line.strip():
            api_calls.append(line)  # Collect non-indented lines after all functions

    api_calls = "\n".join(
        api_calls
    )  # Combine the non-indented lines into a single string
    return api_calls


def extract_imports(result: str):
    """
    extract the import from code string"
    """
    imports = []
    for line in result.split("\n"):
        if line.startswith("import") or line.startswith("from"):
            imports.append(line)
    return imports


def extract_function_name(function):
    """
    Given a python function, use rule-based method to extract the function name.
    :param function: a python function described in string.
    :return: the function name.
    """

    function = function.strip()
    if function.startswith("def"):
        function = function[3:]
    if function.endswith(":"):
        function = function[:-1]
    function = function.strip()
    function_name = function.split("(")[0].strip()
    return function_name


def extract_class_name(function):
    """
    Given a python function, use rule-based method to extract the function name.
    :param function: a python function described in string.
    :return: the function name.
    """

    function = function.strip()
    if function.startswith("class"):
        function = function[5:]
    if function.endswith(":"):
        function = function[:-1]
    function = function.strip()
    function_name = function.split("(")[0].strip()
    return function_name


def extract_function_args_num(function):
    """
    Given a python function, use rule-based method to extract the number of arguments.
    :param function: a python function described in string.
    :return: the number of arguments.
    """
    function = function.strip()
    if function.startswith("def"):
        function = function[3:]
    if function.endswith(":"):
        function = function[:-1]
    function = function.strip()

    # Extract the arguments part from the function definition
    args_str = function.split("(")[1].split(")")[0].strip()

    # Check if there are no arguments
    if args_str == "":
        return 0

    # Split the arguments by comma and count them
    args_list = args_str.split(",")
    num_args = len(args_list)

    return num_args


def extract_function_description(function):
    function = function.strip()
    if function.startswith("def"):
        function = function[3:]
    if function.endswith(":"):
        function = function[:-1]
    # return function
    if '"""' in function:
        items = function.split('"""')
    elif "'''" in function:
        items = function.split("'''")
    else:
        return "None"
    docstring = items[1].strip()
    if "Parameters" in docstring:
        description = docstring.split("Parameters")[0].strip()
        return description
    else:
        return docstring


def extract_class_description(function):
    """
    Extract the docstring from the given function or class definition.

    Parameters:
    function (str): The string representation of the function or class definition.

    Returns:
    str: The extracted docstring, or an empty string if no docstring is found.
    """
    # Regular expression to match docstring
    pattern = re.compile(r'("""|\'\'\')([\s\S]*?)\1')
    match = pattern.search(function)

    if match:
        docstring = match.group(2).strip()
        if "Parameters" in docstring:
            description = docstring.split("Parameters")[0].strip()
            return description
        else:
            return docstring
    else:
        return ""


def extract_function_docstring(function):
    """
    Given a python function, use rule-based method to extract the function docstring.
    :param function: a python function described in string.
    :return:
    """
    function = function.strip()
    if function.startswith("def"):
        function = function[3:]
    if function.endswith(":"):
        function = function[:-1]
    # return function
    if '"""' in function:
        return function.split('"""')[1].strip().split('"""')[0].strip()
    elif "'''" in function:
        return function.split("'''")[1].strip().split("'''")[0].strip()
    else:
        return None


def extract_class_docstring(function):
    """
    Extract the docstring from the given function or class definition.

    Parameters:
    function (str): The string representation of the function or class definition.

    Returns:
    str: The extracted docstring, or an empty string if no docstring is found.
    """
    # Regular expression to match docstring
    pattern = re.compile(r'("""|\'\'\')([\s\S]*?)\1')
    match = pattern.search(function)

    if match:
        docstring = match.group(2).strip()
        return docstring
    else:
        return ""


def remove_function_docstring(function):
    """
    Given a Python function described in string, remove its docstring.
    :param function: a python function described in string.
    :return: the function string without the docstring.
    """
    function = function.strip()

    # Use regular expression to find and remove the docstring
    docstring_pattern = re.compile(r'""".*?"""|\'\'\'.*?\'\'\'', re.DOTALL)
    function_without_docstring = re.sub(docstring_pattern, "", function).strip()

    return function_without_docstring


def extract_assert_code(result: str):
    """
    extract the assert code from code string"
    """
    if "assert" not in result:
        return None

    assert_code = ""
    result = result.split("```python")[1]
    result = result.split("```")[0]
    return result


#! 检索相关
import numpy as np
import torch
from torch.utils.data import DataLoader


def compute_simcse(model, tokenizer, texts):

    data_loader = DataLoader(texts, shuffle=False, batch_size=32)
    embeddings = []
    for batch in data_loader:
        # Tokenize input texts
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        for key in inputs.keys():
            inputs[key] = inputs[key].cuda()
        # Get the embeddings
        with torch.no_grad():
            embedding = (
                model(**inputs, output_hidden_states=True, return_dict=True)
                .pooler_output.detach()
                .cpu()
                .numpy()
            )
            embeddings.extend(embedding)

    # compute similarity
    embeddings = torch.from_numpy(np.array(embeddings))
    del tokenizer, model, data_loader
    return embeddings


def sort_by_similarity(model, tokenizer, current_solutions, ori_dataset):

    current_solution_embeddings = compute_simcse(
        model, tokenizer, current_solutions["question"]
    )
    instruction_embeddings = compute_simcse(model, tokenizer, ori_dataset["question"])

    similarity_matrix = torch.zeros(
        (len(current_solution_embeddings), len(instruction_embeddings))
    )
    for i in range(len(instruction_embeddings) // 1000 + 1):
        start, end = i * 1000, min((i + 1) * 1000, len(instruction_embeddings))
        part_of_instruction_embeddings = instruction_embeddings[start:end]
        similarity_matrix[:, start:end] = torch.nn.functional.cosine_similarity(
            current_solution_embeddings.unsqueeze(1).cuda(),
            part_of_instruction_embeddings.unsqueeze(0).cuda(),
            dim=2,
        ).cpu()
    print(similarity_matrix)
    # for each sample in instruction dataset, find the highest similarity value in current solutions. The correspoding index does not matter.
    similarity_scores, _ = torch.max(similarity_matrix, dim=0)
    print(
        similarity_matrix.shape, similarity_scores.shape
    )  # similarity_scores.shape == (len(ori_dataset),)
    # sort similarity_scores in ascending order
    sorted_similarity_scores, sorted_similarity_indices = torch.sort(
        similarity_scores, dim=0
    )
    print(sorted_similarity_scores[:5], sorted_similarity_scores[-5:])
    # sort bootstrap dataset by sorted_similarity_scores
    sorted_instruction_dataset = ori_dataset.select(sorted_similarity_indices)
    return (
        sorted_instruction_dataset,
        sorted_similarity_scores,
        sorted_similarity_indices,
    )


def extract_math_tools(result: str):
    """
    Extract the functions and classes from code string.
    """
    tools = []
    tool = ""

    for line in result.split("\n"):
        if line.startswith("def") or line.startswith("class"):
            if tool:
                tool = tool.strip()
                tools.append(tool)
            tool = line + "\n"
        elif tool and (line.startswith(" ") or line.startswith("\t")):
            tool += line + "\n"

    if tool:
        tool = tool.strip()
        tools.append(tool)

    return tools


if __name__ == "__main__":
    # t = extract_json("```json\n[{\n  \"a\": 1\n},{\n  \"a\": 1\n}\n]\n```")
    pass
