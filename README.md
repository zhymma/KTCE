# Automated Creation of Reusable and Diverse Toolsets for Enhancing LLM Reasoning

This repository contains the implementation of our AAAI 2025 paper "Automated Creation of Reusable and Diverse Toolsets for Enhancing LLM Reasoning". The paper presents a novel approach to enhance LLM reasoning by creating reusable and diverse toolsets through a knowledge-grounded tool creation with evolution framework.

## Paper Information

- **Title**: Automated Creation of Reusable and Diverse Toolsets for Enhancing LLM Reasoning
- **Conference**: AAAI 2025
- **Paper Link**: [https://ojs.aaai.org/index.php/AAAI/article/view/34664](https://ojs.aaai.org/index.php/AAAI/article/view/34664)






## Project Structure

```
.
├── data/
│   ├── dataset/
│   │   └── math/              # Math dataset files
│   ├── dataset_knowledge_exact/  # Knowledge base
│   ├── tpot/
│   │   ├── called_tools/      # Tool usage records
│   │   ├── output/           # Output results
│   │   └── result/           # Evaluation results
│   └── toolset/
│       ├── initial/          # Initial toolset
│       └── optimize/         # Optimized toolset
├── utils/
│   ├── api.py               # API utilities
│   ├── grader.py            # Answer grading
│   ├── parser.py            # Data parsing
│   └── utils.py             # General utilities
├── 0_get_kps.py             # Knowledge point extraction
├── 0_knowledge_extract.py    # Knowledge extraction
├── 0_processed.py           # Knowledge processing
├── 1_tool_creation_toolset_initial.py  # Initial toolset creation
├── 2_tool_creation_toolset_optimize.py # Tool optimization
├── 2_tool_creation_toolset_potimize_run_script.py  # Optimization script
├── 3_get_final_toolset.py   # Final toolset generation
├── 3_testdatset_parse.py    # Test dataset parsing
├── run_ktce_1_tool_retriever.py  # Tool retrieval
├── run_ktce_2_run.py        # Main execution
├── run_ktce_3_eval.py       # Evaluation
└── README.md
```

## Data Download

Before running the code, you need to download the complete dataset. Please follow these steps:

1. Download the complete dataset from our Google Drive: [Download Dataset](https://drive.google.com/file/d/1M6g-ywH_7tOHM_eZyUAfXcKJ2Q80CYSv/view?usp=sharing)
2. Extract the downloaded files
3. Replace the contents of the `data/` directory with the extracted files


## Usage

The framework consists of the following steps:

### 0. Knowledge Point Extraction and Processing
```bash
# Extract knowledge points
python 0_get_kps.py
# Process knowledge
python 0_processed.py
# Extract knowledge
python 0_knowledge_extract.py
```
These steps extract and process knowledge points from the dataset to create the knowledge base.

### 1. Tool Creation and Initialization
```bash
# Create initial toolset
python 1_tool_creation_toolset_initial.py
```
This step creates the initial toolset by:
- Extracting knowledge from problems
- Generating atomic tools
- Validating tool functionality
- Creating the initial toolset structure

### 2. Tool Optimization
```bash
# Run tool optimization
python 2_tool_creation_toolset_potimize_run_script.py
```
This step optimizes the tools through:
- Tool selection and evaluation
- Tool mutation and crossover
- Performance assessment
- Toolset evolution

### 3. Final Toolset Generation
```bash
# Generate final toolset
python 3_get_final_toolset.py
```
These steps:
- Generate the final optimized toolset
- Process test datasets
- Prepare for evaluation

### 4. Tool Usage and Evaluation
```bash
# Tool retrieval
python run_ktce_1_tool_retriever.py
# Main execution
python run_ktce_2_run.py [test_field]
# Evaluation
python run_ktce_3_eval.py
```
These steps:
- Retrieve appropriate tools for given problems
- Execute the reasoning process
- Evaluate the performance

Available test fields:
- Algebra
- Intermediate Algebra
- Prealgebra
- Geometry
- Counting & Probability
- Precalculus
- Number Theory

## Results

Our approach achieves substantial accuracy improvements ranging from 6.23% to 18.49% on average across challenging mathematical/tabular/scientific reasoning tasks. The toolkit demonstrates superior characteristics including:
- High reusability
- High diversity
- High generalizability on cross-data/LLM performance
- Low complexity

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{KTCE, 
  title={Automated Creation of Reusable and Diverse Toolsets for Enhancing LLM Reasoning}, 
  volume={39}, 
  url={https://ojs.aaai.org/index.php/AAAI/article/view/34664}, 
  DOI={10.1609/aaai.v39i23.34664}, 
  number={23}, 
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  author={Ma, Zhiyuan and Huang, Zhenya and Liu, Jiayu and Wang, Minmao and Zhao, Hongke and Li, Xin}, 
  year={2025}, 
  month={Apr.}, 
  pages={24821-24830} 
}
```