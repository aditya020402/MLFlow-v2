from ada import run_pipeline

result = run_pipeline(
    dataset=r"C:\Users\shubh\Downloads\vscode\MLFlow-v2\Ship_Performance_Dataset.csv",
    instructions=r"C:\Users\shubh\Downloads\vscode\MLFlow-v2\instructions.md",
    output_dir="./ada_output",
    env_file=r"C:\Users\shubh\Downloads\vscode\MLFlow-v2\.env"
)

print(result["model_path"])   # path to best model.pkl
print(result["evaluation"])   # verdict, score, strengths, weaknesses
