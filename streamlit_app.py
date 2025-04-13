import streamlit as st
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import tempfile
import traceback
import ast
import re

st.set_page_config(page_title="Notebook Metric Evaluator")
st.title("📘 ML Notebook Evaluator")

uploaded_file = st.file_uploader("Upload a `.ipynb` file", type="ipynb")

def extract_metric(nb):
    for cell in nb.cells:
        if cell.cell_type == "code" and "outputs" in cell:
            for output in cell.outputs:
                # Get output text from stream or execute_result
                text = output.get("text", "") or output.get("data", {}).get("text/plain", "")
                text = text.strip()

                # Try to parse structured result (dict or float)
                try:
                    result = ast.literal_eval(text)
                    if isinstance(result, dict):
                        if 'accuracy' in result:
                            return {"type": "classification", "accuracy": result['accuracy']}
                        elif 'rmse' in result:
                            return {"type": "regression", "rmse": result['rmse']}
                    elif isinstance(result, (int, float)):
                        return {
                            "type": "classification" if result <= 1 else "regression",
                            "accuracy" if result <= 1 else "rmse": result
                        }
                except:
                    pass

                # Fallback: regex extraction from string output
                acc_match = re.search(r'accuracy[^:\d]*[:\s]+([\d.]+)', text, re.I)
                rmse_match = re.search(r'rmse[^:\d]*[:\s]+([\d.]+)', text, re.I)
                if acc_match:
                    return {"type": "classification", "accuracy": float(acc_match.group(1))}
                if rmse_match:
                    return {"type": "regression", "rmse": float(rmse_match.group(1))}

    return {"error": "No metric (accuracy or rmse) found in notebook outputs."}

if uploaded_file:
    with st.spinner("⏳ Executing notebook..."):
        try:
            # Save uploaded notebook to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".ipynb") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Load the notebook
            with open(tmp_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)

            # Execute the notebook
            ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
            ep.preprocess(nb, {'metadata': {'path': './'}})

            # Extract metric from executed notebook
            metric = extract_metric(nb)

            st.success("✅ Notebook executed successfully!")
            st.subheader("📊 Metric Output:")
            st.json(metric)

        except Exception as e:
            st.error("❌ Error while processing the notebook.")
            st.text(traceback.format_exc())
