from flask import Flask, request, jsonify
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import uuid
import re

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_notebook():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    if not file.filename.endswith('.ipynb'):
        return jsonify({"error": "Only .ipynb files are allowed"}), 400

    file_id = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_FOLDER, file_id + ".ipynb")
    file.save(filepath)

    try:
        metric_name, metric_value = extract_performance_metric(filepath)
        return jsonify({metric_name: metric_value})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(filepath)

def extract_performance_metric(notebook_path):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': './'}})

    # Priority order
    classification_keywords = ['accuracy']
    regression_keywords = ['rmse', 'mean_squared_error', 'mse']
    all_keywords = classification_keywords + regression_keywords

    pattern = re.compile(r'(?P<key>' + '|'.join(all_keywords) + r')[\s:=]*([0]*\.?[0-9]+)', re.IGNORECASE)

    found_metrics = {}

    for cell in nb.cells:
        if cell.cell_type == 'code' and 'outputs' in cell:
            for output in cell.outputs:
                texts = []

                if output.output_type == 'stream':
                    texts.append(output.text)
                elif output.output_type == 'execute_result':
                    data = output.get('data', {})
                    if 'text/plain' in data:
                        texts.append(data['text/plain'])

                for text in texts:
                    for match in pattern.finditer(text):
                        key = match.group('key').lower()
                        value = float(match.group(2))
                        found_metrics[key] = value

    # Return based on priority
    for key in classification_keywords + regression_keywords:
        if key in found_metrics:
            return key, found_metrics[key]

    raise ValueError("No performance metric (accuracy or rmse/mse) found in notebook.")

if __name__ == '__main__':
    app.run(debug=True)
