"""
Clean Unicode surrogate characters from Jupyter notebook outputs
This script fixes UnicodeEncodeError issues when exporting notebooks to HTML
"""

import json
import sys
from pathlib import Path

def clean_unicode(text):
    """Remove invalid Unicode surrogates"""
    if not isinstance(text, str):
        return text
    # Replace surrogates with replacement character
    return text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')

def clean_notebook_cell_output(output):
    """Clean a single cell output"""
    if isinstance(output, dict):
        # Clean text/plain and other text outputs
        if 'text' in output:
            if isinstance(output['text'], str):
                output['text'] = clean_unicode(output['text'])
            elif isinstance(output['text'], list):
                output['text'] = [clean_unicode(line) for line in output['text']]
        
        # Clean data outputs
        if 'data' in output and isinstance(output['data'], dict):
            for key, value in output['data'].items():
                if isinstance(value, str):
                    output['data'][key] = clean_unicode(value)
                elif isinstance(value, list):
                    output['data'][key] = [clean_unicode(v) if isinstance(v, str) else v for v in value]
    
    return output

def clean_notebook(notebook_path):
    """Clean all Unicode surrogates from notebook"""
    print(f"Cleaning notebook: {notebook_path}")
    
    with open(notebook_path, 'r', encoding='utf-8', errors='replace') as f:
        notebook = json.load(f)
    
    cleaned_count = 0
    
    # Clean each cell
    for cell in notebook.get('cells', []):
        # Clean cell source
        if 'source' in cell:
            if isinstance(cell['source'], str):
                original = cell['source']
                cell['source'] = clean_unicode(cell['source'])
                if original != cell['source']:
                    cleaned_count += 1
            elif isinstance(cell['source'], list):
                for i, line in enumerate(cell['source']):
                    original = line
                    cell['source'][i] = clean_unicode(line)
                    if original != cell['source'][i]:
                        cleaned_count += 1
        
        # Clean cell outputs
        if 'outputs' in cell:
            for output in cell['outputs']:
                original_output = str(output)
                cleaned_output = clean_notebook_cell_output(output)
                if str(cleaned_output) != original_output:
                    cleaned_count += 1
    
    # Save cleaned notebook
    backup_path = notebook_path.with_suffix('.ipynb.backup')
    print(f"Creating backup: {backup_path}")
    notebook_path.replace(backup_path)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    
    print(f"✓ Cleaned {cleaned_count} locations with invalid Unicode")
    print(f"✓ Notebook saved: {notebook_path}")
    print(f"✓ Backup saved: {backup_path}")

if __name__ == '__main__':
    notebook_file = Path('q:/workspace/HateSpeechDetection_ver2/eda/unified_dataset_eda.ipynb')
    
    if not notebook_file.exists():
        print(f"Error: Notebook not found: {notebook_file}")
        sys.exit(1)
    
    clean_notebook(notebook_file)
    print("\n🎉 Notebook cleaning complete!")
    print("You can now export to HTML without Unicode errors.")
