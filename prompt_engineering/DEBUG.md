# Ultra-Simple Debug Setup

## How It Works
1. Open `prompt_runner.py` (or any Python file in the project)
2. Click line number to set breakpoint 
3. Press `F5` - VS Code will debug whatever file you have open
4. If the script needs arguments, VS Code will ask you to type them in

## That's It!
- **No need to configure every possible command**
- **No need to map every method**  
- **Just open the file you want to debug and press F5**

## Examples

### Quick Testing
- Open `prompt_runner.py`, press F5, type: `--dataset-type canned --num-samples 2 --strategy baseline`
- Open `prompt_runner.py`, press F5, type: `--dataset-type unified --num-samples 5 --strategy all`

### Component Testing  
- Open `prompts_validator.py`, press F5 (if it has a `if __name__ == "__main__"`)
- Open `strategy_templates_loader.py`, press F5 (to test template loading)
- Open `unified_dataset_loader.py`, press F5 (to test dataset loading)

### Common Debug Scenarios

1. **JSON Parsing Issues**: Set breakpoint in `prompts_validator.py` response parsing section
2. **Strategy Loading**: Set breakpoint in `strategy_templates_loader.py` 
3. **Dataset Loading**: Set breakpoint in `unified_dataset_loader.py`
4. **Metrics Calculation**: Set breakpoint in `evaluation_metrics_calc.py`
5. **File Saving**: Set breakpoint in `persistence_helper.py`

## Debugging "Fallback due to JSON parsing error"

1. Open `prompts_validator.py`
2. Set breakpoint where JSON parsing happens
3. Run with a sample that causes the error
4. Inspect the raw model response to see malformed JSON

## File Structure for Debugging

```
prompt_engineering/
├── prompt_runner.py              # Main CLI - debug entry point
├── prompts_validator.py          # Response parsing - debug JSON issues here  
├── strategy_templates_loader.py  # Strategy loading - debug template issues
├── unified_dataset_loader.py     # Dataset loading - debug data issues
├── evaluation_metrics_calc.py    # Metrics - debug calculation issues
└── persistence_helper.py         # File saving - debug output issues
```

The one configuration works for **everything**. Much simpler!