import pandas as pd
import json

df = pd.read_csv('results/variantbench_100_gold.csv')
gold_labels = df['label'].astype(str).tolist()


def automate_scoring(output_file_path: str):
    try:
        opened_file = open(output_file_path, 'r', encoding="utf-8")
    except FileNotFoundError:
        print(f"Error: The file {output_file_path} does not exist.")
        return

    with opened_file as file:
        try:
            labels = [json.loads(line.strip())['label'] for line in file]
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON from the output file. Could not be in a format where 'label' is a key.")
            return

    if len(labels) != len(gold_labels):
        raise ValueError("The number of labels in the output file does not match the number of gold labels.")

    correct_predictions = 0
    
    for label, gold in zip(labels, gold_labels):
        if label == gold:
            print(f"Correct: {label}")
            correct_predictions += 1
        else:
            print(f"Incorrect prediction: {label} | Gold: {gold}")

    accuracy = correct_predictions / len(gold_labels)
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Correct Predictions: {correct_predictions}")

if __name__ == "__main__":
    output_file_path = 'Claude_experiments/claude-3-haiku-20240307_parsed.jsonl'
    automate_scoring(output_file_path)



    





