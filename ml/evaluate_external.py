import argparse
import json

from ml.pipeline import evaluate_external, load_model

parser = argparse.ArgumentParser(description="Evaluate on a separately supplied external CSV")
parser.add_argument("--data", required=True)
parser.add_argument("--target", required=True)
parser.add_argument("--model", default=None)
args = parser.parse_args()
artifact = load_model(args.model) if args.model else load_model()
print(json.dumps(evaluate_external(args.data, args.target, artifact), indent=2))
