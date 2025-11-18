import argparse
from src.model import AllMAssistant

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='gpt2')
    parser.add_argument('--prompt', default='Create a 10-minute beginner home workout for fat loss.')
    parser.add_argument('--max_new_tokens', type=int, default=200)
    args = parser.parse_args()

    assistant = AllMAssistant(args.model_dir)
    out = assistant.generate(args.prompt, max_new_tokens=args.max_new_tokens)
    print(out)

if __name__ == '__main__':
    main()
