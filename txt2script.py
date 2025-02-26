import re
import os
import argparse
import torch

from handwriting_synthesis import utils
from handwriting_synthesis.sampling import HandwritingSynthesizer


def chunk_text(text, max_words_per_line=10):
    words = text.split()
    chunks = []
    current_chunk = []
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) == max_words_per_line:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts a text file into a handwriting page.')
    parser.add_argument("model_path", type=str, help="Path to saved model")
    parser.add_argument(
        "input_path", type=str, help="A path to a text file that needs to be converted to handwriting")
    parser.add_argument(
        "-b", "--bias", type=float, default=0, help="A probability bias. Unbiased sampling is performed by default."
    )

    parser.add_argument("--output_path", type=str, default='',
                        help="Path to the generated handwriting file "
                             "(by default, it will be saved to the current working directory "
                             "with the input file name and a .png extension)")

    parser.add_argument(
        "--thickness", type=int, default=10,
        help="Handwriting thickness in pixels. Default is 10."
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input_path):
        raise FileNotFoundError(f'Text file not found: {args.input_path}')

    base_file_name = os.path.splitext(os.path.basename(args.input_path))[0]
    output_path = args.output_path or f'{base_file_name}.png'
    thickness = args.thickness

    device = torch.device("cpu")
    synthesizer = HandwritingSynthesizer.load(args.model_path, device, args.bias)

    with open(args.input_path) as f:
        text = f.read()

    processed_text = chunk_text(text, max_words_per_line=10)  # Chunk text into lines of up to 10 words

    # Generate handwriting for each line of text
    for i, line in enumerate(processed_text):
        line_output_path = f"{output_path[:-4]}_{i + 1}.png"  # Modify output path for each line
        utils.text_to_script(synthesizer, line, line_output_path, thickness=thickness)
