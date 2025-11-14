# utils/io_utils.py
import os
import argparse

def parse_io_args(description="入出力の共通引数を処理"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--input", required=True, help="入力ファイルまたはフォルダのパス")
    parser.add_argument("-o", "--output", required=False, default="outputs", help="出力ディレクトリパス")
    args = parser.parse_args()

    # 絶対パスに変換
    input_path = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)

    # 出力フォルダ作成
    os.makedirs(output_dir, exist_ok=True)

    return input_path, output_dir