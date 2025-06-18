import os
import shutil
import subprocess
import sys
import yaml
import gradio as gr

# Style-Bert-VITS2のリポジトリがCloneされる場所
REPO_ROOT = "/content/Style-Bert-VITS2"

# REPO_ROOTが存在することを確認
if not os.path.exists(REPO_ROOT):
    raise RuntimeError(f"Style-Bert-VITS2のリポジトリが見つかりません: {REPO_ROOT}")

# 最初にREPO_ROOTに移動
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

# 必要なディレクトリを作成
dataset_root = "/content/dataset/Style-Bert-VITS2/Data"
assets_root = "/content/dataset/Style-Bert-VITS2/model_assets"
input_dir = "/content/dataset/Style-Bert-VITS2/inputs"

os.makedirs(dataset_root, exist_ok=True)
os.makedirs(assets_root, exist_ok=True)
os.makedirs(input_dir, exist_ok=True)
os.makedirs("configs", exist_ok=True)

# paths.ymlを作成
with open("configs/paths.yml", "w", encoding="utf-8") as f:
    yaml.dump({"dataset_root": dataset_root, "assets_root": assets_root}, f)

# 必要なモジュールをインポート
from gradio_tabs.train import preprocess_all, get_path
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker


def run_pipeline(
    model_name: str,
    initial_prompt: str,
    use_jp_extra: bool,
    batch_size: int,
    epochs: int,
    save_every_steps: int,
    normalize: bool,
    trim: bool,
    yomi_error: str,
    files: list[gr.File],
):
    """Gradioから呼ばれるメイン処理"""
    logs = []
    
    try:
        # 1. 音声ファイルをinput_dirにコピー
        logs.append(f"📁 音声ファイルをコピー中...")
        for f in files:
            dst = os.path.join(input_dir, os.path.basename(f.name))
            shutil.copy(f.name, dst)
        logs.append(f"✅ {len(files)}個の音声ファイルをコピーしました")
        
        # 2. slice.pyを実行
        logs.append(f"\n🔄 音声を分割中...")
        result = subprocess.run(
            ["python", "slice.py", "-i", input_dir, "--model_name", model_name],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logs.append(f"❌ エラー: {result.stderr}")
            return None, "\n".join(logs)
        logs.append("✅ 音声分割完了")
        
        # 3. transcribe.pyを実行
        logs.append(f"\n🔄 音声を書き起こし中...")
        result = subprocess.run(
            [
                "python", "transcribe.py", 
                "--model_name", model_name,
                "--initial_prompt", initial_prompt,
                "--use_hf_whisper"
            ],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logs.append(f"❌ エラー: {result.stderr}")
            return None, "\n".join(logs)
        logs.append("✅ 書き起こし完了")
        
        # 4. 前処理を実行
        logs.append(f"\n🔄 前処理を実行中...")
        pyopenjtalk_worker.initialize_worker()
        
        preprocess_all(
            model_name=model_name,
            batch_size=batch_size,
            epochs=epochs,
            save_every_steps=save_every_steps,
            num_processes=2,
            normalize=normalize,
            trim=trim,
            freeze_EN_bert=False,
            freeze_JP_bert=False,
            freeze_ZH_bert=False,
            freeze_style=False,
            freeze_decoder=False,
            use_jp_extra=use_jp_extra,
            val_per_lang=0,
            log_interval=200,
            yomi_error=yomi_error,
        )
        logs.append("✅ 前処理完了")
        
        # 5. config.ymlを作成
        logs.append(f"\n🔄 設定ファイルを作成中...")
        paths = get_path(model_name)
        dataset_path = str(paths.dataset_path)
        config_path = str(paths.config_path)
        
        with open("default_config.yml", "r", encoding="utf-8") as f:
            yml_data = yaml.safe_load(f)
        yml_data["model_name"] = model_name
        with open("config.yml", "w", encoding="utf-8") as f:
            yaml.dump(yml_data, f, allow_unicode=True)
        logs.append("✅ 設定ファイル作成完了")
        
        # 6. 学習を実行
        logs.append(f"\n🚀 学習を開始中...")
        if use_jp_extra:
            cmd = [
                "python", "train_ms_jp_extra.py",
                "--config", config_path,
                "--model", dataset_path,
                "--assets_root", assets_root
            ]
        else:
            cmd = [
                "python", "train_ms.py",
                "--config", config_path,
                "--model", dataset_path,
                "--assets_root", assets_root
            ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logs.append(f"❌ エラー: {result.stderr}")
            return None, "\n".join(logs)
        logs.append("✅ 学習完了")
        
        # 7. ONNX変換
        logs.append(f"\n🔄 ONNX変換中...")
        # onnx変換用のブランチをチェックアウト
        if not os.path.exists("convert_onnx.py"):
            subprocess.run(["git", "fetch", "-p"], check=True)
            subprocess.run(["git", "checkout", "dev"], check=True)
            subprocess.run(["pip", "install", "-r", "requirements-colab.txt", "--timeout", "120"], check=True)
        
        result = subprocess.run(
            ["python", "convert_onnx.py", "--model", os.path.join(assets_root, model_name)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logs.append(f"❌ エラー: {result.stderr}")
            return None, "\n".join(logs)
        logs.append("✅ ONNX変換完了")
        
        # 8. ZIP化
        logs.append(f"\n📦 モデルをZIP化中...")
        zip_path = os.path.join(assets_root, f"{model_name}.zip")
        shutil.make_archive(
            zip_path[:-4], 
            "zip", 
            os.path.join(assets_root, model_name)
        )
        logs.append(f"✅ ZIP化完了: {model_name}.zip")
        
        logs.append(f"\n🎉 すべての処理が完了しました！")
        return zip_path, "\n".join(logs)
        
    except Exception as e:
        logs.append(f"\n❌ 予期しないエラー: {str(e)}")
        return None, "\n".join(logs)


# Gradio UI
with gr.Blocks(title="Style-Bert-VITS2 Trainer") as demo:
    gr.Markdown("""
    # Style-Bert-VITS2 学習 Web UI
    
    1. 必要なパラメータを入力してください
    2. 音声ファイル（WAV形式）をアップロードしてください
    3. 「学習を開始」ボタンを押すと処理が始まります
    4. 完了後、学習済みモデルのZIPファイルをダウンロードできます
    """)
    
    with gr.Row():
        with gr.Column():
            model_name_in = gr.Text(
                label="モデル名",
                value="your_model_name",
                info="英数字とアンダースコアのみ使用可能"
            )
            initial_prompt_in = gr.Text(
                label="初期プロンプト",
                value="こんにちは。元気、ですかー？ふふっ、私は……ちゃんと元気だよ！",
                info="書き起こしの例文（句読点や固有名詞の書き方の参考）"
            )
        
        with gr.Column():
            use_jp_extra_in = gr.Checkbox(
                label="JP-Extraを有効化",
                value=True,
                info="日本語特化版（英語・中国語は使用不可）"
            )
            batch_size_in = gr.Number(
                label="バッチサイズ",
                value=4,
                precision=0,
                info="VRAMに応じて調整"
            )
            epochs_in = gr.Number(
                label="エポック数",
                value=100,
                precision=0,
                info="学習の繰り返し回数"
            )
    
    with gr.Row():
        with gr.Column():
            normalize_in = gr.Checkbox(label="音量正規化", value=False)
            trim_in = gr.Checkbox(label="無音トリム", value=False)
        
        with gr.Column():
            save_every_in = gr.Number(
                label="保存頻度（ステップ）",
                value=1000,
                precision=0
            )
            yomi_error_in = gr.Dropdown(
                label="読みエラー時の挙動",
                choices=["raise", "skip", "use"],
                value="skip",
                info="raise:エラーで停止, skip:該当行をスキップ, use:無理やり使用"
            )
    
    audio_files_in = gr.Files(
        label="音声ファイル（WAV形式）",
        file_types=["audio"],
        file_count="multiple"
    )
    
    run_btn = gr.Button("🚀 学習を開始", variant="primary", size="lg")
    
    with gr.Row():
        result_zip_out = gr.File(label="学習済みモデル（ZIP）")
        logs_out = gr.Textbox(
            label="処理ログ",
            lines=20,
            max_lines=30,
            interactive=False
        )
    
    run_btn.click(
        fn=run_pipeline,
        inputs=[
            model_name_in,
            initial_prompt_in,
            use_jp_extra_in,
            batch_size_in,
            epochs_in,
            save_every_in,
            normalize_in,
            trim_in,
            yomi_error_in,
            audio_files_in,
        ],
        outputs=[result_zip_out, logs_out],
    )
    
    gr.Markdown("""
    ---
    ### 使い方のヒント
    - 音声ファイルは10秒以上、合計10分以上推奨
    - バッチサイズはVRAMが足りない場合は小さくしてください
    - エポック数は100で十分な場合が多いです
    """)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Gradio shareモードを有効化")
    args = parser.parse_args()
    
    demo.launch(share=args.share, allowed_paths=[assets_root])