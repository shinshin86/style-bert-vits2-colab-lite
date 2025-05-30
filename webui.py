import os
import shutil
import subprocess
import sys
import yaml
import gradio as gr

# --------------------------------------------------
# 固定パス
# --------------------------------------------------
REPO_ROOT = "/content/Style-Bert-VITS2"
DATASET_ROOT = "/content/dataset/Style-Bert-VITS2/Data"
ASSETS_ROOT = "/content/dataset/Style-Bert-VITS2/model_assets"
INPUT_DIR = "/content/dataset/Style-Bert-VITS2/inputs"

for p in (DATASET_ROOT, ASSETS_ROOT, INPUT_DIR):
    os.makedirs(p, exist_ok=True)

if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# 遅延 import
from gradio_tabs.train import preprocess_all, get_path  # type: ignore
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker  # type: ignore

# --------------------------------------------------
# Helper
# --------------------------------------------------
def _run(cmd: list[str]):
    """subprocess を呼び出し、エラー時は例外を出す (check=True)。"""
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


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
    """Gradio から呼ばれるメイン処理。ZIP とログを返す。"""

    # 入力 WAV を配置
    for f in files:
        dst = os.path.join(INPUT_DIR, os.path.basename(f.name))
        shutil.copy(f.name, dst)
    logs = [f"{len(files)} 音声ファイルを {INPUT_DIR} にコピーしました。"]

    # paths.yml を書き出し
    with open(os.path.join(REPO_ROOT, "configs/paths.yml"), "w", encoding="utf-8") as fp:
        yaml.dump({"dataset_root": DATASET_ROOT, "assets_root": ASSETS_ROOT}, fp)

    # 前処理
    _run(["python", "slice.py", "-i", INPUT_DIR, "--model_name", model_name])
    _run(
        [
            "python",
            "transcribe.py",
            "--model_name",
            model_name,
            "--initial_prompt",
            initial_prompt,
            "--use_hf_whisper",
        ]
    )

    # リサンプリング処理を明示的に追加（raw/ -> wavs/）
    raw_dir = os.path.join(DATASET_ROOT, model_name, "raw")
    wavs_dir = os.path.join(DATASET_ROOT, model_name, "wavs")
    resample_cmd = ["python", "resample.py", "--input_dir", raw_dir, "--output_dir", wavs_dir]
    if normalize:
        resample_cmd.append("--normalize")
    if trim:
        resample_cmd.append("--trim")
    _run(resample_cmd)

    # Python API での前処理
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

    # config.yml / パスの取得
    paths = get_path(model_name)
    dataset_path = str(paths.dataset_path)
    config_path = str(paths.config_path)

    # default_config.yml をコピーして model_name を書き換えた config.yml を用意
    with open(os.path.join(REPO_ROOT, "default_config.yml"), "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    cfg["model_name"] = model_name
    with open(os.path.join(REPO_ROOT, "config.yml"), "w", encoding="utf-8") as fp:
        yaml.dump(cfg, fp, allow_unicode=True)

    # 生成された config.json の存在確認
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"学習用 config.json が見つかりません: {config_path}\n"
            "◆ 入力 WAV が不足していないか確認\n"
            "◆ preprocess が正常終了しているかログを確認\n"
        )

    # 学習
    if use_jp_extra:
        _run(
            [
                "python",
                "train_ms_jp_extra.py",
                "--config",
                config_path,
                "--model",
                dataset_path,
                "--assets_root",
                ASSETS_ROOT,
            ]
        )
    else:
        _run(
            [
                "python",
                "train_ms.py",
                "--config",
                config_path,
                "--model",
                dataset_path,
                "--assets_root",
                ASSETS_ROOT,
            ]
        )

    # ONNX 変換
    if not os.path.exists(os.path.join(REPO_ROOT, "convert_onnx.py")):
        _run(["git", "fetch", "-p"])
        _run(["git", "checkout", "dev"])
        _run(["pip", "install", "-r", "requirements-colab.txt", "--timeout", "120"])

    _run(["python", "convert_onnx.py", "--model", os.path.join(ASSETS_ROOT, model_name)])

    # ZIP 化
    zip_path = os.path.join(ASSETS_ROOT, f"{model_name}.zip")
    shutil.make_archive(zip_path[:-4], "zip", os.path.join(ASSETS_ROOT, model_name))

    logs.append("処理が完了しました！")
    return zip_path, "\n".join(logs)


# --------------------------------------------------
# Gradio UI 定義
# --------------------------------------------------
with gr.Blocks(title="Style-Bert-VITS2 Trainer") as demo:

    gr.Markdown("""
    # Style‑Bert‑VITS2 学習 & ONNX 変換 Web UI
    1. 必要なパラメータを入力 / 選択します。
    2. "音声ファイル" に WAV をドラッグ & ドロップします。
    3. **[学習・変換を実行]** ボタンを押すと、一括で処理が走ります。
    4. 完了後に ZIP がダウンロード出来ます。
    """)

    with gr.Row():
        model_name_in = gr.Text(label="モデル名", value="your_model_name")
        initial_prompt_in = gr.Text(
            label="初期プロンプト",
            value="こんにちは。元気、ですかー？ふふっ、私は……ちゃんと元気だよ！",
        )

    with gr.Row():
        use_jp_extra_in = gr.Checkbox(label="JP‑Extra を有効化", value=True)
        normalize_in = gr.Checkbox(label="音量正規化", value=False)
        trim_in = gr.Checkbox(label="無音トリム", value=False)

    with gr.Row():
        batch_size_in = gr.Number(label="バッチサイズ", value=4, precision=0)
        epochs_in = gr.Number(label="エポック数", value=100, precision=0)
        save_every_in = gr.Number(label="保存頻度 (steps)", value=1000, precision=0)

    yomi_error_in = gr.Dropdown(
        label="読みエラー時の挙動", choices=["raise", "skip", "use"], value="skip"
    )

    audio_files_in = gr.Files(label="音声ファイル (wav)", file_types=["audio"])

    run_btn = gr.Button("学習・変換を実行", variant="primary")

    result_zip_out = gr.File(label="結果 ZIP")
    logs_out = gr.Textbox(label="ステータス / ログ", lines=15)

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Gradio share モードを有効化")
    args = parser.parse_args()

    demo.launch(share=args.share, allowed_paths=[ASSETS_ROOT])
