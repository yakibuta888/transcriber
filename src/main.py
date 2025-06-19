import os
import PySimpleGUI as sg
import threading

from application.services.transcribe_service import TranscribeService

from settings import logger


# --------------------------------------
# 処理本体
# --------------------------------------
def task(window):
    window.write_event_value('-PROGRESS-', '処理開始')
    try:
        transcribe_service = TranscribeService(
            audio_file=infile,
            model=model,
            hf_token=hf_token
        )
        transcribe_service.transcribe_and_save(
            outdir=outdir,
            outname=outname,
            option_args=option_args
        )
    except ValueError as e:
        window.write_event_value('-ERROR-', 'モデル選択が不正です。管理者にお問い合わせください。')
        logger.error(f"Transcription failed. @main.task: {e}")
        return
    except RuntimeError as e:
        window.write_event_value('-ERROR-', '音声認識に失敗しました。入力ファイルを確認してください。\n解決しない場合は管理者にお問い合わせください。')
        logger.error(f"Transcription failed. @main.task: {e}")
        return
    except FileNotFoundError as e:
        window.write_event_value('-ERROR-', '入力ファイルが見つかりません。ファイルパスを確認してください。\n解決しない場合は管理者にお問い合わせください。')
        logger.error(f"File not found. @main.task: {e}")
        return
    except Exception as e:
        window.write_event_value('-ERROR-', f'予期しないエラーが発生しました。管理者にお問い合わせください。')
        logger.error(f"Unexpected error occurred. @main.task: {e}")
    window.write_event_value('-PROGRESS-', '処理完了')


# --------------------------------------
# 詳細オプション
# --------------------------------------
detail_options = {
    'add_punctuation': True,
    'num_speakers': '',
    'min_speakers': '',
    'max_speakers': '',
    'add_silence_start': '',
    'add_silence_end': '',
    'chunk_length': '15',
    'batch_size': '8',
    'fa2': False,
    'hf_token': ''
}
def show_detail_window(current_options):
    # 詳細オプションウィンドウのレイアウト
    layout = [
        [sg.Checkbox('句読点を付与(kotoba-whisperのみ)', default=current_options.get('add_punctuation', True), key='-ADD_PUNCTUATION-')],
        [sg.Text('話者数', size=(27, 1)), sg.InputText(current_options.get('num_speakers', ''), key='-NUM_SPEAKERS-')],
        [sg.Text('話者数の最小値', size=(27, 1)), sg.InputText(current_options.get('min_speakers', ''), key='-MIN_SPEAKERS-')],
        [sg.Text('話者数の最大値', size=(27, 1)), sg.InputText(current_options.get('max_speakers', ''), key='-MAX_SPEAKERS-')],
        [sg.Text('音声の先頭に追加する無音の秒数\n(kotoba-whisperのみ)', size=(27, 2)), sg.InputText(current_options.get('add_silence_start', ''), key='-ADD_SILENCE_START-')],
        [sg.Text('音声の末尾に追加する無音の秒数\n(kotoba-whisperのみ)', size=(27, 2)), sg.InputText(current_options.get('add_silence_end', ''), key='-ADD_SILENCE_END-')],
        [sg.Text('音声チャンク長', size=(27, 1)), sg.InputText(current_options.get('chunk_length', '15'), key='-CHUNK_LENGTH-')],
        [sg.Text('バッチサイズ', size=(27, 1)), sg.InputText(current_options.get('batch_size', '8'), key='-BATCH_SIZE-')],
        [sg.Checkbox('Flash Attention 2 を利用する', default=current_options.get('fa2', False), key='-FA2-')],
        [sg.Text('Hugging Face トークン', size=(27, 1)), sg.InputText(current_options.get('hf_token', ''), key='-HF_TOKEN-')],
        [sg.Text('※詳細オプションはモデルによって異なる場合があります。')],
        [sg.Button('OK'), sg.Button('キャンセル')]
    ]
    window = sg.Window('詳細オプション設定', layout, modal=True)
    result = window.read()
    if result is None:
        window.close()
        return current_options
    event, values = result
    if event == 'OK':
        # サブウィンドウで設定した値を返す
        updated = {
            'add_punctuation': values['-ADD_PUNCTUATION-'],
            'num_speakers': values['-NUM_SPEAKERS-'],
            'min_speakers': values['-MIN_SPEAKERS-'],
            'max_speakers': values['-MAX_SPEAKERS-'],
            'add_silence_start': values['-ADD_SILENCE_START-'],
            'add_silence_end': values['-ADD_SILENCE_END-'],
            'chunk_length': values['-CHUNK_LENGTH-'],
            'batch_size': values['-BATCH_SIZE-'],
            'fa2': values['-FA2-'],
            'hf_token': values['-HF_TOKEN-']
        }
        window.close()
        return updated
    elif event == sg.WIN_CLOSED or event == 'キャンセル':
        # キャンセルまたはウィンドウ閉じる場合は元の値を返す
        window.close()
        return current_options
    else:
        # 何かしらのエラーが発生した場合は元の値を返す
        window.write_event_value('-ERROR-', '詳細オプションの設定に失敗しました。')
        window.close()
        return current_options


# --------------------------------------
# メインウィンドウ
# --------------------------------------

# メインウィンドウのレイアウト
sg.theme('Dark Blue 3')

layout = [
    [sg.Text('モデル', size=(15, 1)), sg.Combo(['whisper-large-v3', 'kotoba-whisper-v2.2'], default_value='whisper-large-v3', key='-MODEL-')],
    [sg.Text('入力ファイル', size=(15, 1)), sg.InputText(key='-INFILE-'), sg.FileBrowse('ファイル選択')],
    [sg.Checkbox('出力先を入力ファイルと同じフォルダにする', default=True, key='-SAMEFOLDER-', enable_events=True)],
    [sg.Text('出力先フォルダ', size=(15, 1)), sg.InputText(key='-OUTDIR-', disabled=True), sg.FolderBrowse('フォルダ選択', target='-OUTDIR-', key='-OUTDIR_BROWSE-', disabled=True)],
    [sg.Text('出力ファイル名', size=(15, 1)), sg.InputText('', key='-OUTNAME-')],
    [sg.Button('詳細オプション'), sg.Text('', key='-DETAIL_OPTIONS-')],
    [sg.Button('実行'), sg.Button('終了')],
    [sg.Text('ログ', size=(5, 1)), sg.Multiline(size=(55, 5), key='-LOG-', disabled=True, autoscroll=True)]
]

# ウィンドウ生成
window = sg.Window('文字起こしApp', layout)

# イベントループ
task_running = False
while True:
    result = window.read(timeout=100)
    if result is None:
        continue
    event, values = result
    if event in (sg.WIN_CLOSED, '終了'):
        break
    
    # 入力ファイル選択時 or チェックボックスON時に出力先フォルダを自動設定
    if event in ('-INFILE-', '-SAMEFOLDER-'):
        if values['-SAMEFOLDER-']:
            infile = values['-INFILE-']
            window['-OUTDIR-'].update(disabled=True)
            window['-OUTDIR_BROWSE-'].update(disabled=True)
        else:
            window['-OUTDIR-'].update(disabled=False)
            window['-OUTDIR_BROWSE-'].update(disabled=False)

    if event == '詳細オプション':
        detail_options = show_detail_window(detail_options)
        detail_value = f"""句読点付与: {'ON' if detail_options.get('add_punctuation') else 'OFF'}
話者数: {detail_options.get('num_speakers', '自動')}
話者数の最小値: {detail_options.get('min_speakers', '自動')}
話者数の最大値: {detail_options.get('max_speakers', '自動')}
音声の先頭に追加する無音: {detail_options.get('add_silence_start', '')}秒
音声の末尾に追加する無音: {detail_options.get('add_silence_end', '')}秒
音声チャンク長: {detail_options.get('chunk_length', '')}秒
バッチサイズ: {detail_options.get('batch_size', '')}
Flash Attention 2: {'ON' if detail_options.get('fa2') else 'OFF'}
Hugging Face トークン: {detail_options.get('hf_token', '未設定')}
"""
        window['-DETAIL_OPTIONS-'].update(value=detail_value)

    if event == '実行' and not task_running:
        task_running = True
        window['-LOG-'].update(value='処理を開始します...\n', append=True)

        # 入力値取得
        infile = values['-INFILE-']
        model = values['-MODEL-']
        # チェックボックスの状態で出力先フォルダを決定
        if values['-SAMEFOLDER-']:
            outdir = os.path.dirname(infile) if infile else os.path.dirname(os.path.abspath(__file__))
        else:
            outdir = values['-OUTDIR-']
        outname = values['-OUTNAME-']
        add_punctuation = detail_options['add_punctuation']
        num_speakers = detail_options['num_speakers']
        min_speakers = detail_options['min_speakers']
        max_speakers = detail_options['max_speakers']
        add_silence_start = detail_options['add_silence_start']
        add_silence_end = detail_options['add_silence_end']
        chunk_length = detail_options['chunk_length']
        batch_size = detail_options['batch_size']
        fa2 = detail_options['fa2']
        hf_token = detail_options['hf_token']

        if not infile:
            sg.popup_error('入力ファイルが指定されていません。')
            task_running = False
            continue

        # オプション引数の設定
        option_args = {
            'add_punctuation': add_punctuation,
            'num_speakers': int(num_speakers) if num_speakers else None,
            'min_speakers': int(min_speakers) if min_speakers else None,
            'max_speakers': int(max_speakers) if max_speakers else None,
            'add_silence_start': float(add_silence_start) if add_silence_start else None,
            'add_silence_end': float(add_silence_end) if add_silence_end else None,
            'chunk_length': int(chunk_length) if chunk_length else 15,
            'batch_size': int(batch_size) if batch_size else None,
            'fa2': fa2,
        }

        threading.Thread(target=task, args=(window,), daemon=True).start()
        
    if event == '-PROGRESS-':
        progress_message = values['-PROGRESS-']
        if progress_message == '処理開始':
            window['-LOG-'].update(value='処理を開始しました...\n', append=True)
        elif progress_message == '処理完了':
            window['-LOG-'].update(value='処理が完了しました。\n', append=True)
            task_running = False
    
    if event == '-ERROR-':
        error_message = values['-ERROR-']
        window['-LOG-'].update(value=f'エラー: {error_message}\n', append=True)
        task_running = False

window.close()
