import streamlit as st
import whisper
import os
import tempfile
import requests
import pandas as pd
import numpy as np
import time
import re
from deep_translator import GoogleTranslator

# --- MoviePy 2.0+ 导入方式 (适配 Python 3.13) ---
from moviepy import VideoFileClip, CompositeVideoClip, ColorClip, VideoClip
from PIL import Image, ImageDraw, ImageFont

# --- 🛠️ 字体下载与加载 (修复 URL 和校验) ---
@st.cache_resource
def load_fonts():
    font_filename = "NotoSansCJKtc-Regular.otf"
    font_path = os.path.join(os.getcwd(), font_filename)
    
    # 修复了这里的 URL 拼写错误
    font_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"

    needs_download = False
    
    # 1. 检查文件是否存在
    if not os.path.exists(font_path):
        needs_download = True
    # 2. 检查文件大小 (防止下载了损坏的 0kb 文件)
    elif os.path.getsize(font_path) < 1024 * 100: # 小于 100KB 肯定是坏的
        st.warning("检测到旧字体文件损坏，正在删除并重新下载...")
        os.remove(font_path)
        needs_download = True

    if needs_download:
        with st.spinner("正在下载中文字体 (约 16MB，首次运行需 30秒)..."):
            try:
                # 增加超时设置
                r = requests.get(font_url, timeout=60)
                r.raise_for_status()
                with open(font_path, "wb") as f:
                    f.write(r.content)
                st.success("✅ 字体下载成功！")
                time.sleep(1) # 等待文件写入完成
            except Exception as e:
                st.error(f"❌ 字体下载失败: {e}")
                # 下载失败则清理垃圾文件
                if os.path.exists(font_path):
                    os.remove(font_path)
                return None

    return font_path

st.set_page_config(page_title="粤语视频工坊 Pro", layout="wide", page_icon="🎬")
st.title("🎬 粤语视频工坊 Pro (V7.0 最终完结版)")

# --- 辅助函数 ---
@st.cache_resource
def load_model():
    return whisper.load_model("base")

def get_jyutping_list(text):
    from ToJyutping import get_jyutping_list
    return get_jyutping_list(text)

def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', str(text)))

def context_aware_translate(current_text, prev_text=""):
    if not current_text or not current_text.strip(): return ""
    try:
        time.sleep(0.3)
        translator = GoogleTranslator(source='zh-TW', target='en')
        res = translator.translate(current_text)
        if res and res != current_text and not contains_chinese(res):
            return res
        return "[翻译失败，请手动修改]"
    except:
        pass
    return "[网络错误]"

# --- 智能换行绘制 (兼容 Pillow 10.0+) ---
def draw_text_wrapper(draw, text, font, max_width, start_y, color, line_spacing=10):
    if not text: return start_y
    lines = []
    
    # 英文/粤拼 (按空格换行)
    if ' ' in text and not contains_chinese(text):
        words = text.split(' ')
        current_line = []
        for word in words:
            test_line = ' '.join(current_line + [word])
            # 兼容性写法：检测 textlength 或 textsize
            try: w = draw.textlength(test_line, font=font)
            except AttributeError: w = draw.textsize(test_line, font=font)[0]
            
            if w <= max_width:
                current_line.append(word)
            else:
                if current_line: lines.append(' '.join(current_line)); current_line = [word]
                else: lines.append(word); current_line = []
        if current_line: lines.append(' '.join(current_line))
    else:
        # 中文 (按字换行)
        current_line = ""
        for char in text:
            test_line = current_line + char
            try: w = draw.textlength(test_line, font=font)
            except AttributeError: w = draw.textsize(test_line, font=font)[0]
            if w <= max_width: current_line += char
            else: lines.append(current_line); current_line = char
        if current_line: lines.append(current_line)

    current_y = start_y
    for line in lines:
        try: w = draw.textlength(line, font=font); h = font.size
        except AttributeError: w, h = draw.textsize(line, font=font)
        # 居中计算
        draw.text(((720 - w) / 2, current_y), line, font=font, fill=color)
        current_y += h + line_spacing
    return current_y

# --- 主程序逻辑 ---
if 'subtitles_df' not in st.session_state:
    st.session_state.subtitles_df = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None

with st.sidebar:
    st.header("1. 上传视频")
    uploaded_file = st.file_uploader("限制 200MB 以内", type=["mp4", "mov"])
    
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        st.session_state.video_path = tfile.name
        st.video(st.session_state.video_path)
        
        if st.button("🚀 开始识别与翻译", type="primary"):
            model = load_model()
            with st.status("AI 正在工作中...", expanded=True) as status:
                st.write("📂 提取音频...")
                # MoviePy 2.0+ 写法
                video = VideoFileClip(st.session_state.video_path)
                audio_path = "temp_audio.wav"
                try: video.audio.write_audiofile(audio_path, verbose=False, logger=None)
                except: video.audio.write_audiofile(audio_path)
                
                st.write("🧠 识别粤语...")
                result = model.transcribe(audio_path, language='Chinese')
                
                st.write("📝 生成初稿...")
                data = []
                prev_text = ""
                for seg in result['segments']:
                    txt = seg['text']
                    jp_list = get_jyutping_list(txt)
                    jp_str = " ".join([i[1] if i[1] else i[0] for i in jp_list])
                    eng = context_aware_translate(txt, prev_text)
                    prev_text = txt
                    data.append({
                        "start": round(seg['start'], 2),
                        "end": round(seg['end'], 2),
                        "text": txt,
                        "jyutping": jp_str,
                        "english": eng
                    })
                
                st.session_state.subtitles_df = pd.DataFrame(data)
                if os.path.exists(audio_path): os.remove(audio_path)
                status.update(label="✅ 完成！", state="complete", expanded=False)

# --- 编辑区域 ---
if st.session_state.subtitles_df is not None:
    st.divider()
    st.header("2. 智能校对")
    
    col_tip, col_btn = st.columns([3, 1])
    with col_tip:
        st.info("💡 直接修改下方表格。修改后点击「保存」或直接生成视频均可生效。")

    edited_df = st.data_editor(st.session_state.subtitles_df, num_rows="dynamic", use_container_width=True, key="editor")

    with col_btn:
        st.write("")
        if st.button("✨ 刷新翻译与粤拼"):
            with st.spinner("正在重新生成..."):
                updated_data = []
                prev_text = ""
                for index, row in edited_df.iterrows():
                    new_text = row['text']
                    jp_list = get_jyutping_list(new_text)
                    new_jp = " ".join([i[1] if i[1] else i[0] for i in jp_list])
                    new_eng = context_aware_translate(new_text, prev_text)
                    prev_text = new_text
                    updated_data.append({
                        "start": row['start'], "end": row['end'],
                        "text": new_text, "jyutping": new_jp, "english": new_eng
                    })
                st.session_state.subtitles_df = pd.DataFrame(updated_data)
                st.success("已更新！")
                st.rerun()

    if st.button("💾 保存当前修改"):
        st.session_state.subtitles_df = edited_df
        st.success("✅ 修改已保存！")

    st.divider()
    st.header("3. 视频合成")
    
    if st.button("🎬 生成视频"):
        # 1. 优先加载字体，失败则停止
        font_path = load_fonts()
        
        if font_path is None:
             st.error("❌ 无法生成：字体文件下载失败。请检查网络后刷新页面重试。")
        else:
            v_path = st.session_state.video_path
            # 使用最新编辑的数据
            if edited_df is not None:
                subs = edited_df.to_dict('records')
            else:
                subs = st.session_state.subtitles_df.to_dict('records')
            
            progress = st.progress(0)
            status = st.empty()
            
            try:
                status.text("正在初始化...")
                W, H = 720, 960
                padding = 50
                max_text_width = W - (padding * 2)
                
                clip = VideoFileClip(v_path)
                
                # MoviePy 2.0 resize 兼容写法
                try: clip = clip.resized(width=W)
                except AttributeError: clip = clip.resize(width=W)
                
                target_h = 500
                if clip.h > target_h:
                    clip = clip.crop(y1=(clip.h - target_h)/2, height=target_h)
                
                def make_frame(t):
                    img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
                    draw = ImageDraw.Draw(img)
                    cur = next((s for s in subs if s['start'] <= t <= s['end']), None)
                    nxt = next((s for s in subs if s['start'] > t), None)
                    
                    try:
                        # 加载字体 (增加 try-except 防止字体文件损坏导致崩溃)
                        f_cn = ImageFont.truetype(font_path, 52)
                        f_jp = ImageFont.truetype(font_path, 28)
                        f_en = ImageFont.truetype(font_path, 24)
                    except Exception as e:
                        print(f"Font Error: {e}")
                        f_cn = ImageFont.load_default()
                        f_jp = ImageFont.load_default()
                        f_en = ImageFont.load_default()
                    
                    cursor_y = target_h + 40
                    if cur:
                        cursor_y = draw_text_wrapper(draw, cur['text'], f_cn, max_text_width, cursor_y, "#FFD700", 12)
                        cursor_y += 12 
                        cursor_y = draw_text_wrapper(draw, cur['jyutping'], f_jp, max_text_width, cursor_y, "#87CEEB", 8)
                        cursor_y += 12
                        cursor_y = draw_text_wrapper(draw, str(cur['english']), f_en, max_text_width, cursor_y, "#FFFFFF", 8)
                    if nxt:
                        draw.text((50, 900), f"Next: {nxt['text']}", font=f_jp, fill="#555555")
                    return np.array(img)

                status.text("正在渲染 (约3分钟，请勿刷新)...")
                sub_clip = VideoClip(make_frame, duration=clip.duration)
                bg_clip = ColorClip(size=(W, H), color=(20, 20, 20), duration=clip.duration)
                
                # MoviePy 2.0 关键修正：使用 with_position
                final = CompositeVideoClip([
                    bg_clip,
                    clip.with_position(('center', 'top')), 
                    sub_clip.with_position('center')
                ])
                
                out_file = "cantonese_final_v7.mp4"
                final.write_videofile(out_file, fps=24, codec='libx264', audio_codec='aac', logger=None)
                
                status.success("完成！")
                progress.progress(100)
                with open(out_file, "rb") as f:
                    st.download_button("⬇️ 下载视频", f, file_name="cantonese_tutor_final.mp4")
                st.video(out_file)
                
            except Exception as e:
                st.error(f"合成出错: {e}")
                # 遇到错误尝试清除可能损坏的字体
                if os.path.exists(font_path):
                    os.remove(font_path)
