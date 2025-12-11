import streamlit as st
import whisper
import os
import tempfile
import requests
import pandas as pd
import numpy as np
import time
from deep_translator import GoogleTranslator

# --- MoviePy 2.0+ 导入方式 ---
from moviepy import VideoFileClip, CompositeVideoClip, ColorClip, VideoClip
from PIL import Image, ImageDraw, ImageFont

# --- 字体下载 ---
@st.cache_resource
def load_fonts():
    font_path = "NotoSansCJKtc-Regular.otf"
    if not os.path.exists(font_path):
        url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"
        with st.spinner("正在下载中文字体支持..."):
            try:
                r = requests.get(url, timeout=60)
                with open(font_path, "wb") as f:
                    f.write(r.content)
            except:
                st.error("字体下载失败，请检查网络连接。")
    return font_path

st.set_page_config(page_title="粤语视频工坊 Pro", layout="wide", page_icon="🎬")
st.title("🎬 粤语视频工坊 Pro (智能交互版)")

# --- 辅助函数 ---
@st.cache_resource
def load_model():
    return whisper.load_model("base")

def get_jyutping_list(text):
    from ToJyutping import get_jyutping_list
    return get_jyutping_list(text)

def safe_translate(text):
    try:
        # 避免频繁请求
        time.sleep(0.1)
        res = GoogleTranslator(source='zh-TW', target='en').translate(text)
        if res and res != text:
            return res
    except:
        pass
    return "[Translation Error]"

# --- 智能换行绘制函数 ---
def draw_text_wrapper(draw, text, font, max_width, start_y, color, line_spacing=10):
    if not text: return start_y
    lines = []
    if ' ' in text:
        words = text.split(' ')
        current_line = []
        for word in words:
            test_line = ' '.join(current_line + [word])
            try: w = draw.textlength(test_line, font=font)
            except AttributeError: w = draw.textsize(test_line, font=font)[0]
            if w <= max_width: current_line.append(word)
            else:
                if current_line: lines.append(' '.join(current_line)); current_line = [word]
                else: lines.append(word); current_line = []
        if current_line: lines.append(' '.join(current_line))
    else: lines = [text]

    current_y = start_y
    for line in lines:
        try: w = draw.textlength(line, font=font); h = font.size
        except AttributeError: w, h = draw.textsize(line, font=font)
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
                video = VideoFileClip(st.session_state.video_path)
                audio_path = "temp_audio.wav"
                try: video.audio.write_audiofile(audio_path, verbose=False, logger=None)
                except: video.audio.write_audiofile(audio_path)
                
                st.write("🧠 识别粤语...")
                result = model.transcribe(audio_path, language='Chinese')
                
                st.write("📝 生成初稿...")
                data = []
                for seg in result['segments']:
                    txt = seg['text']
                    # 初次生成
                    jp_list = get_jyutping_list(txt)
                    jp_str = " ".join([i[1] if i[1] else i[0] for i in jp_list])
                    eng = safe_translate(txt)
                    
                    data.append({
                        "start": round(seg['start'], 2),
                        "end": round(seg['end'], 2),
                        "text": txt,
                        "jyutping": jp_str,
                        "english": eng
                    })
                
                st.session_state.subtitles_df = pd.DataFrame(data)
                if os.path.exists(audio_path): os.remove(audio_path)
                status.update(label="✅ 初稿完成！请在右侧校对。", state="complete", expanded=False)

# --- 校对与导出 ---
if st.session_state.subtitles_df is not None:
    st.divider()
    st.header("2. 智能校对")
    
    col_tip, col_btn = st.columns([3, 1])
    with col_tip:
        st.info("💡 操作技巧：只管修改「粤语汉字」列，改完点击右边的刷新按钮，英文和粤拼会自动修正！")
    
    # 允许用户编辑
    edited_df = st.data_editor(st.session_state.subtitles_df, num_rows="dynamic", use_container_width=True, key="editor")

    # --- 🆕 新增功能：一键重新翻译 ---
    with col_btn:
        st.write("") # 占位对齐
        if st.button("✨ 刷新翻译与粤拼", type="primary"):
            with st.spinner("正在根据您的修改重新生成..."):
                updated_data = []
                # 遍历用户编辑后的表格
                for index, row in edited_df.iterrows():
                    new_text = row['text']
                    
                    # 重新生成粤拼 (因为汉字变了，发音肯定变了)
                    jp_list = get_jyutping_list(new_text)
                    new_jp = " ".join([i[1] if i[1] else i[0] for i in jp_list])
                    
                    # 重新翻译英文 (因为汉字变了，意思肯定变了)
                    new_eng = safe_translate(new_text)
                    
                    updated_data.append({
                        "start": row['start'],
                        "end": row['end'],
                        "text": new_text,       # 使用修改后的汉字
                        "jyutping": new_jp,     # 新粤拼
                        "english": new_eng      # 新翻译
                    })
                
                # 更新 Session State 并强制刷新界面
                st.session_state.subtitles_df = pd.DataFrame(updated_data)
                st.success("✅ 已根据中文更新所有翻译！")
                st.rerun()

    st.divider()
    st.header("3. 视频合成")
    
    if st.button("🎬 生成视频"):
        font_path = load_fonts()
        v_path = st.session_state.video_path
        # 使用最新的数据进行合成
        subs = st.session_state.subtitles_df.to_dict('records')
        
        progress = st.progress(0)
        status = st.empty()
        
        try:
            status.text("正在初始化...")
            W, H = 720, 960
            padding = 60
            max_text_width = W - (padding * 2)
            
            clip = VideoFileClip(v_path)
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
                    f_cn = ImageFont.truetype(font_path, 55)
                    f_jp = ImageFont.truetype(font_path, 30)
                    f_en = ImageFont.truetype(font_path, 26)
                except:
                    f_cn = ImageFont.load_default(); f_jp = ImageFont.load_default(); f_en = ImageFont.load_default()
                
                cursor_y = target_h + 50
                if cur:
                    cursor_y = draw_text_wrapper(draw, cur['text'], f_cn, max_text_width, cursor_y, "#FFD700", 15)
                    cursor_y += 15 
                    cursor_y = draw_text_wrapper(draw, cur['jyutping'], f_jp, max_text_width, cursor_y, "#87CEEB", 10)
                    cursor_y += 15
                    cursor_y = draw_text_wrapper(draw, str(cur['english']), f_en, max_text_width, cursor_y, "#FFFFFF", 10)
                if nxt:
                    draw.text((50, 880), f"Next: {nxt['text']}", font=f_jp, fill="#555555")
                return np.array(img)

            status.text("正在渲染 (约3分钟)...")
            sub_clip = VideoClip(make_frame, duration=clip.duration)
            bg_clip = ColorClip(size=(W, H), color=(20, 20, 20), duration=clip.duration)
            final = CompositeVideoClip([bg_clip, clip.with_position(('center', 'top')), sub_clip.with_position('center')])
            
            out_file = "cantonese_final_v4.mp4"
            final.write_videofile(out_file, fps=24, codec='libx264', audio_codec='aac', logger=None)
            
            status.success("完成！")
            progress.progress(100)
            with open(out_file, "rb") as f:
                st.download_button("⬇️ 下载视频", f, file_name="cantonese_tutor_smart.mp4")
            st.video(out_file)
            
        except Exception as e:
            st.error(f"合成出错: {e}")
