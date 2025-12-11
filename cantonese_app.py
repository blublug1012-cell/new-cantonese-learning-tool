import streamlit as st
import whisper
import os
import tempfile
import requests
import pandas as pd
import numpy as np
import time
from deep_translator import GoogleTranslator
from ToJyutping import get_jyutping_list
# 显式导入 moviepy 的组件
from moviepy.editor import VideoFileClip, CompositeVideoClip, ColorClip, VideoClip
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
st.title("🎬 粤语视频工坊 Pro (Python 3.9 修复版)")

# --- 缓存加载 Whisper 模型 ---
@st.cache_resource
def load_model():
    return whisper.load_model("base")

# --- 辅助：混合翻译函数 ---
def safe_translate(text):
    # 1. 尝试 Google
    try:
        time.sleep(0.3) # 防封停
        # 强制指定源语言为繁体中文(zh-TW)，目标为英文(en)
        res = GoogleTranslator(source='zh-TW', target='en').translate(text)
        # 如果翻译结果不为空且不等于原文
        if res and res != text:
            return res
    except:
        pass
    
    # 2. 如果失败，尝试备用（不翻译，直接显示错误提示）
    return "[Translation Error]"

# --- 核心逻辑 ---
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
            
            with st.status("AI 正在流水线工作中...", expanded=True) as status:
                st.write("📂 提取音频...")
                video = VideoFileClip(st.session_state.video_path)
                audio_path = "temp_audio.wav"
                video.audio.write_audiofile(audio_path, verbose=False, logger=None)
                
                st.write("🧠 识别粤语...")
                # 提示 Whisper 它是中文
                result = model.transcribe(audio_path, language='Chinese')
                
                st.write("📝 生成粤拼与翻译...")
                data = []
                for seg in result['segments']:
                    txt = seg['text']
                    
                    # 1. 粤拼
                    jp_list = get_jyutping_list(txt)
                    jp_str = " ".join([i[1] if i[1] else i[0] for i in jp_list])
                    
                    # 2. 翻译
                    eng = safe_translate(txt)
                        
                    data.append({
                        "start": round(seg['start'], 2),
                        "end": round(seg['end'], 2),
                        "text": txt,
                        "jyutping": jp_str,
                        "english": eng
                    })
                
                st.session_state.subtitles_df = pd.DataFrame(data)
                
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    
                status.update(label="✅ 处理完成！", state="complete", expanded=False)

# --- 校对与导出 ---
if st.session_state.subtitles_df is not None:
    st.divider()
    st.header("2. 字幕校对")
    st.info("💡 提示：双击「英文翻译」列可直接修改内容。")
    
    edited_df = st.data_editor(
        st.session_state.subtitles_df,
        num_rows="dynamic",
        column_config={
            "start": "开始(s)",
            "end": "结束(s)",
            "text": "粤语汉字",
            "jyutping": "粤拼",
            "english": "英文翻译"
        },
        use_container_width=True
    )
    
    if st.button("💾 保存修改"):
        st.session_state.subtitles_df = edited_df
        st.success("已保存！")

    st.divider()
    st.header("3. 视频合成")
    
    if st.button("🎬 生成视频 (3:4 竖屏)"):
        font_path = load_fonts()
        v_path = st.session_state.video_path
        subs = st.session_state.subtitles_df.to_dict('records')
        
        progress = st.progress(0)
        status = st.empty()
        
        try:
            status.text("正在初始化...")
            W, H = 720, 960
            
            # 视频层
            clip = VideoFileClip(v_path)
            clip = clip.resize(width=W)
            
            target_h = 500
            if clip.h > target_h:
                clip = clip.crop(y1=(clip.h - target_h)/2, height=target_h)
            
            def make_frame(t):
                # 透明背景
                img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                
                cur = next((s for s in subs if s['start'] <= t <= s['end']), None)
                nxt = next((s for s in subs if s['start'] > t), None)
                
                try:
                    f_cn = ImageFont.truetype(font_path, 50)
                    f_jp = ImageFont.truetype(font_path, 32)
                    f_en = ImageFont.truetype(font_path, 26)
                except:
                    f_cn = ImageFont.load_default()
                    f_jp = ImageFont.load_default()
                    f_en = ImageFont.load_default()
                
                y_start = target_h + 40
                
                if cur:
                    # 汉字
                    w1 = draw.textlength(cur['text'], font=f_cn)
                    draw.text(((W-w1)/2, y_start), cur['text'], font=f_cn, fill="#FFD700")
                    # 粤拼
                    w2 = draw.textlength(cur['jyutping'], font=f_jp)
                    draw.text(((W-w2)/2, y_start + 80), cur['jyutping'], font=f_jp, fill="#87CEEB")
                    # 英文
                    w3 = draw.textlength(str(cur['english']), font=f_en)
                    draw.text(((W-w3)/2, y_start + 130), str(cur['english']), font=f_en, fill="#FFFFFF")

                if nxt:
                    draw.text((50, y_start + 220), f"Next: {nxt['text']}", font=f_jp, fill="#555555")
                    
                return np.array(img)

            status.text("正在渲染 (请耐心等待，约2-3分钟)...")
            sub_clip = VideoClip(make_frame, duration=clip.duration)
            
            final = CompositeVideoClip([
                ColorClip((W, H), color=(20, 20, 20), duration=clip.duration),
                clip.set_position(('center', 'top')),
                sub_clip.set_position('center')
            ])
            
            out_file = "cantonese_final.mp4"
            final.write_videofile(out_file, fps=24, codec='libx264', audio_codec='aac', logger=None)
            
            status.success("完成！")
            progress.progress(100)
            
            with open(out_file, "rb") as f:
                st.download_button("⬇️ 下载视频", f, file_name="cantonese_tutor.mp4")
            
            st.video(out_file)
            
        except Exception as e:
            st.error(f"合成出错: {e}")