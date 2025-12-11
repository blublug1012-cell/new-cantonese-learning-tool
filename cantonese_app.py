import streamlit as st
import whisper
import os
import tempfile
import requests
import pandas as pd
import numpy as np
import time
from deep_translator import GoogleTranslator

# --- 核心修改：适配 MoviePy 2.0+ 的新写法 ---
# 1. 不再从 moviepy.editor 导入，而是直接从 moviepy 导入
from moviepy import VideoFileClip, CompositeVideoClip, ColorClip, VideoClip
# 2. 导入 PIL 库
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
st.title("🎬 粤语视频工坊 Pro (2025 新版适配)")

# --- 辅助函数 ---
@st.cache_resource
def load_model():
    return whisper.load_model("base")

def get_jyutping_list(text):
    # 延迟导入，防止库冲突
    from ToJyutping import get_jyutping_list
    return get_jyutping_list(text)

def safe_translate(text):
    try:
        time.sleep(0.2)
        # 强制指定繁体中文->英文
        res = GoogleTranslator(source='zh-TW', target='en').translate(text)
        if res and res != text:
            return res
    except:
        pass
    return "[Translation Error]"

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
            
            with st.status("AI 正在流水线工作中...", expanded=True) as status:
                st.write("📂 提取音频...")
                # MoviePy 2.0 写法: 直接调用，大部分兼容
                video = VideoFileClip(st.session_state.video_path)
                audio_path = "temp_audio.wav"
                
                # 兼容性处理：不同版本 write_audiofile 参数略有不同，但通常兼容
                try:
                    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
                except:
                    # 如果参数报错，尝试最简调用
                    video.audio.write_audiofile(audio_path)
                
                st.write("🧠 识别粤语...")
                result = model.transcribe(audio_path, language='Chinese')
                
                st.write("📝 生成数据...")
                data = []
                for seg in result['segments']:
                    txt = seg['text']
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
                
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    
                status.update(label="✅ 处理完成！", state="complete", expanded=False)

# --- 校对与导出 ---
if st.session_state.subtitles_df is not None:
    st.divider()
    st.header("2. 字幕校对")
    edited_df = st.data_editor(st.session_state.subtitles_df, num_rows="dynamic", use_container_width=True)
    
    if st.button("💾 保存修改"):
        st.session_state.subtitles_df = edited_df
        st.success("已保存！")

    st.divider()
    st.header("3. 视频合成")
    
    if st.button("🎬 生成视频"):
        font_path = load_fonts()
        v_path = st.session_state.video_path
        subs = st.session_state.subtitles_df.to_dict('records')
        
        progress = st.progress(0)
        status = st.empty()
        
        try:
            status.text("正在初始化...")
            W, H = 720, 960
            
            # --- 核心修改：MoviePy 2.0 的 resize 写法 ---
            clip = VideoFileClip(v_path)
            
            # 尝试使用新版 API resized()，如果失败回退到 resize()
            try:
                # MoviePy 2.0+ 推荐写法
                clip = clip.resized(width=W)
            except AttributeError:
                # 旧版或过渡版写法
                clip = clip.resize(width=W)
            
            # 裁剪高度
            target_h = 500
            if clip.h > target_h:
                clip = clip.crop(y1=(clip.h - target_h)/2, height=target_h)
            
            def make_frame(t):
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
                    w1 = draw.textlength(cur['text'], font=f_cn)
                    draw.text(((W-w1)/2, y_start), cur['text'], font=f_cn, fill="#FFD700")
                    
                    w2 = draw.textlength(cur['jyutping'], font=f_jp)
                    draw.text(((W-w2)/2, y_start + 80), cur['jyutping'], font=f_jp, fill="#87CEEB")
                    
                    w3 = draw.textlength(str(cur['english']), font=f_en)
                    draw.text(((W-w3)/2, y_start + 130), str(cur['english']), font=f_en, fill="#FFFFFF")

                if nxt:
                    draw.text((50, y_start + 220), f"Next: {nxt['text']}", font=f_jp, fill="#555555")
                    
                return np.array(img)

            status.text("正在渲染 (约3分钟)...")
            sub_clip = VideoClip(make_frame, duration=clip.duration)
            
            # MoviePy 2.0 的 ColorClip 可能需要 color 作为元组
            bg_clip = ColorClip(size=(W, H), color=(20, 20, 20), duration=clip.duration)
            
            final = CompositeVideoClip([
                bg_clip,
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
            # 打印详细错误，方便排查
            import traceback
            st.text(traceback.format_exc())
