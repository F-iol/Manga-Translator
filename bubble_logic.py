import tkinter as tk
from pynput import keyboard
from PIL import ImageGrab, Image, ImageDraw, ImageFont
import threading
import time
import os
import cv2
import numpy as np
import io 
import warnings

warnings.filterwarnings('ignore')

#globals
DELAY_SECONDS = 0.1   
CUSTOM_FONT_PATH = './fonts/PermanentMarker-Regular.ttf'



def keys_to_pynput_set(key_strings: list[str]) -> set:
    pynput_set = set()
    for key_str in key_strings:
        key_upper = key_str.upper()
        if key_upper == "CONTROL": pynput_set.add(keyboard.Key.ctrl)
        elif key_upper == "SHIFT": pynput_set.add(keyboard.Key.shift)
        elif key_upper == "ALT": pynput_set.add(keyboard.Key.alt)
        elif key_upper == "ESCAPE": pynput_set.add(keyboard.Key.esc)
        else:
            try:
                pynput_set.add(keyboard.KeyCode.from_char(key_str.lower()))
            except ValueError:
                if hasattr(keyboard.Key, key_str.lower()):
                    pynput_set.add(getattr(keyboard.Key, key_str.lower()))
    return pynput_set

class SnippingTool(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.withdraw()
        self.attributes('-fullscreen', True)
        self.attributes('-alpha', 0.3)
        self.attributes('-topmost', True)
        self.config(bg='gray')
        
        self.canvas = tk.Canvas(self, cursor="cross", bg="gray", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.selection = None
        
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.bind("<Escape>", self.cancel)
        self.bind("<Button-3>", self.cancel)

    def start(self):
        self.selection = None
        self.deiconify()
        self.wait_window() 
        return self.selection

    def on_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, 
            outline='red', width=2
        )

    def on_drag(self, event):
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_release(self, event):
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        
        x1, y1 = min(self.start_x, end_x), min(self.start_y, end_y)
        x2, y2 = max(self.start_x, end_x), max(self.start_y, end_y)
        
        if (x2 - x1) > 10 and (y2 - y1) > 10:
            self.selection = (int(x1), int(y1), int(x2), int(y2))
            self.destroy()
        else:
            self.cancel()

    def cancel(self, event=None):
        self.selection = None
        self.destroy()

class TranslationEngine:
    def __init__(self, crop_coords, delay_seconds, manager):
        self.crop_coords = crop_coords
        self.delay_seconds = delay_seconds
        self.manager = manager 
        
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self._is_running = True 
        
        self.models = manager.MODELS 
        if not self.manager.MODELS_LOADED:
            manager.output_callback("ERROR: Translation models failed to load.")

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop_event.set()
        self._is_running = False
        self.manager.output_callback("Continuous translation stopped.")

    def is_running(self):
        return self._is_running
        
    def _run_loop(self):
        while not self._stop_event.is_set():
            if self._stop_event.wait(self.delay_seconds): break

            try:
                capture = ImageGrab.grab(bbox=self.crop_coords)
            except Exception as e:
                self.manager.output_callback(f"Capture failed: {e}")
                time.sleep(1)
                continue

            final_pil = self._process_image(capture)

            if not self._stop_event.is_set():
                image_bytes = self.manager._pil_image_to_bytes(final_pil)
                if self.manager.image_callback:
                    self.manager.image_callback(image_bytes)
                
        self._is_running = False 
        
    def _process_image(self, capture_pil):
        if not self.manager.MODELS_LOADED: return capture_pil

        # Prediction / look for bubbles
        img_np = np.array(capture_pil)
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        original_cv2 = img_cv2.copy()
        
        try:
            results = self.models['bubble'].predict(source=img_cv2, conf=0.4, verbose=False)
        except Exception as e:
            self.manager.output_callback(f"YOLO prediction failed: {e}")
            return capture_pil

        if not results or not results[0].masks:
            return capture_pil 
            
        r = results[0]
        h, w = img_cv2.shape[:2]
        
        # Build Mask and translate data
        full_mask = np.zeros((h, w), dtype=np.uint8)
        overlays = [] # Store text and coords to draw later
        
        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x1 >= x2 or y1 >= y2: continue
            
            raw_mask = r.masks.data[i].cpu().numpy()
            resized_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            full_mask = cv2.bitwise_or(full_mask, (resized_mask > 0.5).astype(np.uint8) * 255)

            crop = original_cv2[y1:y2, x1:x2]
            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            
            ocr_text = self.models['ocr'](pil_crop)
            translated_text = self._translate_text(ocr_text)
            
            overlays.append((translated_text, x1, y1, x2-x1, y2-y1))

        # cover original text
        mean_val = cv2.mean(original_cv2, mask=full_mask)
        color = (int(mean_val[0]), int(mean_val[1]), int(mean_val[2]))
        img_cv2[full_mask > 0] = color
        
        # draw text
        pil_draw_img = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
        
        for text, x, y, w_box, h_box in overlays:
            pil_draw_img = self._draw_text(pil_draw_img, text, x, y, w_box, h_box)

        return pil_draw_img

    def _translate_text(self, text):
        try:
            if not text.strip(): return ""
            tokenizer = self.models['tokenizer']
            model = self.models['translator']
            device = self.models['device']

            
            inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
            translated = model.generate(**inputs, num_beams=5, no_repeat_ngram_size=2, length_penalty=2.0, max_length=150, early_stopping=True)
            return tokenizer.decode(translated[0], skip_special_tokens=True)
        except Exception:
            return "[TRANSLATION ERROR]"

    def _draw_text(self, img, text, x, y, w, h):
        draw = ImageDraw.Draw(img)
        
        # some padding to stay away from bubbles
        padding = 3
        safe_w = max(10, w - (padding * 2))
        safe_h = max(10, h - (padding * 2))

        # brute forcing to get a nice font size
        font_size = 14 # change here but that should do the trick ONLY EVEN NUMBERS
        min_font_size = 8
        final_font = None
        final_lines = []
        
        while font_size >= min_font_size:
            try:
                font = ImageFont.truetype(CUSTOM_FONT_PATH, size=font_size)
            except IOError:
                font = ImageFont.load_default()
                final_font = font
                final_lines = self._wrap_text(text, font, safe_w, draw)
                break

            lines = self._wrap_text(text, font, safe_w, draw)
            
            bbox = draw.textbbox((0, 0), "Hg", font=font)
            line_height = bbox[3] - bbox[1]
            total_text_height = line_height * len(lines)
            
            # Check vertically
            if total_text_height <= safe_h:
                final_font = font
                final_lines = lines
                break
            
            font_size -= 2 # Shrink and try again

        if final_font is None:
            try: final_font = ImageFont.truetype(CUSTOM_FONT_PATH, size=min_font_size)
            except: final_font = ImageFont.load_default()
            final_lines = self._wrap_text(text, final_font, safe_w, draw)

        text_block = "\n".join(final_lines)
        bbox = draw.multiline_textbbox((0, 0), text_block, font=final_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        text_x = x + (w - text_w) // 2
        text_y = y + (h - text_h) // 2

        #background for letters
        draw.multiline_text(
            (text_x, text_y), 
            text_block, 
            font=final_font, 
            fill="black", 
            align="center",
            stroke_width=2, 
            stroke_fill="white"
        )
        
        return img

    def _wrap_text(self, text, font, max_width, draw):
        """Helper to wrap text into lines based on pixel width."""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = " ".join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            w = bbox[2] - bbox[0]
            
            if w <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                
        if current_line:
            lines.append(" ".join(current_line))
        
        if not lines and words:
            return words 
            
        return lines

class BubbleTranslatorManager:
    def __init__(self):
        self.root_tk = tk.Tk()
        self.root_tk.withdraw()
        
        self.active_engine = None
        self.current_keys = set()
        self.listener = None
        
        self._start_combo_list = ["Shift", "E"]
        self._stop_combo_list = ["Shift", "S"]
        self._stop_v2_combo_list = ["Escape"]

        self.output_callback = lambda text: None 
        self.image_callback = lambda data: None
        self.hotkey_callback = lambda: None 

        self.MODELS = {}
        self.MODELS_LOADED = False
    
    def get_start_combination(self) -> list[str]: return self._start_combo_list
    def set_start_combination(self, key_strings: list[str]): self._start_combo_list = key_strings
    def get_stop_combination(self) -> list[str]: return self._stop_combo_list
    def set_stop_combination(self, key_strings: list[str]): self._stop_combo_list = key_strings
    def get_stop_v2_combination(self) -> list[str]: return self._stop_v2_combo_list
    def set_stop_v2_combination(self, key_strings: list[str]): self._stop_v2_combo_list = key_strings

    def set_models(self,model_dict):
        self.MODELS = model_dict

        if 'ocr' in self.MODELS and 'translator' in self.MODELS:
            self.MODELS_LOADED=True
            
        else:
            self.MODELS_LOADED=False
    

    def set_gui_callbacks(self, output_callback, image_callback, hotkey_callback):
        self.output_callback = output_callback
        self.image_callback = image_callback
        self.hotkey_callback = hotkey_callback 

    def _pil_image_to_bytes(self, pil_image: Image.Image) -> bytes:
        if pil_image is None: return b''
        try:
            byte_arr = io.BytesIO()
            pil_image.save(byte_arr, format='PNG') 
            return byte_arr.getvalue()
        except Exception:
            return b''

    def start_hotkey_listener(self):
        if self.listener is None:
            self.listener = keyboard.Listener(on_press=self._on_key_press, on_release=self._on_key_release)
            self.listener.start()

    def stop_listeners(self):
        self.stop_active_engine()
        if self.listener:
            self.listener.stop()
        try:
            self.root_tk.quit()
        except Exception:
            pass 
        
    def stop_active_engine(self):
        if self.active_engine and self.active_engine.is_running():
            self.active_engine.stop()
            self.active_engine = None

    def start_continuous_translation(self):
        self.stop_active_engine()
        
        snipper = SnippingTool(self.root_tk) 
        coords = snipper.start() 
        
        if coords:
            self.active_engine = TranslationEngine(coords, DELAY_SECONDS, self)
            self.active_engine.start()
        else:
            self.output_callback("Selection cancelled.")

    def _on_key_press(self, key):
        k = key
        if hasattr(key, 'char') and key.char:
            k = keyboard.KeyCode.from_char(key.char.lower())
        
        self.current_keys.add(k)
        
        start_set = keys_to_pynput_set(self._start_combo_list)
        stop_set = keys_to_pynput_set(self._stop_combo_list)
        stop_set_v2 = keys_to_pynput_set(self._stop_v2_combo_list)

        if stop_set_v2 and stop_set_v2.issubset(self.current_keys):
            self.output_callback(f"Stopping translation")
            self.stop_active_engine()
            self.current_keys.clear()
            return
        
        if start_set and start_set.issubset(self.current_keys):
            self.hotkey_callback() 
            self.current_keys.clear()
            return
            
        if stop_set and stop_set.issubset(self.current_keys):
            self.output_callback(f"Stopping translation")
            self.stop_active_engine() 
            self.current_keys.clear()
            return

    def _on_key_release(self, key):
        k = key
        if hasattr(key, 'char') and key.char:
            k = keyboard.KeyCode.from_char(key.char.lower())
        if k in self.current_keys:
            self.current_keys.remove(k)

if __name__ == "__main__":
    manager = BubbleTranslatorManager()
    manager.start_hotkey_listener()
    
    try:
        manager.root_tk.mainloop() 
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_listeners()