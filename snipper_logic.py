import tkinter as tk
import numpy as np
import threading
import time
from pynput import keyboard
from PIL import ImageGrab, Image, ImageTk, ImageDraw, ImageFont

CUSTOM_FONT_PATH = './fonts/PermanentMarker-Regular.ttf'

def keys_to_pynput_set(key_strings: list[str]) -> set:
    pynput_set = set()
    for key_str in key_strings:
        key_upper = key_str.upper()
        if key_upper == "CONTROL": pynput_set.add(keyboard.Key.ctrl_l)
        elif key_upper == "SHIFT": pynput_set.add(keyboard.Key.shift_l)
        elif key_upper == "ALT": pynput_set.add(keyboard.Key.alt_l)
        elif key_upper == "ESCAPE": pynput_set.add(keyboard.Key.esc)
        else:
            try:
                pynput_set.add(keyboard.KeyCode.from_char(key_str.lower()))
            except Exception:
                pass
    return pynput_set

def translate_text(text: str, models: dict) -> str:
    if not text: return ""
    
    tokenizer = models.get('tokenizer')
    model = models.get('translator')
    device = models.get('device', 'cpu')

    if not tokenizer or not model:
        return text # Return original if models missing

    try:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        tokens = model.generate(**inputs, max_length=100, num_beams=6, early_stopping=True, do_sample=False)
        return tokenizer.decode(tokens[0], skip_special_tokens=True)
    except Exception:
        return f"[Error]"

def _read_text_from_image(img: Image, ocr_model) -> str:
    if not ocr_model:
        return "OCR Unavailable"
    try:
        return ocr_model(img).strip()
    except Exception as e:
        return f"OCR Failed"

#fill cropped out image with avg color so it doesnt really THAT out of place
def _get_mean_color_and_overlay_text(img: Image, text: str, font_path: str = CUSTOM_FONT_PATH) -> Image:
    np_img = np.array(img.convert('RGB')) 
    mean_color = tuple(map(int, np.mean(np_img, axis=(0, 1))))
    
    luminance = (0.299 * mean_color[0] + 0.587 * mean_color[1] + 0.114 * mean_color[2]) #black/white text depends on lumiancne
    text_color = "black" if luminance > 128 else "white"
    
    width, height = img.size
    new_img = Image.new('RGB', (width, height), color=mean_color)
    draw = ImageDraw.Draw(new_img)
    
    try:
        font_pil = ImageFont.truetype(font_path, size=14) 
    except IOError:
        font_pil = ImageFont.load_default()
    
    lines, current_line = [], ""
    max_chars = int(width / 10)
    
    for word in text.split():
        if len(current_line) + len(word) + 1 > max_chars and current_line:
            lines.append(current_line)
            current_line = word
        else:
            current_line = f"{current_line} {word}" if current_line else word
    if current_line: lines.append(current_line)
    
    wrapped = '\n'.join(lines)
    
    try:
        bbox = draw.textbbox((0, 0), wrapped, font=font_pil)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        text_w, text_h = draw.textsize(wrapped, font=font_pil)

    x = max(0, (width - text_w) / 2)
    y = max(0, (height - text_h) / 2)
    
    draw.text((x, y), wrapped, font=font_pil, fill=text_color)
    return new_img

class SnipingTool(tk.Toplevel):
    def __init__(self, master, manager):
        super().__init__(master)
        self.manager = manager # Access models via self.manager.models
        self.transparent_color = 'magenta3' 

        self.config(bg=self.transparent_color)
        self.attributes('-transparentcolor', self.transparent_color)
        self.attributes('-alpha', 0.3) 
        self.attributes('-fullscreen', True)
        self.attributes('-topmost', True)
        self.withdraw() 
        
        self.canvas = tk.Canvas(self, cursor='cross', bg="#A0A0A0", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas.bind('<ButtonPress-1>', self.on_press)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.bind('<Escape>', lambda x: self.destroy())
        self.bind('<ButtonPress-3>', lambda x: self.destroy())
        
        self.start_x = self.start_y = None
        self.rect = None
        self.cutout = None 
        
        self.deiconify() # Show window immediately

    def on_press(self, event):
        self.start_x, self.start_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        if self.rect: self.canvas.delete(self.rect)
        if self.cutout: self.canvas.delete(self.cutout)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=3, dash=(4, 2))
        self.cutout = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, fill=self.transparent_color, outline='')

    def on_mouse_move(self, event):
        if self.start_x is not None:
            cur_x, cur_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
            self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)
            self.canvas.coords(self.cutout, self.start_x, self.start_y, cur_x, cur_y)
            
    def on_release(self, event):
        if self.start_x is not None:
            end_x, end_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
            self.capture_and_close(end_x, end_y)
        self.destroy()

    def capture_and_close(self, end_x, end_y):
        x1, y1 = min(self.start_x, end_x), min(self.start_y, end_y)
        x2, y2 = max(self.start_x, end_x), max(self.start_y, end_y)
        
        # Check is screen is big enough
        if (x2 - x1) > 10 and (y2 - y1) > 10:
            self.withdraw()
            
            capture = ImageGrab.grab(bbox=(x1, y1, x2, y2))
            
            models = self.manager.models
            ocr_model = models.get('ocr')
            

            original_text = _read_text_from_image(capture, ocr_model)
            translated_text = translate_text(original_text, models)
            self.manager.log_translation_result(original_text, translated_text)

            if self.manager.get_display_image():
                display_img = _get_mean_color_and_overlay_text(capture, translated_text, CUSTOM_FONT_PATH)
                self.display_feedback_window((x1, y1, x2, y2), display_img)

    def display_feedback_window(self, coords, display_img: Image):
        x1, y1, x2, y2 = coords
        w, h = int(x2 - x1), int(y2 - y1)
        
        self.tk_image = ImageTk.PhotoImage(display_img) 
        
        win = tk.Toplevel(self.master)
        win.geometry(f'{w}x{h}+{int(x1)}+{int(y1)}')
        win.overrideredirect(True) 
        win.attributes('-topmost', True)
        
        tk.Label(win, image=self.tk_image).pack(fill=tk.BOTH, expand=True)
        
        tk.Button(win, text="X", command=win.destroy, bg='red', fg='white', relief='flat', font=('Arial', 6, 'bold')).place(relx=1.0, rely=0.0, anchor='ne') 
        
        win.after(5000, win.destroy)

class SnippingHotkeyManager:
    def __init__(self):
        self.models = {}
        self.current_keys = set()
        self.is_cropping_active = False
        
        self._display_original = True
        self._display_translated = True
        self._display_image = True
        self._gui_output_callback = lambda x: None
        
        self._combination = [keyboard.Key.shift_l, keyboard.KeyCode.from_char('q')] 
        
        self.root = None
        self.tk_thread = None
        self.listener = None
        
    def set_models(self, model_dict):
        """Receives dependencies from Main UI."""
        self.models = model_dict

    # Getters/Setters
    def get_display_original(self): return self._display_original
    def get_display_translated(self): return self._display_translated
    def get_display_image(self): return self._display_image
    
    def set_display_translated_from_qt(self, s): self.set_display_translated(s == 2)
    def set_display_original_from_qt(self, s): self.set_display_original(s == 2)
    def set_display_image_from_qt(self, s): self.set_display_image(s == 2)

    def set_gui_output_callback(self, cb): self._gui_output_callback = cb
    def set_display_original(self, v): self._display_original = v
    def set_display_translated(self, v): self._display_translated = v
    def set_display_image(self, v): self._display_image = v
        
    def log_translation_result(self, original_text: str, translated_text: str):
        parts = []
        if self._display_original: parts.append(f"Source: {original_text}")
        if self._display_translated: parts.append(f"EN: {translated_text}")
        if parts: self._gui_output_callback("\n".join(parts))

    @property
    def combination(self): return self._combination

    def set_combination(self, key_strings: list[str]):
        pynput_keys = []
        key_map = {"CONTROL": keyboard.Key.ctrl_l, "SHIFT": keyboard.Key.shift_l, "ALT": keyboard.Key.alt_l}
        for k in key_strings:
            u = k.upper()
            if u in key_map: pynput_keys.append(key_map[u])
            elif len(k) == 1 and k.isalpha(): pynput_keys.append(keyboard.KeyCode.from_char(k.lower()))
            else:
                 try: pynput_keys.append(getattr(keyboard.Key, k.lower()))
                 except: pass
        if len(pynput_keys) >= 2: self._combination = pynput_keys

    def _tk_setup(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.mainloop()

    def start_listeners(self):
        self.tk_thread = threading.Thread(target=self._tk_setup, daemon=True)
        self.tk_thread.start()
        time.sleep(0.1) 
        self.listener = keyboard.Listener(on_press=self._on_key_press, on_release=self._on_key_release)
        self.listener.start()

    def stop_listeners(self):
        if self.listener: self.listener.stop()
        if self.root: self.root.quit()
        if self.tk_thread: self.tk_thread.join(1)

    def _get_key_to_check(self, key):
        if hasattr(key, 'char') and key.char: return keyboard.KeyCode.from_char(key.char.lower())
        return key
        
    def _on_key_press(self, key):
        k = self._get_key_to_check(key)
        if k in self.combination: self.current_keys.add(k)
        if all(x in self.current_keys for x in self.combination):
            if not self.is_cropping_active: self._launch_snipping_tool()

    def _on_key_release(self, key):
        k = self._get_key_to_check(key)
        if k in self.current_keys: self.current_keys.discard(k)

    def _launch_snipping_tool(self):
        if self.root:
            self.is_cropping_active = True
            self.root.after(0, self._snipping_tool_launcher)
        else: self.is_cropping_active = False 

    def _snipping_tool_launcher(self):
        if self.is_cropping_active:
            snipper = SnipingTool(self.root, self)
            self.root.wait_window(snipper)
            self.is_cropping_active = False

_SNIPPER_MANAGER = None
def get_snipping_manager():
    global _SNIPPER_MANAGER
    if _SNIPPER_MANAGER is None:
        _SNIPPER_MANAGER = SnippingHotkeyManager()
        _SNIPPER_MANAGER.start_listeners()
    return _SNIPPER_MANAGER