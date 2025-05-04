import pytesseract
from PIL import ImageGrab, Image
import pyautogui
import cv2
import numpy as np
import time

# if you didnt get Tesseract OCR in the PATH, remove hashtags if you did, not required.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def get_clipboard_image():
    try:
        img = ImageGrab.grabclipboard()
        if isinstance(img, Image.Image):
            return img
        else:
            print("No image found in clipboard.")
            return None
    except Exception as e:
        print(f"Error grabbing image: {e}")
        return None

def extract_text_and_confidence(image):
    # convert to OpenCV gray
    open_cv = np.array(image)
    gray = cv2.cvtColor(open_cv, cv2.COLOR_RGB2GRAY)
    # get OCR data including confidences
    data = pytesseract.image_to_data(gray, lang='rus+eng', output_type=pytesseract.Output.DICT)
    # join all text elements
    text = " ".join([t for t in data['text'] if t.strip() != ""]).strip()
    # collect numeric confidences ≥ 0
    confs = []
    for c in data['conf']:
        try:
            cf = float(c)
            if cf >= 0:
                confs.append(cf)
        except Exception:
            pass
    avg_conf = sum(confs) / len(confs) if confs else 0
    return text, avg_conf

def image_entropy(gray):
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_norm = hist.ravel()/hist.sum()
    entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))
    return entropy

def image_to_ascii(image, width=80):
    img = image.convert('L')
    w, h = img.size
    ratio = h / w
    new_h = int(ratio * width * 0.55)
    img = img.resize((width, new_h))
    pixels = list(img.getdata())
    chars = "@#S%?*+;:,. "
    ascii_str = "".join(chars[p//25] for p in pixels)
    return "\n".join(ascii_str[i:i+width] for i in range(0, len(ascii_str), width))

def main():
    input("to get started, copy an image or screenshot it and press enter")
    img = get_clipboard_image()
    if not img:
        return

    open_cv = np.array(img)
    gray = cv2.cvtColor(open_cv, cv2.COLOR_RGB2GRAY)

    text, avg_conf = extract_text_and_confidence(img)
    ent = image_entropy(gray)
    print(f"📊 Entropy: {ent:.2f}, OCR Confidence: {avg_conf:.2f}")

    if avg_conf > 40 and len(text) > 5:
        print("detected text:")
        print(text)
        output = text
    else:
        print("Image without text detected! Turning into ASCII:")
        art = image_to_ascii(img)
        print(art)
        output = art

    input("Please get your app ready and press enter...")
    time.sleep(5)
    print("Typing soon...")
    pyautogui.write(output, interval=0.005)

if __name__ == "__main__":
    main()
