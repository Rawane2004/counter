from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import easyocr
import re
import unicodedata

app = FastAPI(title="Grocery Scanner OCR", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # في النشر الفعلي استبدل * بعنوان موقعك فقط
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل النموذج مرة واحدة عند تشغيل الخادم، وليس مع كل صورة
reader = easyocr.Reader(["ar", "en"], gpu=False)

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫٬", "01234567890123456789.,")
PRICE_HINTS = ["sar", "ريال", "ر.س", "sr", "price", "السعر", "عرض", "sale"]
IGNORE_HINTS = ["kg", "g ", " g", "ml", "l ", " l", "جرام", "غرام", "كجم", "كيلو", "مل", "لتر", "قطعة", "حبة", "barcode"]


def normalize_digits(value: str) -> str:
    return unicodedata.normalize("NFKC", value).translate(ARABIC_DIGITS)


def prepare_image(raw: bytes) -> np.ndarray:
    data = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("تعذر فتح الصورة. جرّب صورة بصيغة JPG أو PNG.")
    h, w = image.shape[:2]
    # تكبير الصور الصغيرة لتحسين قراءة الكتابة المطبوعة
    if max(h, w) < 1400:
        scale = 1400 / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # تحسين التباين مع المحافظة على الصورة الملونة قدر الإمكان
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return enhanced


def parse_price(lines):
    candidates = []
    for idx, item in enumerate(lines):
        text = normalize_digits(item["text"]).lower().replace("٫", ".")
        nearby = " ".join(x["text"] for x in lines[max(0, idx-1):min(len(lines), idx+2)]).lower()
        # لا نعتبر الأرقام الطويلة (مثل الباركود) أسعاراً
        for match in re.finditer(r"(?<!\d)(\d{1,4}(?:[.,]\d{1,2})?)(?!\d)", text):
            raw_number = match.group(1).replace(",", ".")
            try:
                value = float(raw_number)
            except ValueError:
                continue
            if value <= 0 or value > 100000:
                continue
            score = float(item["confidence"])
            if any(hint in nearby for hint in PRICE_HINTS):
                score += 0.25
            if any(hint in nearby for hint in IGNORE_HINTS):
                score -= 0.45
            # الأسعار ذات الكسور شائعة في بطاقات المتاجر
            if "." in raw_number:
                score += 0.08
            # استبعاد الأرقام الصغيرة جداً التي غالباً تكون أحجاماً أو أعداداً
            if value < 0.1:
                score -= 0.2
            candidates.append({"value": round(value, 2), "confidence": round(min(score, 1.0), 3), "text": match.group(1)})
    if not candidates:
        return None, 0.0, []
    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    best = candidates[0]
    return best["value"], best["confidence"], candidates[:5]


def clean_name(lines, price_candidates):
    chunks = []
    for item in lines:
        text = item["text"].strip()
        normalized = normalize_digits(text)
        # استبعاد الأسطر الرقمية أو التي يغلب عليها السعر
        if not re.search(r"[A-Za-z\u0600-\u06FF]", text):
            continue
        if any(h in normalized.lower() for h in ["www.", ".com", "barcode"]):
            continue
        chunks.append(text)
    # نحتفظ بالنص المقروء، ثم نزيل التكرار والمسافات الزائدة
    name = " ".join(chunks)
    name = re.sub(r"\s+", " ", name).strip(" -|:؛")
    return name[:120]


CATEGORY_KEYWORDS = {
    "المشروبات": ["ماء", "عصير", "حليب", "قهوة", "شاي", "water", "juice", "milk", "coffee", "tea", "drink"],
    "الألبان": ["لبن", "زبادي", "جبن", "قشطة", "yogurt", "cheese", "cream"],
    "اللحوم": ["دجاج", "لحم", "لحم بقري", "chicken", "beef", "meat", "lamb"],
    "الفواكه والخضار": ["تفاح", "موز", "برتقال", "طماطم", "بطاطس", "apple", "banana", "orange", "tomato", "potato"],
    "الحلويات": ["شوكولاتة", "حلوى", "بسكويت", "كيك", "chocolate", "candy", "biscuit", "cake"],
    "المخبوزات": ["خبز", "توست", "كرواسون", "bread", "toast", "croissant"],
    "الوجبات الخفيفة": ["شيبس", "مكسرات", "رقائق", "chips", "nuts", "snack", "popcorn"],
    "المنظفات": ["منظف", "صابون", "مسحوق", "detergent", "soap", "cleaner"],
}


def classify(name):
    lower = name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(word in lower for word in keywords):
            return category
    return "أخرى"


@app.get("/health")
def health():
    return {"ok": True, "ocr": "EasyOCR ar+en"}


@app.post("/scan")
async def scan(image: UploadFile = File(...)):
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="الملف المرسل ليس صورة.")
    raw = await image.read()
    if not raw or len(raw) > 12 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="الصورة فارغة أو أكبر من 12MB.")
    try:
        processed = prepare_image(raw)
        results = reader.readtext(processed, detail=1, paragraph=False, decoder="greedy")
        lines = []
        for box, text, confidence in results:
            if text.strip():
                lines.append({"text": text.strip(), "confidence": float(confidence), "y": float((box[0][1] + box[2][1]) / 2)})
        # ترتيب تقريبي من أعلى الصورة إلى أسفلها
        lines.sort(key=lambda x: x["y"])
        price, price_confidence, price_candidates = parse_price(lines)
        name = clean_name(lines, price_candidates)
        if not name:
            name = ""
        overall = round(sum(x["confidence"] for x in lines) / len(lines), 3) if lines else 0
        return JSONResponse({
            "success": True,
            "name": name,
            "price": price,
            "category": classify(name),
            "confidence": overall,
            "price_confidence": price_confidence,
            "needs_confirmation": True,
            "price_found": price is not None,
            "message": "راجع الاسم والسعر قبل الإضافة." if price is not None else "تمت قراءة النص، لكن لم يظهر سعر واضح. أدخل السعر يدوياً أو امسح بطاقة الرف.",
            "raw_text": "\n".join(x["text"] for x in lines),
            "price_candidates": price_candidates,
        })
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"حدث خطأ أثناء قراءة الصورة: {str(exc)}")
