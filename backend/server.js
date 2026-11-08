// backend/server.js
const express = require("express");
const cors = require("cors");
const multer = require("multer");
const Tesseract = require("tesseract.js");

const app = express();
const upload = multer({ dest: "uploads/" });

app.use(cors());
app.use(express.json());

app.post("/scan", upload.single("image"), async (req, res) => {
  try {
    const result = await Tesseract.recognize(
      req.file.path,
      "eng",
      { logger: () => {} }
    );

    const text = result.data.text;

    // استخراج سعر بسيط
    const priceMatch = text.match(/\$?\d+(\.\d{2})?/);
    const price = priceMatch ? priceMatch[0] : "N/A";

    res.json({
      success: true,
      price,
      rawText: text
    });
  } catch (err) {
    res.status(500).json({ success: false });
  }
});

app.listen(3000, () => {
  console.log("Backend running on http://localhost:3000");
});