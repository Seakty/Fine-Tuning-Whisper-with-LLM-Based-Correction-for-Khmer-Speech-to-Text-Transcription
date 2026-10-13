"""Manual test against the real Gemini API (needs GEMINI_API_KEY set):
    python -m src.correction "your raw ASR text here"
    python -m src.correction --samples     # run the known-messy examples from docs/results.md
"""
import sys

from . import GeminiCorrectionError, correct_text

# Real messy outputs pulled from docs/results.md, for a quick eyeball check of quality.
SAMPLE_MESSY_OUTPUTS = [
    "គណៈបរក្ខណ្ឌនម្នាយ អង្គការប្រជាជន៍អស្រ្វិខាងលិច ឈាមខាងត្បូង ស្វាប់ពរ ទទួលបានសំឡេងគាំទៅភាគ្រាអនៅ",
    "នេះចេន គឺជាប្រព័ន្ធមួយដែលអែហ្វ អាអែរ អាអាងថានឹងអាចឲ្យយន្តហោះ ហោះហើរបាន និលរយៈចម្ងាយចិត្ត និងសន",
    "ក្រោយមកទៀត គាត់ត្រូវបានប្លាទីទៅកាន់មន្ទីរពេទ្យ អេដិនប្រុក ដែលស្ហិតនៅក្នុងទីក្រុម ខែម្រេត្ត។",
]

if len(sys.argv) > 1 and sys.argv[1] == "--samples":
    texts = SAMPLE_MESSY_OUTPUTS
else:
    if len(sys.argv) < 2:
        sys.exit('usage: python -m src.correction "raw text" | python -m src.correction --samples')
    texts = [sys.argv[1]]

for raw in texts:
    print(f"\nRAW:       {raw}")
    try:
        print(f"CORRECTED: {correct_text(raw)}")
    except GeminiCorrectionError as e:
        print(f"FAILED: {e.user_message}\n  detail: {e.detail}")