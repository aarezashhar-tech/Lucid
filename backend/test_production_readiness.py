import requests
import io
import time
from PIL import Image, ImageDraw
import pypdf

time.sleep(1)

print('=== 1. Health Check & ML Model Loading ===')
r_health = requests.get('http://localhost:5000/api/health')
assert r_health.status_code == 200, f'Health failed: {r_health.status_code}'
health_json = r_health.json()
print('Status:', health_json.get('status'))
print('Model Loaded:', health_json.get('modelLoaded'))
print('Vectorizer Loaded:', health_json.get('vectorizerLoaded'))
assert health_json.get('modelLoaded') is True
assert health_json.get('vectorizerLoaded') is True

print('\n=== 2. CORS Verification (https://www.yourdomain.com) ===')
headers = {'Origin': 'https://www.yourdomain.com'}
r_cors = requests.options('http://localhost:5000/api/analyze', headers=headers)
print('OPTIONS status:', r_cors.status_code)
print('Access-Control-Allow-Origin:', r_cors.headers.get('Access-Control-Allow-Origin'))
assert r_cors.headers.get('Access-Control-Allow-Origin') == 'https://www.yourdomain.com'

r_post_cors = requests.post('http://localhost:5000/api/analyze', json={'text': 'Test claim for CORS'}, headers=headers)
print('POST Access-Control-Allow-Origin:', r_post_cors.headers.get('Access-Control-Allow-Origin'))
assert r_post_cors.headers.get('Access-Control-Allow-Origin') == 'https://www.yourdomain.com'

print('\n=== 3. RAG Retrieval & Stance Fact Verification ===')
claims = [
    ('5G causes COVID-19', 'FALSE'),
    ('Pope endorsed Trump', 'FALSE'),
    ('Earth is flat', 'FALSE')
]
for claim, expected_verdict in claims:
    res = requests.post('http://localhost:5000/api/analyze', json={'text': claim}).json()
    verdict = res.get('verdict')
    score = res.get('trustScore')
    contra = res.get('evidence', {}).get('contradictingCount', 0)
    supp = res.get('evidence', {}).get('supportingCount', 0)
    print(f'Claim: "{claim}" -> Verdict: {verdict} (Score: {score}, Contra: {contra}, Supp: {supp})')
    assert verdict == expected_verdict, f'Expected {expected_verdict}, got {verdict}'

print('\n=== 4. Image Upload & OCR Verification ===')
# Generate test image with text
img = Image.new('RGB', (450, 100), color=(255, 255, 255))
d = ImageDraw.Draw(img)
d.text((20, 40), 'Breaking news: WHO reports global health guidelines.', fill=(0, 0, 0))
img_bytes = io.BytesIO()
img.save(img_bytes, format='PNG')
img_bytes.seek(0)

files = {'file': ('headline.png', img_bytes, 'image/png')}
r_ocr = requests.post('http://localhost:5000/api/ocr', files=files)
print('OCR Status:', r_ocr.status_code)
assert r_ocr.status_code == 200
ocr_json = r_ocr.json()
print('Extracted Text:', repr(ocr_json.get('extractedText', '')))
assert 'WHO' in ocr_json.get('extractedText', '') or 'health' in ocr_json.get('extractedText', '').lower()

print('\n=== 5. PDF Upload Verification ===')
writer = pypdf.PdfWriter()
page = writer.add_blank_page(width=300, height=300)
pdf_bytes = io.BytesIO()
writer.write(pdf_bytes)
pdf_bytes.seek(0)

pdf_files = {'file': ('report.pdf', pdf_bytes, 'application/pdf')}
r_pdf = requests.post('http://localhost:5000/api/analyze-file', files=pdf_files)
print('PDF Upload Status:', r_pdf.status_code)
assert r_pdf.status_code == 200

print('\n>>> ALL PRODUCTION READINESS CHECKS PASSED SUCCESSFULLY! <<<')
