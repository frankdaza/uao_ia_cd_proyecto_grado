import os
import re
import urllib.request
import ssl
import json
import time

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

bib_path = '/Users/knarf-studio/Projects/UAO/uao_ia_cd_proyecto_grado/docs/proyecto_de_grado/Referencias.bib'
out_dir = '/Users/knarf-studio/Projects/UAO/uao_ia_cd_proyecto_grado/docs/proyecto_de_grado/papers_evidencia'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

with open(bib_path, 'r', encoding='utf-8') as f:
    content = f.read()

entries = content.split('@')
for entry in entries[1:]:
    lines = entry.split('\n')
    header = lines[0]
    entry_id = header.split('{')[1].split(',')[0].strip()
    
    url_match = re.search(r'url\s*=\s*{(.*?)}', entry)
    doi_match = re.search(r'doi\s*=\s*{(.*?)}', entry)
    
    url = url_match.group(1) if url_match else None
    doi = doi_match.group(1) if doi_match else None
    
    if not url and doi:
        url = f"https://doi.org/{doi}"
        
    pdf_url = None
    if url:
        if 'arxiv.org/abs/' in url:
            pdf_url = url.replace('/abs/', '/pdf/') + '.pdf'
        elif 'mdpi.com' in url:
            pdf_url = url + '/pdf'
        elif 'nature.com/articles/' in url:
            pdf_url = url + '.pdf'
        elif 'quantum-journal.org/papers/' in url:
            pdf_url = url + 'pdf/'
        elif 'kaggle.com' in url:
            pdf_url = None
        elif doi:
            try:
                upw_url = f"https://api.unpaywall.org/v2/{doi}?email=unpaywall@example.com"
                req = urllib.request.Request(upw_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, context=ctx, timeout=5) as resp:
                    data = json.loads(resp.read())
                    if data.get('best_oa_location') and data['best_oa_location'].get('url_for_pdf'):
                        pdf_url = data['best_oa_location']['url_for_pdf']
            except Exception:
                pass
                
    # Fallback open access heuristic if unpaywall fails
    if not pdf_url and url and ('aps.org' in url or 'ieee.org' in url or 'ijcsm.researchcommons.org' in url):
        # We'll just try to hit typical open access links or store the URL
        if 'ijcsm.researchcommons.org' in url:
            pdf_url = url.replace('/iss2/9/', '/iss2/9/fulltext.pdf') # Guessing URL structure

    print(f"[{entry_id}] Extracted URL: {url} | PDF URL: {pdf_url}")

    pdf_path = os.path.join(out_dir, f"{entry_id}.pdf")
    txt_path = os.path.join(out_dir, f"{entry_id}.txt")
    
    if os.path.exists(pdf_path):
        print(f" -> Already downloaded {pdf_path}")
        continue

    if pdf_url:
        try:
            req = urllib.request.Request(pdf_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
            with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
                if resp.getcode() == 200:
                    with open(pdf_path, 'wb') as out_f:
                        out_f.write(resp.read())
                    print(f" -> Downloaded PDF")
                    if os.path.exists(txt_path):
                        os.remove(txt_path)
                else:
                    raise Exception(f"HTTP {resp.getcode()}")
        except Exception as e:
            print(f" -> Failed to download PDF: {e}")
            with open(txt_path, 'w') as out_f:
                out_f.write(f"Target URL: {url}\nPDF attempted: {pdf_url}\nError: {str(e)}\n\nCould not automatically download PDF due to Paywall, Captcha, or Error.")
    else:
        print(f" -> No PDF available for automatic download.")
        with open(txt_path, 'w') as out_f:
            out_f.write(f"Target URL: {url}\nNo open PDF URL identified automatically. Please check manually.")

    time.sleep(1)

print("Finished processing.")
