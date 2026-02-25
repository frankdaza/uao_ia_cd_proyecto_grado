import re
import os
import urllib.request
import urllib.error
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

# Fix litjens2017survey
if 'litjens2017survey' in content and 'doi = {10.1016/j.media.2017.07.005}' not in content:
    content = re.sub(
        r'(@article{litjens2017survey,[\s\S]*?)(\n})', 
        r'\1,\n  doi       = {10.1016/j.media.2017.07.005},\n  url       = {https://doi.org/10.1016/j.media.2017.07.005}\2', 
        content
    )

# Fix willemink2020preparing
if 'willemink2020preparing' in content and 'doi = {10.1148/radiol.2020192224}' not in content:
    content = re.sub(
        r'(@article{willemink2020preparing,[\s\S]*?)(\n})', 
        r'\1,\n  doi       = {10.1148/radiol.2020192224},\n  url       = {https://pubs.rsna.org/doi/10.1148/radiol.2020192224}\2', 
        content
    )

# Fix esteva2019guide
if 'esteva2019guide' in content and 'doi = {10.1038/s41591-018-0316-z}' not in content:
    content = re.sub(
        r'(@article{esteva2019guide,[\s\S]*?)(\n})', 
        r'\1,\n  doi       = {10.1038/s41591-018-0316-z},\n  url       = {https://www.nature.com/articles/s41591-018-0316-z}\2', 
        content
    )

# Fix haddou2025hqcm
if 'haddou2025hqcm' in content and 'url =' not in content:
    content = re.sub(
        r'(@article{haddou2025hqcm,[\s\S]*?)(\n})', 
        r'\1,\n  url     = {https://arxiv.org/abs/2506.21937}\2', 
        content
    )

with open(bib_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("BibTeX updated successfully.")

entries = content.split('@')
results = []
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
            pdf_url = None # can't download datasets like this easily
        elif doi:
            # try unpaywall API
            try:
                upw_url = f"https://api.unpaywall.org/v2/{doi}?email=unpaywall@example.com"
                req = urllib.request.Request(upw_url)
                with urllib.request.urlopen(req, context=ctx) as resp:
                    data = json.loads(resp.read())
                    if data.get('best_oa_location') and data['best_oa_location'].get('url_for_pdf'):
                        pdf_url = data['best_oa_location']['url_for_pdf']
            except Exception as e:
                pass
    
    print(f"[{entry_id}] URL: {url} | PDF: {pdf_url}")
    
    if pdf_url:
        pdf_path = os.path.join(out_dir, f"{entry_id}.pdf")
        if not os.path.exists(pdf_path):
            try:
                req = urllib.request.Request(pdf_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
                with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
                    with open(pdf_path, 'wb') as out_f:
                        out_f.write(resp.read())
                print(f" -> Downloaded to {pdf_path}")
            except Exception as e:
                print(f" -> Failed to download: {e}")
                # Create a placeholder error file
                with open(pdf_path.replace('.pdf', '.txt'), 'w') as out_f:
                    out_f.write(f"Could not download PDF automatically.\nSource URL: {url}\nError: {str(e)}")
        else:
            print(f" -> Already downloaded")
    else:
        # Save a text placeholder
        txt_path = os.path.join(out_dir, f"{entry_id}.txt")
        if not os.path.exists(txt_path):
            with open(txt_path, 'w') as out_f:
                out_f.write(f"No direct PDF link found.\nSource URL: {url}\nDOI: {doi}")
            print(f" -> No PDF link, created text placeholder.")
            
    time.sleep(1) # Be polite to servers

