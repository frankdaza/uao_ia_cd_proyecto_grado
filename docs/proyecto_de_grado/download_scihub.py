import os
import re
import urllib.request
import ssl
import time

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

bib_path = '/Users/knarf-studio/Projects/UAO/uao_ia_cd_proyecto_grado/docs/proyecto_de_grado/Referencias.bib'
out_dir = '/Users/knarf-studio/Projects/UAO/uao_ia_cd_proyecto_grado/docs/proyecto_de_grado/papers_evidencia'

dois = {}
with open(bib_path, 'r', encoding='utf-8') as f:
    content = f.read()

entries = content.split('@')
for entry in entries[1:]:
    lines = entry.split('\n')
    header = lines[0]
    entry_id = header.split('{')[1].split(',')[0].strip()
    
    doi_match = re.search(r'doi\s*=\s*{(.*?)}', entry)
    url_match = re.search(r'url\s*=\s*{(.*?)}', entry)
    dois[entry_id] = {
        'doi': doi_match.group(1) if doi_match else None,
        'url': url_match.group(1) if url_match else None
    }

txt_files = [f for f in os.listdir(out_dir) if f.endswith('.txt')]
if not txt_files:
    print("No missing papers (no .txt files) found.")
    
for txt in txt_files:
    entry_id = txt.replace('.txt', '')
    data = dois.get(entry_id, {})
    identifier = data.get('doi')
    
    if not identifier:
        # Some don't have DOIs, we can try searching by URL or Title, but sci-hub is best with DOI.
        print(f"[{entry_id}] No DOI found to search on Sci-Hub.")
        continue
        
    print(f"[{entry_id}] Searching Sci-Hub for {identifier}...")
    
    mirror = "https://sci-hub.ru/"
    search_url = f"{mirror}{identifier}"
    try:
        req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'})
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            
            # Look for the new meta tag
            meta_match = re.search(r'<meta\s+name="citation_pdf_url"\s+content="([^"]+)"', html)
            
            if meta_match:
                pdf_url = meta_match.group(1)
                if pdf_url.startswith('//'):
                    pdf_url = "https:" + pdf_url
                elif pdf_url.startswith('/'):
                    pdf_url = mirror.rstrip('/') + pdf_url
                
                print(f" -> Found PDF URL via citation_pdf_url: {pdf_url}")
                
                # Fetch PDF
                pdf_req = urllib.request.Request(pdf_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(pdf_req, context=ctx, timeout=30) as pdf_resp:
                    if pdf_resp.getcode() == 200:
                        pdf_path = os.path.join(out_dir, f"{entry_id}.pdf")
                        with open(pdf_path, 'wb') as pdf_file:
                            pdf_file.write(pdf_resp.read())
                        print(f" -> Successfully downloaded to {pdf_path}")
                        
                        os.remove(os.path.join(out_dir, txt))
            else:
                # Fallback to embed/iframe
                iframe_match = re.search(r'<embed[^>]+id="pdf"[^>]+src="([^"]+)"', html)
                if not iframe_match:
                    iframe_match = re.search(r'<iframe[^>]+id="pdf"[^>]+src="([^"]+)"', html)
                if iframe_match:
                    pdf_url = iframe_match.group(1)
                    if pdf_url.startswith('//'):
                        pdf_url = "https:" + pdf_url
                    print(f" -> Found PDF URL via embed/iframe: {pdf_url}")
                    # Fetch
                    pdf_req = urllib.request.Request(pdf_url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(pdf_req, context=ctx, timeout=30) as pdf_resp:
                        if pdf_resp.getcode() == 200:
                            pdf_path = os.path.join(out_dir, f"{entry_id}.pdf")
                            with open(pdf_path, 'wb') as pdf_file:
                                pdf_file.write(pdf_resp.read())
                            print(f" -> Successfully downloaded to {pdf_path}")
                            os.remove(os.path.join(out_dir, txt))
                else:
                    print(f" -> PDF link not found in HTML. Check if there is a CAPTCHA.")
    except Exception as e:
        print(f" -> Failed extraction: {e}")
        
    time.sleep(2)

print("Finished processing Sci-Hub downloads.")
