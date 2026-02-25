import re
import urllib.request
import urllib.error
import ssl
import json

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

with open('/Users/knarf-studio/Projects/UAO/uao_ia_cd_proyecto_grado/docs/proyecto_de_grado/Referencias.bib', 'r') as f:
    content = f.read()

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
        
    status = "MISSING"
    if url:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'})
        try:
            resp = urllib.request.urlopen(req, context=ctx, timeout=10)
            status = f"OK ({resp.getcode()})"
        except urllib.error.HTTPError as e:
            status = f"HTTP ERROR {e.code}"
        except urllib.error.URLError as e:
            status = f"URL ERROR {e.reason}"
        except Exception as e:
            status = f"ERROR {str(e)}"
            
    results.append({'id': entry_id, 'url': url, 'doi': doi, 'status': status})

print(json.dumps(results, indent=2))
