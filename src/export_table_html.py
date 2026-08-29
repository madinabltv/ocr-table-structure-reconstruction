from __future__ import annotations
import argparse, html, json
from pathlib import Path

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--header',required=True,type=Path)
    p.add_argument('--body',required=True,type=Path); p.add_argument('--output',required=True,type=Path)
    p.add_argument('--header-overrides',type=Path); p.add_argument('--title',default='Восстановленная таблица')
    return p.parse_args()

def main():
    a=parse_args(); header=json.loads(a.header.read_text(encoding='utf-8')); body=json.loads(a.body.read_text(encoding='utf-8'))
    overrides={}
    if a.header_overrides: overrides=json.loads(a.header_overrides.read_text(encoding='utf-8'))['cells']
    by_row={}
    for cell in header['cells']: by_row.setdefault(cell['row'],[]).append(cell)
    lines=['<!doctype html>','<html lang="ru"><head><meta charset="utf-8">',f'<title>{html.escape(a.title)}</title>',
           '<style>body{font-family:Arial,sans-serif;margin:32px}table{border-collapse:collapse;max-width:1200px;width:100%}th,td{border:1px solid #555;padding:8px 10px;text-align:center;vertical-align:middle}th{background:#f0f2f5}.missing{background:#fff3cd;color:#775b00}.meta{color:#555;font-size:14px}</style>',
           '</head><body>',f'<h1>{html.escape(a.title)}</h1>',
           '<p class="meta">Структура восстановлена по сегментам линий; пустые OCR-ячейки отмечены цветом.</p>','<table><thead>']
    for row in range(header['header_rows']):
        lines.append('<tr>')
        for cell in sorted(by_row.get(row,[]),key=lambda c:c['column']):
            key=f"{cell['row']},{cell['column']}"; text=overrides.get(key,cell['text']) or '[OCR: текст не распознан]'
            attrs=[]
            if cell['rowspan']>1: attrs.append(f'rowspan="{cell["rowspan"]}"')
            if cell['colspan']>1: attrs.append(f'colspan="{cell["colspan"]}"')
            lines.append(f'<th {" ".join(attrs)}>{html.escape(text)}</th>')
        lines.append('</tr>')
    lines.append('</thead><tbody>')
    for row in body['rows']:
        lines.append('<tr>')
        for cell in row['cells']:
            cls=' class="missing"' if cell.get('missing') else ''
            text=cell['text'] if cell['text'] else '[пропуск OCR]'
            lines.append(f'<td{cls}>{html.escape(text)}</td>')
        lines.append('</tr>')
    lines.extend(['</tbody></table>','</body></html>'])
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text('\n'.join(lines),encoding='utf-8')
    print(f"Header cells: {len(header['cells'])}"); print(f"Body rows: {len(body['rows'])}")
    print(f"Overrides used: {len(overrides)}"); print(f"HTML: {a.output}")
if __name__=='__main__': main()
