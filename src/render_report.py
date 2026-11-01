"""Render the project's limited Markdown dialect as a clean paginated PDF.
Requires requirements-report.txt and a locally installed Unicode font.
No font binaries are distributed with the project.
Usage: python -m src.render_report [--source reports/report.md] [--output reports/report.pdf]
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
from xml.sax.saxutils import escape
from .common import ROOT


def main():
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, Preformatted, KeepTogether
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default=str(ROOT/'reports'/'report.md'))
    p.add_argument('--output',default=str(ROOT/'reports'/'report.pdf'))
    p.add_argument('--final',action='store_true',help='Use only after updating the report with actual GPU and judge results.')
    a=p.parse_args();source=Path(a.source);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    if a.final:
        required=[ROOT/'artifacts/adapter/experiment.json', ROOT/'artifacts/metrics/automatic_metrics.json', ROOT/'artifacts/judge/summary.json']
        if any(not f.exists() for f in required):
            raise RuntimeError('Final mode requires real training, automatic evaluation and judge summaries. Update the Markdown before rendering.')
    font_dir=Path('/usr/share/fonts/truetype/dejavu')
    if not (font_dir/'DejaVuSans.ttf').exists():
        raise RuntimeError('Install a Unicode font such as DejaVu Sans locally, or edit font_dir in this script.')
    for name,file in [('Body','DejaVuSans.ttf'),('Bold','DejaVuSans-Bold.ttf'),('Mono','DejaVuSansMono.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(font_dir/file)))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Body',boldItalic='Bold')
    # PDF colors are layout accents, not chart semantics.
    ink=colors.HexColor('#203044');muted=colors.HexColor('#5E6B79');accent=colors.HexColor('#24566B')
    styles={
       'body':ParagraphStyle('body',fontName='Body',fontSize=9.4,leading=13.4,textColor=ink,spaceAfter=7),
       'h1':ParagraphStyle('h1',fontName='Bold',fontSize=25,leading=30,textColor=ink,spaceAfter=10),
       'h2':ParagraphStyle('h2',fontName='Bold',fontSize=16.5,leading=21,textColor=accent,spaceBefore=2,spaceAfter=12,keepWithNext=True),
       'h3':ParagraphStyle('h3',fontName='Bold',fontSize=10.5,leading=14,textColor=ink,spaceBefore=5,spaceAfter=7,keepWithNext=True),
       'table':ParagraphStyle('table',fontName='Body',fontSize=8.0,leading=10.6,textColor=ink),
       'th':ParagraphStyle('th',fontName='Bold',fontSize=8.0,leading=10.6,textColor=colors.white),
       'caption':ParagraphStyle('caption',fontName='Body',fontSize=7.4,leading=10,textColor=muted,spaceAfter=5),
       'code':ParagraphStyle('code',fontName='Mono',fontSize=7.7,leading=10.5,textColor=ink,backColor=colors.HexColor('#F1F4F6'),borderPadding=6,spaceAfter=9),
    }
    def inline(s):
        s=escape(s)
        s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
        s=re.sub(r'`([^`]+)`',r'<font name="Mono" size="8">\1</font>',s)
        return s
    lines=source.read_text(encoding='utf-8').splitlines();story=[];i=0
    width=A4[0]-88
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line=='<!-- PAGE_BREAK -->':story.append(PageBreak());i+=1;continue
        if line.startswith('```'):
            i+=1;block=[]
            while i<len(lines) and not lines[i].startswith('```'):block.append(lines[i]);i+=1
            story.append(Preformatted('\n'.join(block),styles['code'],maxLineLength=95));i+=1;continue
        if line.startswith('|'):
            table_lines=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                l=lines[i].strip();i+=1
                if re.fullmatch(r'[|:\-\s]+',l):continue
                table_lines.append([x.strip() for x in l.strip('|').split('|')])
            n=len(table_lines[0]);data=[]
            for ri,row in enumerate(table_lines):data.append([Paragraph(inline(x),styles['th' if ri==0 else 'table']) for x in row])
            if n==2:colwidths=[width*.34,width*.66]
            elif n==3:colwidths=[width*.40,width*.30,width*.30]
            elif n==4:colwidths=[width*.35,width*.16,width*.16,width*.33]
            elif n==7:colwidths=[width*.20]+[width*.8/6]*6
            else:colwidths=[width/n]*n
            table=Table(data,colWidths=colwidths,hAlign='LEFT',repeatRows=1)
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),accent),('VALIGN',(0,0),(-1,-1),'TOP'),
                 ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
                 ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
                 ('LINEBELOW',(0,0),(-1,0),.6,accent),('LINEBELOW',(0,1),(-1,-1),.35,colors.HexColor('#D9E0E5')),
                 ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F5F7F8')])]))
            story.extend([table,Spacer(1,9)]);continue
        image=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',line)
        if image:
            path=(source.parent/image.group(2)).resolve()
            img=Image(str(path));scale=min(width/img.imageWidth,140/img.imageHeight)
            img.drawWidth=img.imageWidth*scale;img.drawHeight=img.imageHeight*scale;img.hAlign='CENTER'
            story.append(KeepTogether([img,Paragraph(inline(image.group(1)),styles['caption'])]));i+=1;continue
        if line.startswith('#'):
            level=len(line)-len(line.lstrip('#'));text=line[level:].strip()
            story.append(Paragraph(inline(text),styles['h'+str(min(level,3))]));i+=1;continue
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(('#','|','```','<!-- PAGE_BREAK -->','![')):
            paragraph.append(lines[i].strip());i+=1
        story.append(Paragraph(inline(' '.join(paragraph)),styles['body']))
    def page(canvas,doc):
        canvas.setFillColor(muted);canvas.setFont('Body',7.6)
        canvas.drawString(44,A4[1]-28,'РК 1  /  ДОМ И КВАРТИРА')
        canvas.drawRightString(A4[0]-44,A4[1]-28,'РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА' if a.final else 'ПОДГОТОВИТЕЛЬНАЯ ВЕРСИЯ')
        canvas.setStrokeColor(colors.HexColor('#D9E0E5'));canvas.line(44,40,A4[0]-44,40)
        canvas.drawString(44,27,'Дом и квартира / SFT и независимая оценка' if a.final else 'GPU-эксперимент и независимая оценка еще не выполнены')
        canvas.drawRightString(A4[0]-44,27,str(doc.page))
    doc=SimpleDocTemplate(str(out),pagesize=A4,leftMargin=44,rightMargin=44,topMargin=48,bottomMargin=51,
                          title='Дом и квартира — рубежный контроль 1',author='Учебный проект',
                          subject='Подготовительный отчет: код, данные и протокол QLoRA/SFT')
    doc.build(story,onFirstPage=page,onLaterPages=page)
    print(out)

if __name__=='__main__':main()
