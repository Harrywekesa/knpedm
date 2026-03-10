import zipfile
import xml.etree.ElementTree as ET
import sys

def read_docx(path):
    word_schema = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    paragraphs = []
    
    with zipfile.ZipFile(path) as docx:
        tree = ET.XML(docx.read('word/document.xml'))
        for paragraph in tree.iter(word_schema + 'p'):
            texts = [node.text
                     for node in paragraph.iter(word_schema + 't')
                     if node.text]
            if texts:
                paragraphs.append(''.join(texts))
    
    return '\n'.join(paragraphs)

if __name__ == '__main__':
    with open('project_parsed.txt', 'w', encoding='utf-8') as f:
        f.write(read_docx(sys.argv[1]))
