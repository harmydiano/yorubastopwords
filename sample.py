import docx2txt
filenames = ['Ṣé Wàá Gba Ẹ̀bùn Ọlọ́run Tó Dára Jù.docx']
for a in range(len(filenames)):
    my_text = docx2txt.process(filenames[a])
    with open("output_%d.txt" % a, 'w', encoding='utf-8') as text_file:
        text_file.write(my_text)
        print("sucessfully written files to text")