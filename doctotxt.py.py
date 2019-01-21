
import docx2txt
def doctotxt(filename):
    my_text = docx2txt.process(filename)
    with open(".\\mynewdirectory\\Output2.txt", "w") as text_file:
        return text_file.write((my_text).encode("utf-8"))
