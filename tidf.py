# -*- coding: utf-8 -*-
from __future__ import division,absolute_import
import docx2txt
import math
import re


class TDIF:

    def __init__(self,file_uploads):
        #self.filenames = ['Ṣé Wàá Gba Ẹ̀bùn Ọlọ́run Tó Dára Jù.docx']
        self.single_file = ['train','test']
        self.filenames = file_uploads
        self.freq = {}
        self.new_freq = {}
        self.new_frequency = {}
        self.normal_freq = {}
        self.new_entrop = {}
        self.new_variance = {}
        self.merge_word = {}
        self.word_count = {}
        self.idf_total = {}
        self.tfidf_total = {}
        self.splitted_words = ''
        self.new_tfidf = {}

    def split_list(self,the_list, chunk_size):
        result_list = []
        while the_list:
            result_list.append(the_list[:chunk_size])
            the_list = the_list[chunk_size:]
        return result_list

    def count_file(self):
        if len(self.filenames) ==1:
            return 1
        elif len(self.filenames) ==2:
            return 2

    def text_doc(self):
        if self.count_file() ==2:
            print("yes")
            for a in range(len(self.filenames)):
                my_text = docx2txt.process(self.filenames[a])
                with open("output_%d.txt" % a, 'w', encoding='utf-8') as text_file:
                    text_file.write(my_text)
                    print("sucessfully written files to text")
        elif self.count_file() == 1:
            for a in range(len(self.filenames)):
                my_text = docx2txt.process(self.filenames[a])
                with open("output_%d.txt" % a, 'w', encoding='utf-8') as text_file:
                    text_file.write(my_text)
                    print("sucessfully written files to text")


    def remove_puncts(self):
        if self.count_file() == 2:
            print("yes")
            for a in range(len(self.filenames)):
                with open("output_%d.txt" % a, 'r', encoding='utf-8') as text_file:
                    read = text_file.read()
                    text_string = read.lower()
                    punctuations = "!.,?);(:{}[]*0123456789-"
                    validwords = "u'([a-zA-ZÀÁÈÉẸẸ̀Ẹ́Ọ̀ỌỌ́ÙÓÚÒṢàùÌÍèọ̀èáéọòẹẹ́ìíọ́óúẹ̀ṣ]*)"
                    text_string = text_string.translate(punctuations)
                    text_string = re.sub(validwords,"",text_string)
                    self.new_freq[a] = text_string.split()
        elif self.count_file() == 1:
            for a in range(len(self.single_file)):
                with open("output_1.txt", 'r', encoding='utf-8') as text_file:
                    read = text_file.read()
                    text_string = read.lower()
                    punctuations = "!.,?);(:{}[]*0123456789-"
                    validwords = "u'([a-zA-ZÀÁÈÉẸẸ̀Ẹ́Ọ̀ỌỌ́ÙÓÚÒṢàùÌÍèọ̀èáéọòẹẹ́ìíọ́óúẹ̀ṣ]*)"
                    text_string = text_string.translate(punctuations)
                    text_string = re.sub(validwords,"",text_string)
                    items = text_string.split()
                    self.splitted_words = items
                    length_items = int(len(items)/2)
                    self.new_freq[a] = self.split_list(items,length_items)[a]

    def frequency_check(self):
        i = 0
        for word in self.freq[i]:
            self.merge_word[word] = self.freq[i][word]
        for words in self.freq[i+1]:
            self.merge_word[words] = self.freq[i+1][words]
        #calculate the frequency here
        self.freq_calc()
        for joinwords in self.normal_freq:
            if self.normal_freq[joinwords] > 5:
                return True
            else:
                return False

    def freq_calc(self):
        for words in self.splitted_words:
            if len(words) < 5 and not words.isdigit():
                if words in self.normal_freq:
                    self.normal_freq[words] = self.normal_freq[words] + 1
                else:
                    self.normal_freq[words] = 1
    def freq_answer(self):
        average,length = self.freq_avg()
        for words in self.normal_freq:
            if self.frequency_check():
                if float(int(self.normal_freq[words])/length) > length/1000:
                    self.new_frequency[words]=1


    def freq_avg(self):
        add = sum(self.normal_freq.values())
        print('add',add)
        length = len(self.normal_freq)
        print('length',length)
        average = float(add / length)
        return average,length

    def ent_calc(self,ent=0):
        average,length = self.freq_avg()
        for words in self.normal_freq:
            if self.frequency_check():
                if ent < 0 and ent != 0 and self.normal_freq[words] > 5:
                    self.new_entrop[words] = 1
                ent = float(ent + float(self.normal_freq[words] / length) * float(math.log(float(self.normal_freq[words]) / length, 2)))
                ent = float(-ent)

    def var_calc(self):
        average,length = self.freq_avg()

        print(average,length)
        for words in self.normal_freq:
            if self.frequency_check():
                if float(((average - int(self.normal_freq[words])) ** 2) / length) > length and self.normal_freq[words] > 5:
                    self.new_variance[words] = 1

    def count_words(self):
        for i in range(len(self.new_freq)):
            self.freq[i] = {}
            for word in self.new_freq[i]:
                if len(word) < 5 and not word.isdigit():
                    if word in self.freq[i]:
                        self.freq[i][word] = self.freq[i][word] + 1
                    else:
                        self.freq[i][word] = 1
        return self.freq

    def tf(self,word, blob):
       # print(word,blob)
        return word/blob

    def caltf(self):
        self.count_words()
        self.word_occurence()
        for i in range(len(self.new_freq)):
            for word in self.freq[i]:
                words = self.freq[i]
                length = len(self.freq[i])
                self.freq[i][word] = self.tf(words[word], length)
        return self.freq

    def word_occurence(self):
        i=0
        for word in self.freq[i]:
            if word in self.freq[i+1]:
                #print(word, self.freq[i][word],self.freq[i+1][word])
                total = (self.freq[i][word]) + (self.freq[i+1][word])
                self.word_count[word] = total

    def n_containing(self,word, bloblist):
        return sum(1 for blob in bloblist if word in blob.words)

    def idf(self):
        for word in self.word_count:
            self.idf_total[word] = (math.log(len(self.new_freq) / (self.word_count[word])))
        return self.idf_total

    def tfidf(self):
        self.idf()
        k = 0
        if k < 2:
            for i in self.freq[k]:
                for j in self.idf_total:
                    if i == j:
                        self.tfidf_total[i] = self.freq[k][i] * self.idf_total[j]
        k = k+1
        return self.tfidf_total

    def avg_tfidf(self):
        length = len(self.tfidf_total)
        total = sum(self.tfidf_total.values())
        for words in self.tfidf_total:
            if self.tfidf_total[words] < float(total/length):
                self.new_tfidf[words] = 1


#app = TDIF()
#app.text_doc()
#app.remove_puncts()
#print(app.caltf())
#print(app.idf())
#print(app.tfidf())