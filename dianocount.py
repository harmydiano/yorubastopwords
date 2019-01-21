#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division,absolute_import
import re
import string
import math

def countmain(learn,test,time):
    print learn,test
    
    frequency = {}
    freq={}
    new_freq={}
    new_entrop={}
    new_variance={}
    document_text = open('Output3.txt', 'r')
    read=document_text.read()
    text_string = read.lower()
    punctuations = '!.,?);(:{}[]*0123456789-'
    text_string = text_string.translate(None,punctuations)
    new =text_string.split()
    total =len(new)
    validwords="u'([a-zA-ZÀÁÈÉẸẸ̀Ẹ́Ọ̀ỌỌ́ÙÓÚÒṢàùÌÍèọ̀èáéọòẹẹ́ìíọ́óúẹ̀ṣ]*)"
    ent=0.0
    for a in new:
        if len(a)<5 and not a.isdigit():
            if freq.has_key(a):
                freq[a] = freq[a]+1
            else:
                freq[a] = 1
    add=sum(freq.values())
    length=len(freq)
    average=float(add/length)

    def frequency_check():
        for words in freq:
            if freq[words] > 5:
                return True
            else:
                return False

    def ent_calc():
        for words in freq:
            if frequency_check():
                if ent < 0 and freq[words] > 5:
                    new_entrop[words] = 1

    def var_calc():
        for word in freq:
            if frequency_check():
                if float(((average - int(freq[words])) ** 2) / length) > length and freq[words] > 5:
                    new_variance[words] = 1



    
    for words in freq:
        if freq[words] >5:
            if float(int(freq[words])/length) >length/1000:
                new_freq[words]=1
            if ent<0 and freq[words] >5:
                new_entrop[words]=1
            if float(((average-int(freq[words]))**2)/length) > length and freq[words] >5:
                new_variance[words] = 1
            ent = float(ent + float(freq[words]/length) *float(math.log(float(freq[words])/length,2)))
            ent=float(-ent)
            print words, freq[words], "frequency is: %.5f and  entropy is %.3f variance: %.3f" % (float(int(freq[words])/length), ent, float(((average-freq[words])**2)/length))
    print "%d words" % length
    print "average is %.2f" % float(add/length)
    
    
    filename_f = "%sfrequency.txt" %time
    filename_e = "%sentropy.txt" %time
    filename_v = "%svariance.txt" %time
    filename_fe = "%sfreqandentropystopwords.txt" %time
    filename_fv = "%sfreqandvariancestopwords.txt" %time
    filename_ev = "%sfreqandvariancestopwords.txt" %time
    filename_os = "%soriginalstopwords.txt" %time
    
    with open(learn+filename_f, "w") as text_file:
        print learn+filename_f
        with open(test+filename_f, "w") as text_file1:
            for a in new_freq:
                text_file.write("{}\n".format(a))
                text_file1.write("{}\n".format(a))
    
    with open(learn+filename_e, "w") as ent_file:
        with open(test+filename_e, "w") as ent_file1:
            for b in new_entrop:
                ent_file.write("{}\n".format(b))
                ent_file1.write("{}\n".format(b))
            
    with open(learn+filename_v, "w") as var_file:
        with open(test+filename_v, "w") as var_file1:
            for c in new_variance:
                var_file.write("{}\n".format(c))
                var_file1.write("{}\n".format(c))
    
    read_freq = open(learn+filename_f, 'r')
    readf=read_freq.readlines()
    read_ent= open(learn+filename_e, 'r')
    reade=read_ent.readlines()
    read_var= open(learn+filename_v, 'r')
    readv=read_var.readlines()
    
    with open(learn+filename_fe, "w") as freqe_file:
        with open(test+filename_fe, "w") as freqe_file1:
            for a in readf:
                if a in readf and a in reade:
                    freqe_file.write("{}\n".format(a))
                    freqe_file1.write("{}\n".format(a))
    with open(learn+filename_fv, "w") as freqv_file:
        
        with open(test+filename_fv, "w") as freqv_file1:
            for a in readf:
                if a in readf and a in readv:
                    freqv_file.write("{}\n".format(a))
                    freqv_file1.write("{}\n".format(a))
    with open(learn+filename_ev, "w") as ev_file:
        
        with open(test+filename_ev, "w") as ev_file1:
            for a in reade:
                if a in reade and a in readv:
                    ev_file.write("{}\n".format(a))
                    ev_file1.write("{}\n".format(a))
    with open(learn+filename_os, "w") as stopw_file:
        with open(test+filename_os, "w") as stopw_file1:
            for a in readf:
                if a in readf and a in reade and a in readv:
                    stopw_file.write("{}\n".format(a))  
                    stopw_file1.write("{}\n".format(a))


