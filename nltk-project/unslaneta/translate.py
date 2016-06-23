#!/usr/bin/python3

from sys import argv, exit
from os import pathsep

from xml.dom import minidom
from nltk.translate import AlignedSent, Alignment, IBMModel2, IBMModel1, phrase_based, PhraseTable, StackDecoder
from collections import defaultdict

import string
import math
import dill as persist
import re


def build_dict_from_xml(doc):

    print("--- Processing data from xml")
    talks = {}

    for talk in doc.getElementsByTagName("head"):
        talkid = talk.getElementsByTagName("talkid")[0].firstChild.data
        talks[talkid] = {}

        sentenceNodes = talk.getElementsByTagName("seekvideo")

        for sentence in sentenceNodes:
            id = sentence.getAttribute("id")
            text = sentence.firstChild.data

            talks[talkid][id] = text

    return talks


def get_aligned_sentences():

    print("--- Aligning sentences")
    doc_en = minidom.parse("./corpora/ted_en-20160408.xml")
    #doc_en = minidom.parse("./corpora/ted_en-small.xml")
    talks_en = build_dict_from_xml(doc_en)

    doc_hu = minidom.parse("./corpora/ted_hu-20160408.xml")
    #doc_hu = minidom.parse("./corpora/ted_hu-small.xml")
    talks_hu = build_dict_from_xml(doc_hu)

    sentence_pairs = []

    regex1 = re.compile(r"—")
    regex2 = re.compile(r" +")

    for talkid, sentences in talks_en.items():
        for sent_id, sent_en in sentences.items():
            if talkid in talks_hu:
                sentences_hu = talks_hu[talkid]
                if sent_id in sentences_hu:
                    sent_hu = sentences_hu[sent_id]
                    sent_hu = sent_hu.translate(str.maketrans("","", string.punctuation))
                    sent_hu = re.sub(regex1, "", sent_hu)
                    sent_hu = re.sub(regex2, " ", sent_hu).lstrip().rstrip()
                    sent_en = sent_en.translate(str.maketrans("","", string.punctuation))
                    sent_en = re.sub(regex1, "", sent_en)
                    sent_en = re.sub(regex2, " ", sent_en).lstrip().rstrip()
                    if sent_en and sent_hu:
                        sentence_pairs.append((sent_en, sent_hu))

    return sentence_pairs

def build_ibm2_model(sentence_pairs):

    print("--- Building IBM2 Model")
    bitext = []

    for (sent_en, sent_hu) in sentence_pairs:
        bitext.append(AlignedSent(sent_en.split(" "), sent_hu.split(" ")))

    return IBMModel2(bitext, 5), bitext

def remove_nones(bitext):

    bitext_new = []
    regex1 = re.compile(r"\([0-9]+, None\), ", re.IGNORECASE)
    regex2 = re.compile(r"\(None, [0-9]+\), ", re.IGNORECASE)
    regex3 = re.compile(r"\([0-9]+, None\)", re.IGNORECASE)
    regex4 = re.compile(r"\(None, [0-9]+\)", re.IGNORECASE)

    for b in bitext:
        alignment_str = re.sub(regex1, "", b.alignment.unicode_repr())
        alignment_str = re.sub(regex2, "", alignment_str)
        alignment_str = re.sub(regex3, "", alignment_str)
        alignment_str = re.sub(regex4, "", alignment_str)

        alignment_str = alignment_str.replace("Alignment", "").replace("), ", "#").replace(", ", "-").replace("#(", " ").replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace("#", "")
        bitext_new.append(AlignedSent(b.words, b.mots, Alignment.fromstring(alignment_str)))
    return bitext_new

def build_phrases(bitext):

    print("--- Building phrases")
    phrases = []

    for b in bitext:
        bitext_words = ' '.join(word for word in b.words if word != '')
        bitext_mots = ' '.join(mot for mot in b.mots if mot != '')
        phrase = phrase_based.phrase_extraction(bitext_words, bitext_mots, b.alignment, 2)
        phrases.append(phrase)

    return phrases

def build_phrase_table(phrases):
    print("--- Building phrase table")

    phrase_counts = {}
    phrase_translation_counts = {}

    print("--- Collecting counts")

    for phrase_set in phrases:
        for (src_pos, target_pos, src_phrase, target_phrase) in phrase_set:
            if src_phrase in phrase_counts:
                phrase_counts[src_phrase] += 1
                if target_phrase in phrase_translation_counts[src_phrase]:
                    phrase_translation_counts[src_phrase][target_phrase] += 1
                else:
                    phrase_translation_counts[src_phrase][target_phrase] = 1
            else:
                phrase_counts[src_phrase] = 1
                phrase_translation_counts[src_phrase] = {}
                phrase_translation_counts[src_phrase][target_phrase] = 1

    print("--- Populating phrase table")

    phrase_table = PhraseTable()

    total_phrase_count = len(phrases)
    i = 0

    for phrase_set in phrases:
        if (i % 10000 == 0):
            print("--- --- Processing phrase set", i, "out of", total_phrase_count)
        for (src_pos, target_pos, src_phrase, target_phrase) in phrase_set:
            probability = phrase_translation_counts[src_phrase][target_phrase] / phrase_counts[src_phrase]
            phrase_table.add(tuple(src_phrase.split(' ')), tuple(target_phrase.split(' ')), math.log(probability))
        i += 1

    return phrase_table

def persist_model(model_filename, model):

    print("--- Started the trained model dump:", model_filename)
    with open(model_filename, 'wb') as fout:
        persist.dump(model, fout)

def load_model(model_filename):

    print("--- Started the trained model load:", model_filename)
    with open(model_filename, 'rb') as fin:
        model = persist.load(fin)
    return model

def build_language_model(bitext, phrases):
    print("--- Building language model")

    word_count = {}
    biword_count = {}
    nr_words = 0

    for b in bitext:
        phrase_size = len(b.mots)
        nr_words += phrase_size
        for i in range(0, phrase_size):
            word = b.mots[i].lower()
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

            if i > 0:
                biword = b.mots[i-1].lower() + ' ' + word
                if biword in biword_count:
                    biword_count[biword] += 1
                else:
                    biword_count[biword] = 1

    language_prob = defaultdict(lambda: -999.0)

    for phrase_set in phrases:
        for (src_pos, target_pos, src_phrase, target_phrase) in phrase_set:
            phrase = target_phrase.split(' ')
            nr_exceptions = 0
            try:
                w = phrase[0].lower()
                if not w in word_count:
                    raise Exception("Bad word")
                prob = word_count[phrase[0].lower()] / nr_words

                for i in range(1,len(phrase)):
                    w1 = phrase[i-1].lower()
                    w2 = phrase[i].lower()
                    ww = w1+ ' ' + w2
                    if (ww not in biword_count) or (w1 not in word_count):
                        raise Exception("Bad word")
                    prob *= biword_count[ww] / word_count[w1]
                language_prob[tuple(phrase)] = math.log(prob)
            except:
                nr_exceptions += 1

    language_model = type('',(object,),{'probability_change': lambda self, context, phrase: language_prob[phrase], 'probability': lambda self, phrase: language_prob[phrase]})()
    return language_model

def main():
    #ibm2 = load_model('./models/ibm2.p')
    #bitext = load_model('./models/bitext.p')
    #phrases = load_model('./models/phrases.p')
    #phrase_table = load_model('./models/phrase_table.p')
    #language_model = load_model('./models/language_model.p')

    sentence_pairs = get_aligned_sentences()
    ibm2, bitext = build_ibm2_model(sentence_pairs)
    print("--- Started training")
    bitext = remove_nones(bitext)

    persist_model('./models/ibm2.p', ibm2)
    persist_model('./models/bitext.p', bitext)

    phrases = build_phrases(bitext)
    persist_model('./models/phrases.p', phrases)

    phrase_table = build_phrase_table(phrases)
    persist_model('./models/phrase_table.p', phrase_table)

    print("--- Started creating language model")

    language_model = build_language_model(bitext, phrases)
    persist_model('./models/language_model.p', language_model)

    print("--- Started building decoder")
    stack_decoder = StackDecoder(phrase_table, language_model)
    persist_model('./models/decoder.p', stack_decoder)

    print("--- Ready")

if __name__ == '__main__':
    main()
