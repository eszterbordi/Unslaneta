#!/usr/bin/python3

from sys import argv, exit
from os import pathsep

from xml.dom import minidom
from nltk.translate import AlignedSent, Alignment, IBMModel2, IBMModel1, phrase_based, PhraseTable, StackDecoder
from collections import defaultdict

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

    for talkid, sentences in talks_en.items():
        for sent_id, sent_en in sentences.items():
            if talkid in talks_hu:
                sentences_hu = talks_hu[talkid]
                if sent_id in sentences_hu:
                    sent_hu = sentences_hu[sent_id]
                    sent_hu = sent_hu.replace(', ', ' , ')
                    sent_hu = sent_hu.replace('.', '')
                    sent_en = sent_en.replace(', ', ' , ')
                    sent_en = sent_en.replace('.', '')
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

        alignment_str = alignment_str.replace("Alignment", "").replace("), ", "#").replace(", ", "-").replace("#(", " ").replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        bitext_new.append(AlignedSent(b.words, b.mots, Alignment.fromstring(alignment_str)))
    return bitext_new

def build_phrases(bitext):

    print("--- Building phrases")
    phrases = []

    for b in bitext:
        bitext_words = ' '.join(word for word in b.words)
        bitext_mots = ' '.join(mot for mot in b.mots)
        #print(bitext_words)
        #print(bitext_mots)
        #print(b.alignment.__repr__())

        phrase = phrase_based.phrase_extraction(bitext_words, bitext_mots, b.alignment)
        phrases.append(phrase)

    return phrases

def build_phrase_table(phrases):
    phrase_counts = {}
    phrase_translation_counts = {}

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

    phrase_table = PhraseTable()

    for phrase_set in phrases:
        for (src_pos, target_pos, src_phrase, target_phrase) in phrase_set:
            probability = phrase_translation_counts[src_phrase][target_phrase] / phrase_counts[src_phrase]
            phrase_table.add(tuple(src_phrase.split(' ')), tuple(target_phrase.split(' ')), math.log(probability))

    return phrase_table

def persist_model(model_filename, model):

    print("--- Started the trained model dump")
    with open(model_filename, 'wb') as fout:
        persist.dump(model, fout)

def load_model(model_filename):

    print("--- Started the trained model load")
    with open(model_filename, 'rb') as fin:
        model = persist.load(fin)
    return model

def main():

    #if len(argv) != 2:
    #    print("Usage: " + argv[0].parse(pathsep)[-1] + " <persistedmodel_file>")
    #    exit(1)
    #model_filename = argv[1]
    #print("--- Start\nPersisted model file path: " + model_filename)

    sentence_pairs = get_aligned_sentences()
    print("--- Started training")
    ibm2, bitext = build_ibm2_model(sentence_pairs)
    bitext = remove_nones(bitext)

    #ibm2 = load_model('./models/ibm2.p')
    #bitext = load_model('./models/bitext.p')

    persist_model('./models/ibm2.p', ibm2)
    persist_model('./models/bitext.p', bitext)

    phrases = build_phrases(bitext)
    phrase_table = build_phrase_table(phrases)

    language_prob = defaultdict(lambda: 0.0)
    language_model = type('',(object,),{'probability_change': lambda self, context, phrase: language_prob[phrase], 'probability': lambda self, phrase: language_prob[phrase]})()

    stack_decoder = StackDecoder(phrase_table, language_model)

    stack_decoder.translate(['I', 'am', 'going', 'to', 'school'])

if __name__ == '__main__':
    main()
