#!/usr/bin/python3

from xml.dom import minidom
from nltk.translate import AlignedSent, IBMModel2

def build_dict_from_xml(doc):
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
    doc_en = minidom.parse("../../ted_en-20160408.xml")
    talks_en = build_dict_from_xml(doc_en)

    doc_hu = minidom.parse("../../ted_hu-20160408.xml")
    talks_hu = build_dict_from_xml(doc_hu)

    sentence_pairs = []

    for talkid, sentences in talks_en.items():
        for sent_id, sent_en in sentences.items():
            if talkid in talks_hu:
                sentences_hu = talks_hu[talkid]
                if sent_id in sentences_hu:
                    sent_hu = sentences_hu[sent_id]
                    sentence_pairs.append((sent_en, sent_hu))

    return sentence_pairs

def build_ibm2_model(sentence_pairs):
    bitext = []

    for (sent_en, sent_hu) in sentence_pairs:
        bitext.append(AlignedSent(sent_en.split(" "), sent_hu.split(" ")))

    return IBMModel2(bitext, 5)

def main():
    sentence_pairs = get_aligned_sentences()
    ibm2 = build_ibm2_model(sentence_pairs)

if __name__ == '__main__':
    main()
