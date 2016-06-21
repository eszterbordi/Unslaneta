#!/usr/bin/python3

from xml.dom import minidom
from nltk.translate import AlignedSent, IBMModel2, IBMModel1, phrase_based

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
    doc_en = minidom.parse("./corpora/ted_en-20160408.xml")
    talks_en = build_dict_from_xml(doc_en)

    doc_hu = minidom.parse("./corpora/ted_hu-20160408.xml")
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
    return IBMModel2(bitext, 5), bitext

def build_phrases(sentence_pairs, bitext):

    print("Building phrases")
    phrases = []
    exc = 0
    for s, b in zip(sentence_pairs, bitext):
        try:
            phrase = phrase_based.phrase_extraction(s[0], s[1], b.alignment)
            phrases.append(phrase)
        except Exception as e:
            exc += 1
            # print(e)

    print("Nr exceptions: ", exc)

    return phrases

def main():
    print("start")
    sentence_pairs = get_aligned_sentences()
    ibm2, bitext = build_ibm2_model(sentence_pairs)
    phrases = build_phrases(sentence_pairs, bitext)

    print(round(ibm2.translation_table['könyv']['book'], 3))

if __name__ == '__main__':
    main()
