#!/usr/bin/python3

from xml.dom import minidom

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

def main():
    sentence_pairs = get_aligned_sentences()
    
    for pair in sentence_pairs:
        print(pair)

if __name__ == '__main__':
    main()
