#!/usr/bin/python3

from nltk.translate import StackDecoder
import dill as persist

def load_model(model_filename):

    print("--- Started the trained model load")
    with open(model_filename, 'rb') as fin:
        model = persist.load(fin)
    return model

def main():
    stack_decoder = load_model('./models/decoder.p')

    print("--- Translator ready, start typing sentences!")

    while True:
        sent = input("> ")
        sent_list = sent.split(' ')
        print("> Translation:", stack_decoder.translate(sent_list))

if __name__ == '__main__':
    main()
