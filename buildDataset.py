from corpus import Loader, Type
from dataset_util import convert_to_strings
from labeler import Labeler
from datasets import Dataset

def load_and_process_corpus(corpus_name, corpus_type):
    sent_toks, sent_pos = Loader().load_corpus(corpus_name, corpus_type=corpus_type, shuffle_sentences=False)
    sent_pos = [convert_to_strings(pos) for pos in sent_pos]
    return sent_toks, sent_pos

def label_text(sent_raw):
    chars, labels = Labeler().label_text(sent_raw, Type.sents_raw)
    return chars, labels

def create_dataset(sentences, tokens, characters, pos_labels, space_labels):
    data_dict = {
        "sentences": sentences[:-1],
        "tokens": tokens,
        "characters": characters[:-1],
        "pos_labels": pos_labels,
        "space_labels": space_labels[:-1]
    }
    return Dataset.from_dict(data_dict)

def main():
    dataset_path = "./dataset/"
    corpus_type = Type.sents_tok

    # Load and process corpora
    sent_toks, sent_pos = load_and_process_corpus("bijankhan", corpus_type)
    sent_toks_p, sent_pos_p = load_and_process_corpus("peykareh", corpus_type)
    sent_toks += sent_toks_p
    sent_pos += sent_pos_p

    # Load raw sentences
    sent_raw = Loader().load_corpus("all", corpus_type=Type.sents_raw, shuffle_sentences=False)

    # Label text
    chars, labels = label_text(sent_raw)

    # Create dataset
    dataset = create_dataset(sent_raw, sent_toks, chars, sent_pos, labels)
    dataset.save_to_disk(dataset_path)

    print("Dataset saved successfully!")

if __name__ == "__main__":
    main()
    dataset = Dataset.load_from_disk("./dataset/")