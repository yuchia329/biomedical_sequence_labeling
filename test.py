import pandas as pd
pd.set_option('display.max_rows', None)


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    sentences = []
    current_sentence = []
    current_tag = []
    for line in lines:
        items = line.split("\t")
        if len(items)==2:  # Non-empty line
            text = items[0]
            tag = items[1][:-1]
            print(tag)
            current_sentence.append(text)
            current_tag.append(tag)
        else:  # Empty line indicates the end of a sentence
            if current_sentence:
                sentences.append((current_sentence, current_tag))
                current_sentence = []
                current_tag = []
    return sentences

data = load_data('A2-data/dev.answers')
print(data)