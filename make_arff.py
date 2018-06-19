import os
import glob
import pickle
from scipy.io import arff
from nltk.stem import WordNetLemmatizer
from nltk import download
from random import shuffle

WORD_COUNT = 10
PARAGRAPH_COUNT = 500
STOPWORDS_PATH = 'books/stopwords.txt'
PUNCTUATION = ['"', ':', ',', '.', '!', '?', ';', '(', ')', '\'s', '\'']
STOPWORDS = open(STOPWORDS_PATH).read().strip('\n').split('\n\n')
WORDNET_LEMMATIZER = WordNetLemmatizer()

def lemmatize(s):
    return WORDNET_LEMMATIZER.lemmatize(s)

def printd(d):
    for k,v in d.items():
        print(k, v)

def get_files():
    text_files = glob.glob('books/*.txt')
    files = { }
    sizes = { }
    for f in text_files:
        key = f.split('\\')[-1]
        key = key.replace('.txt', '')
        if key == 'stopwords':
            continue
        data = open(f).read().split('\n\n') # split by paragraph
        author, title = key.split('-')
        parsed_data = [ ]
        shuffle(data)
        for paragraph in data[:PARAGRAPH_COUNT]:
            parsed_paragraph = parse_paragraph(paragraph)
            print('\t%s' % parsed_paragraph)
            parsed_data.append(parsed_paragraph)
        print('Adding %d words from %s by %s' % (len(parsed_data)*WORD_COUNT, title, author))
        if author in files:
            files[author] += parsed_data
        else:
            files[author] = parsed_data
        if author in sizes:
            sizes[author] += os.path.getsize(f)
        else:
            sizes[author] = os.path.getsize(f)
    print('Bytes per Author: ')
    printd(sizes)
    print()
    return files

def parse_word(word, li):
    def remove_punctuation(word):
        if word.isalnum():
            return word
        for c in PUNCTUATION:
            if c in word:
                word = word.replace(c, '')
        return word
    word = remove_punctuation(word).lower()
    if word in STOPWORDS or word == '':
        return li
    lemmatized_word = lemmatize(word)
    if lemmatized_word in STOPWORDS:
        return li
    li += [ lemmatized_word ]
    return li

def parse_paragraph(paragraph):
    li = [x for x in paragraph.replace('\n', ' ').split(' ') if x != '-']
    new_paragraph = [ ]
    shuffle(li)
    for s in li[:WORD_COUNT]:
        if '--' in s:  # could do this in separate pass
            words = s.split('--')
            for w in words:
                li = parse_word(w, new_paragraph)
            continue
        li = parse_word(s, new_paragraph)
    return ' '.join(new_paragraph)

def get_word_counts(files):
    counts = { }
    for author, book in files.items():
        for paragraph in book:
            for w in paragraph.split(' '):
                if w == '':
                    continue
                if w in counts:
                    counts[w] += 1
                else:
                    counts[w] = 1
    return counts

def make_arff_meta(name, files, attributes):
    print('Creating arff metadata ...')
    file_ = '@relation %s\n\n' % name
    print('Adding Relation: %s' % file_, end='\r')
    file_ += '@attribute %s {' % name
    for author, book in files.items():
        file_ += author + ', '
    file_ = file_.strip(', ')
    file_ += '}\n\n'
    for word, _ in attributes:
        s = '@attribute ' + word + ' {0, 1}\n'
        print('Adding Attribute: %s' % s, end='\r')
        file_ += s
    return file_

def get_sampled_sorted_word_list(files, n):
    word_counts = get_word_counts(files)
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1])
    sampled_sorted_words = sorted_words[-(n-1):]
    f = open('wordcount.pylist', 'wb')
    pickle.dump(sampled_sorted_words, f)
    return sampled_sorted_words

def make_arff(dest_path, n):
    def make_arff_data_row(author, paragraph, li):
        record = '%s,' % author
        for name, count in li:
            if name in paragraph:
                record += '1,'
            else:
                record += '0,'
        record = record.strip(',')
        return record

    files = get_files()
    print('Generating arff file with %d features' % n)
    sampled_sorted_words = get_sampled_sorted_word_list(files, n)
    file_ = make_arff_meta('_author_', files, sampled_sorted_words)

    file_ += '''\n@data\n'''
    for author, book in files.items():
        for paragraph in book:
            record = make_arff_data_row(author, paragraph, sampled_sorted_words)
            file_ += record + '\n'
    with open(dest_path, 'w') as f:
        f.write(file_)
    return file_

def load_arff(path):
    try:
        arff_file = arff.loadarff(path)
    except Exception as e:
        raise Exception('could not load a valid arff file ')
    return arff_file

def main(argc, argv):
    if argc < 2:
        print('usage: make_arff.py <int>')
        return
    else:
        n = argv[1]
    download('wordnet')
    arff_path = 'data.arff'
    make_arff(arff_path, int(n))
    load_arff(arff_path)

if __name__ == '__main__':
    import sys
    main(len(sys.argv), sys.argv)

