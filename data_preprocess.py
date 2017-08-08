# coding: utf-8
import os, json, time, re
import random, collections, cPickle
import numpy as np
import pandas as pd

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def process_raw_data(data_path):
    start_time = time.time()
    with open(data_path, 'r') as raw_input:
        counter = 0
        title_df = pd.DataFrame(columns=['title', 'pageView'])
        for line in raw_input:
            json_doc = json.loads(line)
            #expected_keys = ['pv_title', 'pv_url', 'pv_pageViews']
            expected_keys = ['title', 'url', 'traffic']
            if not all([key in json_doc.keys() for key in expected_keys]):
                continue
            title, url, pageView = json_doc['title'], json_doc['url'], json_doc['traffic']
            if not isEnglish(title):
                continue
            title_df.loc[url] = pd.Series({'title': title, 'pageView': pageView})
            counter += 1
            print 'finished processing {} rows using {:.2f} seconds'.format(counter, time.time() - start_time)

    title_df.index.name = 'url'
    print 'finished processing all the data using {:.2f} seconds'.format(time.time() - start_time)
    return title_df


def basic_tokenizer(line, normalize_digits=True):
    line = line.replace("'s", '')
    line = re.sub(r"\'ve", " have ", line)
    line = re.sub(r"can't", "cannot ", line)
    line = re.sub(r"n't", " not ", line)
    line = re.sub(r"I'm", "I am", line)
    line = re.sub(r" m ", " am ", line)
    line = re.sub(r"\'re", " are ", line)
    line = re.sub(r"\'d", " would ", line)
    line = re.sub(r"\'ll", " will ", line)
    line = re.sub(r"\?", " ? ", line)
    line = re.sub(r"!", " ! ", line)
    line = re.sub(r":", " : ", line)

    line = re.sub('[,."#%\'()*+/;<=>@\[\]^_{|}~\\\]', ' ', line)
    line = re.sub('[\n\t ]+', ' ', line)
    words = []
    _DIGIT_RE = re.compile(r"\d")
    for token in line.strip().lower().split():
        if not token:
            continue
        if normalize_digits:
            token = re.sub(_DIGIT_RE, b'#', token)
        words.append(token)
    return len(words), ' '.join(words)


def tokenize_title_column(data, processed_column_name):
    data['title_word_counts'], data[processed_column_name] = zip(*data['title'].map(basic_tokenizer))
    # sort by the title word counts and filter them
    sorted_data = data.sort_values(by=['title_word_counts', 'pageView'], ascending=[True, False])
    index = (sorted_data['title_word_counts'] >= 4) & (sorted_data['title_word_counts'] <= 15)
    filtered_data = sorted_data.loc[index, :]
    return filtered_data


_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

TOKEN_DICT = {}
REVERSE_TOKEN_DICT = {}
for i in xrange(len(_START_VOCAB)):
    TOKEN_DICT[_START_VOCAB[i]] = i
    REVERSE_TOKEN_DICT[i] = _START_VOCAB[i]


def create_vocab_dict(data, column_name, token_freq_threshold=5, UKN_frac_threshold=0.3):
    vocab_dict = {}
    all_titles = []
    selected_titles = []
    selected_title_urls = []
    selected_title_pageView = []

    for title, url, pageView in zip(data[column_name], data.index, data['pageView']):
        words = []
        for token in title.split(' '):
            words.append(token)
            if token not in vocab_dict:
                vocab_dict[token] = 0
            vocab_dict[token] += 1
        all_titles.append((words, url, pageView))
    print 'total {} tokens are ideintified...'.format(len(vocab_dict))

    token_dict, reverse_token_dict = TOKEN_DICT.copy(), REVERSE_TOKEN_DICT.copy()
    UKN_index = len(token_dict) - 1
    unique_counts = 0
    sorted_pairs = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
    for i, pair in enumerate(sorted_pairs):
        if pair[1] >= token_freq_threshold:
            unique_counts += 1
            token_dict[pair[0]] = i + 1 + UKN_index
            reverse_token_dict[(i + 1 + UKN_index)] = pair[0]
        else:
            token_dict[pair[0]] = UKN_index
    print 'total {} unique tokens are included in the token dictionary...'.format(unique_counts)

    for i in xrange(len(all_titles)):
        indexed_title = map(token_dict.get, all_titles[i][0])
        UKN_count = sum([elem == UKN_index for elem in indexed_title])
        if (1. * UKN_count / len(indexed_title)) < UKN_frac_threshold:
            selected_titles.append(indexed_title)
            selected_title_urls.append(all_titles[i][1])
            selected_title_pageView.append(all_titles[i][2])

    print 'total {} titles are included...'.format(len(selected_titles))
    return token_dict, reverse_token_dict, selected_titles, selected_title_urls, selected_title_pageView


def tokenizer_test(data):
    index = random.randint(0, data.shape[0])
    print index
    print data['title'][index]
    print basic_tokenizer(data['title'][index])


def main():
    data_path = '/home/matt.meng'
    #file_name = 'small_articles.json'
    file_name = 'insights_article_data_20170724_20170728.json'
    meta_data_file_name = 'meta_title_data.csv'
    output_pickle_file = 'processed_titles_data.pkl'
    # process the raw JSON file
    title_df = process_raw_data(os.path.join(data_path, file_name))
    print title_df.shape
    title_df.to_csv(os.path.join(data_path, meta_data_file_name), index=True)  # save meta data into .csv file
    # process the .csv file
    data = pd.read_csv(os.path.join(data_path, meta_data_file_name), index_col='url')
    print data.shape
    print 'meta data saved to {}'.format()
    # tokenize the title and create vocabulary dict
    processed_column_name = 'processed_title'
    filtered_data = tokenize_title_column(data, processed_column_name)
    token_dict, reverse_token_dict, titles, selected_title_urls, selected_title_pageView = create_vocab_dict(filtered_data,
                                                                                                             processed_column_name)
    content = {'url': selected_title_urls,
               'titles': titles,
               'pageViw': selected_title_pageView,
               'token_dict': token_dict,
               'reverse_token_dict': reverse_token_dict}

    with open(output_pickle_file, 'wb') as handle:
        cPickle.dump(content, handle, protocol=cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
