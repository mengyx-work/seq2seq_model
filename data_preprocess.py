import os, sys, json, time, re
import random, collections, cPickle
import pandas as pd
import cPickle


def isEnglish(s):
    '''verify if a string is in english
    '''
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def convert_text_JSON_to_csv(data_path, csv_path, delimiter='\t'):
    '''convert the raw JSON file (each line is one JSON file)
    into a CSV file
    '''
    start_time = time.time()
    batch_size = 2000
    raw_content = ""
    expected_keys = ['title', 'url', 'traffic', 'publisherId']
    with open(data_path, 'r') as raw_input:
        with open(csv_path, 'w') as raw_output:
            counter = 0
            raw_output.write(delimiter.join(expected_keys) + "\n")
            for line in raw_input:
                json_doc = json.loads(line)
                if not all([key in json_doc.keys() for key in expected_keys]):
                    continue
                elem_list = [json_doc['title'], json_doc['url'], json_doc['traffic'], json_doc['publisherId']]
                if not all([isEnglish(elem) for elem in elem_list]):
                    continue
                raw_content += delimiter.join(elem_list) + "\n"
                counter += 1
                if counter % batch_size == 0:
                    raw_output.write(raw_content)
                    raw_content = ""
                if counter % (20 * batch_size) == 0:
                    print 'finished processing {} rows using {:.2f} seconds'.format(counter, time.time() - start_time)
    print 'finished processing all the data using {:.2f} seconds'.format(time.time() - start_time)


def basic_tokenizer(line, normalize_digits=True):
    '''tokenize a string of words, remove excesive signs
    and split by space
    '''
    line = line.replace("'s", '')
    line = re.sub(r"\'ve", " have ", line)
    line = re.sub(r"can't", "can not ", line)
    line = re.sub(r"n't", " not ", line)
    line = re.sub(r"I'm", "I am", line)
    line = re.sub(r" m ", " am ", line)
    line = re.sub(r"\'re", " are ", line)
    line = re.sub(r"\'d", " would ", line)
    line = re.sub(r"\'ll", " will ", line)
    line = re.sub(r"-", " ? ", line)
    #line = re.sub(r"!", " ! ", line)
    #line = re.sub(r":", " : ", line)
    line = re.sub('[\.,:!;\?"#%\'()*+/;<=>@\[\]^_{|}~\\\]', ' ', line)
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


def tokenize_title_column(data, processed_column_name, pageView_column_name='pageView', title_column_name='title'):
    '''tokenize the title column in DataFrame data,sort by pageView
    and filter the rows by title word counts.
    Args:
        data (DataFrame): original data
        processed_column_name (string): the column name for processed title
        pageView_column_name(string): pageView colum nname
        title_column_name(string): column name in original data
    Returns:
        filtered_data(DataFrame): a selected Pandas DataFrame
    '''

    data['title_word_counts'], data[processed_column_name] = zip(*data[title_column_name].map(basic_tokenizer))
    # sort by the title word counts and filter them
    sorted_data = data.sort_values(by=['title_word_counts', pageView_column_name], ascending=[True, False])
    index = (sorted_data['title_word_counts'] >= 5) & (sorted_data['title_word_counts'] <= 15)
    filtered_data = sorted_data.loc[index, :]
    print 'finish the tokenization...'
    return filtered_data


def process_title_column(data, title_column_name, pageView_column_name):
    '''build a vocabulary dicttionary for all tokenized words and create
    a list of sets of titles
    Args:
        data (DataFrame): the data
        title_column_name (string): the title column name
        pageView_column_name (string): the pageView column name
    Returns:
        all_titles (list): list of sets of titles and relavant elements
        vocab_dict (dict): vocabulary dictionary
    '''
    all_titles, vocab_dict = [], {}
    count, start_time = 0, time.time()
    for title, url, pageView in zip(data[title_column_name], data.index, data[pageView_column_name]):
        words = []
        for word in title.split(' '):
            words.append(word)
            if word not in vocab_dict:
                vocab_dict[word] = 0
            vocab_dict[word] += 1
        all_titles.append((words, url, pageView))
    print 'total {} words are tokenized using {:.2f} second'.format(len(vocab_dict), time.time() - start_time)
    return all_titles, vocab_dict


stop_words = set(['msn', 'breitbart', 'news', 'commentary', 'coverage'])

def process_title_column_by_space(data, title_column_name, pageView_column_name):
    '''function to create the vocabulary dictionary and collect
    the titles according to the selection rules (include only the nourns.)
    '''
    import spacy
    all_titles, vocab_dict = [], {}
    count, start_time = 0, time.time()
    nlp = spacy.load('en')
    for title, url, pageView in zip(data[title_column_name], data.index, data[pageView_column_name]):
        words = []
        title_content = title.decode('ascii')
        doc = nlp(title_content)
        count += 1
        if count % 10000 == 0:
            print 'finish {} using {:.2f} seconds'.format(count, time.time() - start_time)
        if count == 50000:
            break

        for token in doc:
            word = token.lemma_.encode('ascii')
            if word in stop_words or '#' in word:
                continue
            if (token.pos_ == u'NOUN' or token.pos_ == u'PROPN') and not token.is_stop:
                if word not in words:
                    # the title is restricted to contain only only entities
                    # and exlude the duplicate words
                    words.append(word)
                if word not in vocab_dict:
                    vocab_dict[word] = 0
                vocab_dict[word] += 1
        if len(words) > 0:
            all_titles.append((words, url, pageView))
    print 'total {} tokens are identified...'.format(len(vocab_dict))
    return all_titles, vocab_dict


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


def create_selected_vocab_dict(vocab_dict, UKN_index, token_freq_threshold):
    '''process the vocabulary dictionary and use `token_freq_threshold`
    to create a token dictionary
    '''
    token_dict, reverse_token_dict = TOKEN_DICT.copy(), REVERSE_TOKEN_DICT.copy()
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
    return token_dict, reverse_token_dict


def process_title_with_token_dict(all_titles, token_dict, reverse_token_dict, UKN_index, UKN_frac_threshold):
    '''create a dicionary of outputs, including selected titles, URLs, pageViews
    and the assoicated dictionaries.
    titles are filtered by a threshold in the fraction of unkown words.
    Arguments:
        all_titles (list): a list of title sets
        token_dict (dict): the token dict (word -> index)
        reverse_token_dict (dict): the reverse token dict (index -> word)
        UKN_index: the index of <UNK>
        UKN_frac_threshold (double): a threshold to select titles
    '''
    selected_titles = []
    selected_title_urls = []
    selected_title_pageView = []
    for i in xrange(len(all_titles)):
        indexed_title = map(token_dict.get, all_titles[i][0])
        if len(indexed_title) == 0:
            continue
        UKN_count = sum([elem == UKN_index for elem in indexed_title])
        if (1. * UKN_count / len(indexed_title)) < UKN_frac_threshold:
            selected_titles.append(indexed_title)
            selected_title_urls.append(all_titles[i][1])
            selected_title_pageView.append(all_titles[i][2])

    print 'total {} titles are included...'.format(len(selected_titles))
    # return token_dict, reverse_token_dict, selected_titles, selected_title_urls, selected_title_pageView
    return {'url': selected_title_urls,
            'titles': selected_titles,
            'pageViw': selected_title_pageView,
            'token_dict': token_dict,
            'reverse_token_dict': reverse_token_dict}


def create_crambled_training(content, dropout_frac=0.2):
    training_titles = []
    for title in content['titles']:
        title_len = len(title)
        index = random.randrange(title_len-1)
        scrambled_title = title[index:] + title[:index]
        dropout_index = []
        if dropout_frac*title_len > 1.:
            dropout_index = random.sample(xrange(title_len), int(dropout_frac*title_len))
        process_title = [scrambled_title[idx] for idx in xrange(title_len) if idx not in dropout_index]
        training_titles.append(process_title)
    return {'url': content['url'],
            'target_titles': content['titles'],
            'training_titles': training_titles,
            'pageViw': content['pageViw'],
            'token_dict': content['token_dict'],
            'reverse_token_dict': content['reverse_token_dict']}

def main():
    data_path = '/Users/matt.meng'
    file_name = 'insights_article_data_title_only_20170719_20170728.json'
    meta_data_file_name = 'meta_title_data.csv'
    #output_pickle_file = 'processed_titles_data.pkl'
    output_pickle_file = 'scramble_titles_data.pkl'
    delimiter = '\t\t'

    '''
    convert_text_JSON_to_csv(os.path.join(data_path, file_name),
                             os.path.join(data_path, meta_data_file_name),
                             delimiter)
    '''

    data = pd.read_csv(os.path.join(data_path, meta_data_file_name), index_col='url', delimiter=delimiter,
                       encoding='utf-8')

    data.dropna(how='any', inplace=True)
    data['publisherId'] = data['publisherId'].astype(int).astype(str)
    valid_publisher_ids = ['1001082', '1023406', '1003264', '1040522', '782', '1006541',
                           '1168', '1038583', '1021516', '580', '1020689', '1031851', '1001264',
                           '1039208', '1054980', '1018671', '1031841', '1031842', '1031852',
                           '1008941', '1003764', '1068057', '1038711', '1002628', '1031853',
                           '1021578', '1043813', '1010748', '1040526', '1005092', '612',
                           '1003870', '1001156', '1012083', '1017946', '1041479', '1027016',
                           '1010488', '1017947', '1010497', '1038582', '1045821', '1020968',
                           '1037842', '1029984', '723', '196', '1030941']

    filtered_data = data.loc[data['publisherId'].isin(valid_publisher_ids), :]
    unique_filtered_data = filtered_data[~filtered_data.index.duplicated(keep='first')]
    print unique_filtered_data.shape

    processed_column_name = 'processed_title'
    pageView_column_name = 'traffic'
    filtered_data = tokenize_title_column(unique_filtered_data, processed_column_name, pageView_column_name)
    all_titles, vocab_dict = process_title_column(filtered_data, 'processed_title', 'traffic')
    UKN_index = len(TOKEN_DICT) - 1
    token_dict, reverse_token_dict = create_selected_vocab_dict(vocab_dict, UKN_index, token_freq_threshold=4)
    #print "the selected token size: {}".format(len(token_dict.keys()))
    selected_content = process_title_with_token_dict(all_titles, token_dict, reverse_token_dict, UKN_index, UKN_frac_threshold=0.3)

    processed_content = create_crambled_training(selected_content)
    #print processed_content.keys()
    with open(os.path.join(data_path, output_pickle_file), 'wb') as handle:
        cPickle.dump(processed_content, handle, protocol=cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
