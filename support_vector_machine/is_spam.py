from calc_word_indices import calc_word_indices
from email_features import email_features
from get_vocab_list import get_vocab_list
from process_email import process_email
from stem_text import stem_text


def is_spam(email, trained_clf):
    processed_email = process_email(email)
    stemmed_email = stem_text(processed_email)
    dictionary = get_vocab_list('data/vocab.txt')
    x = calc_word_indices(stemmed_email, dictionary)
    email_features_vector = email_features(x, dictionary)
    return trained_clf.predict(email_features_vector.T)[0] == 1
